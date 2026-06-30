#!/usr/bin/env python3
"""FLUX.2 Klein SINGLE-STREAM DiT block (20 of them) at real dims, for the NPU spike.
Parallel attention+MLP topology: ONE fused linear1 = Linear(D, 3D + 2*MLP) producing
qkv AND the SwiGLU gate/up; qk-norm + RoPE + decomposed attention on the qkv slice;
silu-gated mlp slice; concat([attn, swiglu]) -> linear2(D+MLP -> D); single 3-way
modulation (shift/scale/gate). Runs on the MERGED [txt,img] stream (S=320 @ 256^2).
Reuses the HTP-friendly helpers from dit_block.py (decomposed attn, rank-4 qkv,
precomputed mod, fp16-safe RMSNorm)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dit_block import D, H, DH, MLP, SCALE, build_rope, apply_rope, rms_norm, _heads

S_TXT, S_IMG = 64, 256
S = S_TXT + S_IMG     # 320 merged tokens @ 256^2
_LNP = 1.0 / 4096.0   # fp16-safe LayerNorm prescale. The residual stream hits max ~41000 by the
                      # last step; raw F.layer_norm's variance SUM(x^2) overflows fp16 (65504) on
                      # the NPU (which reduces in fp16, unlike torch/GPU fp32-accum). Prescaling x
                      # by _LNP makes the sum tiny (~268) and CANCELS exactly in the normalize, so
                      # this is bit-identical to F.layer_norm in fp32 but overflow-proof in fp16.


def layer_norm_safe(x, eps=1e-6):
    xs = x * _LNP
    mu = xs.mean(-1, keepdim=True)
    var = (xs - mu).pow(2).mean(-1, keepdim=True)
    return (xs - mu) * torch.rsqrt(var + eps * _LNP * _LNP)


class KleinSingleStreamBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(D, 3 * D + 2 * MLP, bias=False)   # qkv + swiglu gate/up
        self.linear2 = nn.Linear(D + MLP, D, bias=False)
        self.qn = nn.Parameter(torch.ones(DH))
        self.kn = nn.Parameter(torch.ones(DH))

    def forward(self, x, cos, sin, mod):
        # x [B,S,D]  cos/sin [S,DH//2]  mod [B,3D] (shift/scale/gate)
        #
        # The per-token elementwise ops (LayerNorm, AdaLN affine, SwiGLU, residual) run on the
        # MERGED stream [B,S,D]. At 512^2 (S=1536) that 3D tensor is 1536x3072x2 = 9.4MB > the V79
        # 8MB VTCM, and QNN 2.41's HTP tiler can't tile a 3D [S,D] InputSlice over S (it demands the
        # whole 9.4MB: "q::*InputSlice not sufficiently tiled", err 17). The double-stream blocks
        # never hit this -- their per-token ops run on the SEPARATE img/txt streams (<=1024 tokens,
        # <=6.3MB, fit VTCM whole), and their only 9.4MB tensors are 4D [B,H,S,DH] (rope), which
        # tile trivially over the 24-head dim. So: split the merged stream into <=1024-length halves
        # for the per-token ops, and merge only into the 4D head layout for attention. Per-token ops
        # are token-independent, so an arbitrary even split is BIT-IDENTICAL to the merged compute.
        B, Sx, _ = x.shape
        sh, sc, g = (t.unsqueeze(1) for t in mod.chunk(3, -1))
        H2 = Sx // 2
        halves = (x[:, :H2], x[:, H2:])
        qs, ks, vs, mlps = [], [], [], []
        for xp in halves:
            Sp = xp.shape[1]
            xn = layer_norm_safe(xp) * (1 + sc) + sh
            qkv, mlp = self.linear1(xn).split([3 * D, 2 * MLP], dim=-1)
            q, k, v = qkv.split(D, dim=-1)
            qs.append(_heads(q, Sp)); ks.append(_heads(k, Sp)); vs.append(_heads(v, Sp))
            mlps.append(mlp)
        q = torch.cat(qs, dim=2); k = torch.cat(ks, dim=2); v = torch.cat(vs, dim=2)  # [B,H,Sx,DH]
        q, k = rms_norm(q, self.qn), rms_norm(k, self.kn)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        logits = torch.matmul(q, k.transpose(-2, -1)) * SCALE                         # [B,H,Sx,Sx]
        out = torch.matmul(logits.softmax(-1), v).transpose(1, 2).reshape(B, Sx, D)   # [B,Sx,D]
        out_h = (out[:, :H2], out[:, H2:])
        res = []
        for hi, xp in enumerate(halves):
            gate, up = mlps[hi].chunk(2, dim=-1)
            cat = torch.cat([out_h[hi], F.silu(gate) * up], dim=-1)                    # [B,H2,D+MLP]
            res.append(xp + g * self.linear2(cat))
        return torch.cat(res, dim=1)


def make_inputs(S=S, B=1, seed=0):
    gg = torch.Generator().manual_seed(seed)
    x = torch.randn(B, S, D, generator=gg)
    cos, sin = build_rope(S)
    mod = torch.randn(B, 3 * D, generator=gg) * 0.1
    return (x, cos, sin, mod)


if __name__ == "__main__":
    m = KleinSingleStreamBlock().eval()
    print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.1f} M")
    with torch.no_grad():
        o = m(*make_inputs())
        assert o.shape == (1, S, D), o.shape
        print(f"S={S}: out={tuple(o.shape)} OK")
    ep = torch.export.export(m, make_inputs())
    print(f"torch.export OK ({sum(1 for _ in ep.graph.nodes)} graph nodes)")
