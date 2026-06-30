#!/usr/bin/env python3
"""FLUX.2 Klein 4B DOUBLE-STREAM DiT block at real dims, HTP-friendly, for
litert_torch.convert -> .tflite -> QNN/HTP delegate on the Hexagon V79 (S25+).

Spike goal: measure whether the Hexagon NPU (HMX matrix engine) beats the
matrix-core-less Adreno 830 (~7 s/step @ 256^2) on the Klein DiT.

Klein-faithful (cross-validated by the scoping pass against bonsai-cpp/src/flux.hpp,
bonsai-swift KleinTransformer, f2_from_diffusers.py):
  D=3072, H=24, head_dim=128, MLP=9216 (mlp_ratio 3.0), theta=2000, NO biases,
  LayerNorm(elementwise_affine=False) + AdaLN modulate x*(1+scale)+shift,
  qk-norm = RMSNorm over head_dim on q,k only, INTERLEAVED RoPE, SwiGLU (gate_first),
  JOINT attention over concat([txt,img]); 5 double + 20 single blocks total.

HTP-friendly fixes folded in (from the ops scout):
  (a) attention DECOMPOSED into matmul->scale->softmax->matmul (no fused SDPA,
      the #1 CPU-fallback risk on the QNN delegate),
  (b) qkv reshaped rank-4 (split + view + transpose; no rank-5 permute),
  (c) cos/sin + shared distilled modulation passed as INPUTS (constants for HTP),
  (d) fp16-safe RMSNorm (1/256 prescale) so the fp16-act path can't NaN.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 3072
H = 24
DH = D // H          # 128
MLP = 9216           # mlp_ratio 3.0
THETA = 2000.0
TXT_SEQ = 64
IMG_SEQ = 256
SCALE = 1.0 / math.sqrt(DH)
_P = 1.0 / 256.0     # RMSNorm prescale (fp16-safe: avoid x^2 overflow on outliers)
_LNP = 1.0 / 4096.0  # LayerNorm prescale. By the last step the residual stream hits max ~41000;
                     # at 1/256 the variance SUM(x*P)^2 ~ 68482 STILL overflows fp16 (65504) on the
                     # V79 (fp16 reduction). 1/4096 -> sum ~268 (244x margin); cancels in normalize.


def build_rope(seq):
    """Interleaved FLUX RoPE tables: cos/sin [seq, DH//2]."""
    half = DH // 2
    pos = torch.arange(seq, dtype=torch.float32)
    freq = 1.0 / (THETA ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = torch.outer(pos, freq)                 # [seq, 64]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: [B,H,S,DH] interleaved (re,im) pairs; cos/sin: [S, DH//2]."""
    re = x[..., 0::2]                            # [B,H,S,DH//2]
    im = x[..., 1::2]
    c = cos.view(1, 1, cos.shape[0], cos.shape[1])
    s = sin.view(1, 1, sin.shape[0], sin.shape[1])
    out_re = re * c - im * s
    out_im = im * c + re * s
    return torch.stack([out_re, out_im], dim=-1).reshape(x.shape)


def rms_norm(x, weight, eps=1e-6):
    """fp16-safe RMSNorm over the last dim. inv = p*rsqrt(mean((x*p)^2)+eps*p^2)
    == rsqrt(mean(x^2)+eps) but the square never overflows fp16."""
    xs = x * _P
    ms = xs.pow(2).mean(-1, keepdim=True)
    inv = _P * torch.rsqrt(ms + eps * _P * _P)
    return x * inv * weight


def layer_norm(x, eps=1e-5):
    """fp16-safe LayerNorm over the last dim (affine=False). Mathematically equal to
    F.layer_norm(x,(D,)) in fp32, but prescaling by _LNP (1/4096) keeps the variance SUM from
    overflowing fp16: by the last step the residual stream hits max ~41000 -> at 1/256 the
    SUM(x*P)^2 ~ 68482 STILL overflows fp16 (65504) on the HTP (fp16 reduction, unlike torch
    fp32-accum), NaN'ing the chain at step 3. 1/4096 -> sum ~268; cancels exactly in normalize.
    Verified on V79: LN-fixed single on the step-3 overflow input = finite, cos 0.9999."""
    xs = x * _LNP
    mu = xs.mean(-1, keepdim=True)
    xc = xs - mu
    var = xc.pow(2).mean(-1, keepdim=True)
    return xc * torch.rsqrt(var + eps * _LNP * _LNP)


def _heads(t, S):
    """[B,S,D] -> [B,H,S,DH] (rank-4, no rank-5 permute)."""
    return t.view(t.shape[0], S, H, DH).transpose(1, 2)


class KleinDoubleStreamBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # img stream
        self.img_qkv = nn.Linear(D, 3 * D, bias=False)
        self.img_proj = nn.Linear(D, D, bias=False)
        self.img_qn = nn.Parameter(torch.ones(DH))
        self.img_kn = nn.Parameter(torch.ones(DH))
        self.img_mlp_in = nn.Linear(D, 2 * MLP, bias=False)
        self.img_mlp_out = nn.Linear(MLP, D, bias=False)
        # txt stream
        self.txt_qkv = nn.Linear(D, 3 * D, bias=False)
        self.txt_proj = nn.Linear(D, D, bias=False)
        self.txt_qn = nn.Parameter(torch.ones(DH))
        self.txt_kn = nn.Parameter(torch.ones(DH))
        self.txt_mlp_in = nn.Linear(D, 2 * MLP, bias=False)
        self.txt_mlp_out = nn.Linear(MLP, D, bias=False)

    def _swiglu(self, x, lin_in, lin_out):
        gate, up = lin_in(x).chunk(2, dim=-1)     # gate_first
        return lin_out(F.silu(gate) * up)

    def forward(self, img, txt, cos, sin, img_mod, txt_mod):
        # img [B,Si,D]  txt [B,St,D]  cos/sin [St+Si, DH//2]  img/txt_mod [B,6D]
        B, Si, _ = img.shape
        St = txt.shape[1]
        i_sh1, i_sc1, i_g1, i_sh2, i_sc2, i_g2 = (t.unsqueeze(1) for t in img_mod.chunk(6, -1))
        t_sh1, t_sc1, t_g1, t_sh2, t_sc2, t_g2 = (t.unsqueeze(1) for t in txt_mod.chunk(6, -1))

        # --- joint attention ---
        img_n = layer_norm(img) * (1 + i_sc1) + i_sh1
        txt_n = layer_norm(txt) * (1 + t_sc1) + t_sh1
        iq, ik, iv = self.img_qkv(img_n).split(D, dim=-1)
        tq, tk, tv = self.txt_qkv(txt_n).split(D, dim=-1)
        iq, ik, iv = _heads(iq, Si), _heads(ik, Si), _heads(iv, Si)
        tq, tk, tv = _heads(tq, St), _heads(tk, St), _heads(tv, St)
        iq, ik = rms_norm(iq, self.img_qn), rms_norm(ik, self.img_kn)
        tq, tk = rms_norm(tq, self.txt_qn), rms_norm(tk, self.txt_kn)

        q = apply_rope(torch.cat([tq, iq], dim=2), cos, sin)   # [B,H,St+Si,DH]
        k = apply_rope(torch.cat([tk, ik], dim=2), cos, sin)
        v = torch.cat([tv, iv], dim=2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * SCALE  # [B,H,S,S]
        attn = logits.softmax(dim=-1)                          # HTP fp16 softmax is max-stable (verified)
        out = torch.matmul(attn, v)                            # [B,H,S,DH]
        out = out.transpose(1, 2).reshape(B, St + Si, D)
        txt_a, img_a = out[:, :St], out[:, St:]

        img = img + i_g1 * self.img_proj(img_a)
        txt = txt + t_g1 * self.txt_proj(txt_a)

        # --- SwiGLU MLP ---
        img_m = layer_norm(img) * (1 + i_sc2) + i_sh2
        txt_m = layer_norm(txt) * (1 + t_sc2) + t_sh2
        img = img + i_g2 * self._swiglu(img_m, self.img_mlp_in, self.img_mlp_out)
        txt = txt + t_g2 * self._swiglu(txt_m, self.txt_mlp_in, self.txt_mlp_out)
        return img, txt


def make_inputs(img_seq=IMG_SEQ, txt_seq=TXT_SEQ, B=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(B, img_seq, D, generator=g)
    txt = torch.randn(B, txt_seq, D, generator=g)
    cos, sin = build_rope(txt_seq + img_seq)
    img_mod = torch.randn(B, 6 * D, generator=g) * 0.1
    txt_mod = torch.randn(B, 6 * D, generator=g) * 0.1
    return (img, txt, cos, sin, img_mod, txt_mod)


if __name__ == "__main__":
    m = KleinDoubleStreamBlock().eval()
    nparams = sum(p.numel() for p in m.parameters())
    print(f"params: {nparams/1e6:.1f} M")
    with torch.no_grad():
        for s in (256, 1024):
            oi, ot = m(*make_inputs(img_seq=s))
            assert oi.shape == (1, s, D) and ot.shape == (1, TXT_SEQ, D), (oi.shape, ot.shape)
            print(f"img_seq={s}: img_out={tuple(oi.shape)} txt_out={tuple(ot.shape)} OK")
    ep = torch.export.export(m, make_inputs(256))
    nnodes = sum(1 for _ in ep.graph.nodes)
    print(f"torch.export OK ({nnodes} graph nodes)")
