#!/usr/bin/env python3
"""Seeded, weight-tieable KleinDiT split into prologue / double / single / epilogue
sub-graphs that SHARE the parent model's weights. Lets the chunked-DiT orchestration
be validated end-to-end against the parent torch model (identical weights).

tie=True ties all 5 doubles to doubles[0] and all 20 singles to singles[0], so the
whole DiT is covered by just 4 distinct graphs (prologue, one double, one single,
epilogue) -> 4 conversions instead of 27, while still exercising the full chain
(activations differ per position; only weights are shared)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dit_block import D, build_rope, layer_norm, KleinDoubleStreamBlock          # noqa: F401
from single_block import KleinSingleStreamBlock                       # noqa: F401
from flux_full import KleinDiT, IMG_SEQ, CTX_DIM, PATCH_IN, TE_DIM

TXT_SEQ = 64
S = TXT_SEQ + IMG_SEQ          # 320 merged tokens


def build_dit(seed=0, tie=True):
    torch.manual_seed(seed)
    dit = KleinDiT(5, 20).eval()
    if tie:
        sd = {k: v.clone() for k, v in dit.doubles[0].state_dict().items()}
        for d in dit.doubles[1:]:
            d.load_state_dict(sd)
        ss = {k: v.clone() for k, v in dit.singles[0].state_dict().items()}
        for s in dit.singles[1:]:
            s.load_state_dict(ss)
    return dit


class Prologue(nn.Module):
    """latent,context,t_emb -> img, txt, img_mod, txt_mod, smod, sv  (outputs 0..5).

    txt_in is split into 3 matmuls of K=2560 (the 3 concatenated Qwen layers) summed,
    because the single K=7680 matmul faults at execution on HTP V79 (an HTP tiling
    quirk specific to 7680 — K=12288 in the blocks runs fine). The 3 sub-weights are
    column blocks of the canonical txt_in weight, so this is numerically identical."""
    def __init__(self, dit):
        super().__init__()
        self.img_in = dit.img_in
        W = dit.txt_in.weight.data                       # [3072, 7680]
        self.txt0 = nn.Linear(2560, 3072, bias=False)
        self.txt1 = nn.Linear(2560, 3072, bias=False)
        self.txt2 = nn.Linear(2560, 3072, bias=False)
        self.txt0.weight.data = W[:, 0:2560].clone()
        self.txt1.weight.data = W[:, 2560:5120].clone()
        self.txt2.weight.data = W[:, 5120:7680].clone()
        self.time_in_in = dit.time_in_in
        self.time_in_out = dit.time_in_out
        self.dmod_img = dit.dmod_img
        self.dmod_txt = dit.dmod_txt
        self.smod = dit.smod

    def forward(self, latent, context, t_emb):
        img = self.img_in(latent)
        c0, c1, c2 = context.split(2560, dim=-1)
        txt = self.txt0(c0) + self.txt1(c1) + self.txt2(c2)
        vec = self.time_in_out(F.silu(self.time_in_in(t_emb)))
        sv = F.silu(vec)
        return img, txt, self.dmod_img(sv), self.dmod_txt(sv), self.smod(sv), sv


class Epilogue(nn.Module):
    """img_tokens[1,256,D], sv[1,D] -> out[1,256,128]."""
    def __init__(self, dit):
        super().__init__()
        self.finalmod = dit.finalmod
        self.final_lin = dit.final_lin

    def forward(self, img, sv):
        fsh, fsc = self.finalmod(sv).chunk(2, -1)
        img = layer_norm(img) * (1 + fsc.unsqueeze(1)) + fsh.unsqueeze(1)
        return self.final_lin(img)


def val_inputs(seed=0):
    """Fixed validation inputs at txt_seq=64 (matches the shape-specialized binaries)."""
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(1, IMG_SEQ, PATCH_IN, generator=g)
    context = torch.randn(1, TXT_SEQ, CTX_DIM, generator=g)
    t_emb = torch.randn(1, TE_DIM, generator=g)
    cos, sin = build_rope(S)
    return latent, context, t_emb, cos, sin
