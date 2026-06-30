#!/usr/bin/env python3
"""Re-export ONE chunk (pro / dbl{i} / sgl{j} / epi) from the BINARY q1_0 Bonsai gguf at the
512^2 TARGET shape (img_seq=1024, txt_seq=512, merged S=1536). Standalone per-chunk weight
load (NO 15.5 GB full model -> safe on the shared box) + synthetic right-shaped calib (the
recipe overrides acts to fp16, so calib VALUES are irrelevant -- only shapes matter; weight
int8 scales come from the weights). NO PTQ: exact q1_0 weights.
Usage: export_one_q1.py <spec>  ->  q1chunks/<spec>_q1_fp32.tflite + q1chunks/cal_<spec>.txt"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, litert_torch
os.environ.setdefault("BONSAI_GGUF", os.path.expanduser("~/bonsai-export/bonsai_image_4b-q1_0.gguf"))
from real_dit import parse_gguf, deq, GGUF
from dit_block import KleinDoubleStreamBlock, make_inputs as dmk
from single_block import KleinSingleStreamBlock, make_inputs as smk

IMG, TXT, S = 1024, 512, 1536
spec = sys.argv[1]
OUT = os.path.expanduser("~/bonsai-export/q1chunks"); os.makedirs(OUT, exist_ok=True)
infos, ds, blob = parse_gguf(GGUF)
torch.manual_seed(0)


def W(name):
    dims, _, _ = infos[name]; f = deq(blob, ds, infos[name])
    return torch.from_numpy(f.reshape(dims[1], dims[0]).copy()) if len(dims) == 2 else torch.from_numpy(f.copy())


def lin(name):
    w = W(name); m = nn.Linear(w.shape[1], w.shape[0], bias=False); m.weight.data = w; return m


def writecalib(tag, args):                          # args: positional tuple -> serving_default_args_{i}
    parts = []
    for qi, arr in enumerate(args):
        a = arr.numpy() if torch.is_tensor(arr) else arr
        p = f"{OUT}/cal_{tag}_a{qi}.raw"; np.ascontiguousarray(a, dtype=np.float32).tofile(p)
        parts.append(f"serving_default_args_{qi}:={p}")
    open(f"{OUT}/cal_{tag}.txt", "w").write(" ".join(parts) + "\n")


if spec == "pro":
    class P(nn.Module):
        def __init__(s):
            super().__init__()
            s.img_in = lin("img_in.weight")
            Wt = W("txt_in.weight")                  # [3072, 7680]
            s.txt0 = nn.Linear(2560, 3072, bias=False); s.txt0.weight.data = Wt[:, 0:2560].clone()
            s.txt1 = nn.Linear(2560, 3072, bias=False); s.txt1.weight.data = Wt[:, 2560:5120].clone()
            s.txt2 = nn.Linear(2560, 3072, bias=False); s.txt2.weight.data = Wt[:, 5120:7680].clone()
            s.tin = lin("time_in.in_layer.weight")
            s.tout = lin("time_in.out_layer.weight")
            s.dmi = lin("double_stream_modulation_img.lin.weight")
            s.dmt = lin("double_stream_modulation_txt.lin.weight")
            s.sm = lin("single_stream_modulation.lin.weight")

        def forward(s, latent, context, temb):
            img = s.img_in(latent)
            c0, c1, c2 = context.split(2560, dim=-1)
            txt = s.txt0(c0) + s.txt1(c1) + s.txt2(c2)
            vec = s.tout(F.silu(s.tin(temb))); sv = F.silu(vec)
            return img, txt, s.dmi(sv), s.dmt(sv), s.sm(sv), sv
    m = P().eval()
    args = (torch.randn(1, IMG, 128), torch.randn(1, TXT, 7680), torch.randn(1, 256))

elif spec.startswith("dbl"):
    p = f"double_blocks.{int(spec[3:])}."
    m = KleinDoubleStreamBlock().eval()
    m.load_state_dict({
        "img_qkv.weight": W(p + "img_attn.qkv.weight"), "img_proj.weight": W(p + "img_attn.proj.weight"),
        "img_qn": W(p + "img_attn.norm.query_norm.scale"), "img_kn": W(p + "img_attn.norm.key_norm.scale"),
        "img_mlp_in.weight": W(p + "img_mlp.0.weight"), "img_mlp_out.weight": W(p + "img_mlp.2.weight"),
        "txt_qkv.weight": W(p + "txt_attn.qkv.weight"), "txt_proj.weight": W(p + "txt_attn.proj.weight"),
        "txt_qn": W(p + "txt_attn.norm.query_norm.scale"), "txt_kn": W(p + "txt_attn.norm.key_norm.scale"),
        "txt_mlp_in.weight": W(p + "txt_mlp.0.weight"), "txt_mlp_out.weight": W(p + "txt_mlp.2.weight"),
    }, strict=False)
    args = dmk(img_seq=IMG, txt_seq=TXT)

elif spec.startswith("sgl"):
    p = f"single_blocks.{int(spec[3:])}."
    m = KleinSingleStreamBlock().eval()
    m.load_state_dict({
        "linear1.weight": W(p + "linear1.weight"), "linear2.weight": W(p + "linear2.weight"),
        "qn": W(p + "norm.query_norm.scale"), "kn": W(p + "norm.key_norm.scale"),
    }, strict=False)
    args = smk(S=S)

elif spec == "epi":
    from seeded_dit import Epilogue
    holder = type("H", (), {})()
    holder.finalmod = lin("final_layer.adaLN_modulation.1.weight")
    holder.final_lin = lin("final_layer.linear.weight")
    m = Epilogue(holder).eval()
    args = (torch.randn(1, IMG, 3072), torch.randn(1, 3072))

else:
    raise SystemExit(f"unknown spec {spec}")

writecalib(spec, args)
litert_torch.convert(m.eval(), sample_args=args).export(f"{OUT}/{spec}_q1_fp32.tflite")
print(f"exported {spec}_q1_fp32.tflite {os.path.getsize(OUT + f'/{spec}_q1_fp32.tflite') / 1e6:.0f}MB", flush=True)
