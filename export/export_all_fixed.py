#!/usr/bin/env python3
"""Re-export the graphs that the LayerNorm fix changed: 5 doubles + 20 singles + epilogue,
with the fixed dit_block/single_block/seeded_dit. Does NOT touch cal_* (keeps the real
conditioning calib) and does NOT re-export the prologue (no layernorm -> unchanged)."""
import os, time, torch, litert_torch
from dit_block import D, make_inputs as dmk
from single_block import make_inputs as smk
from seeded_dit import Epilogue
from real_dit import build_real_dit

dit = build_real_dit()
epi = Epilogue(dit).eval()


def export(m, args, path):
    t = time.time()
    litert_torch.convert(m, sample_args=args).export(path)
    print(f"  {path}: {os.path.getsize(path)/1e6:.0f}MB {time.time()-t:.0f}s", flush=True)


for i in range(5):
    export(dit.doubles[i].eval(), dmk(img_seq=256, txt_seq=64), f"dbl{i}_fixed_fp32.tflite")
for j in range(20):
    export(dit.singles[j].eval(), smk(), f"sgl{j}_fixed_fp32.tflite")
export(epi, (torch.randn(1, 256, D), torch.randn(1, D)), "epi_fixed_fp32.tflite")
print("DONE all 26 fixed fp32 tflites", flush=True)
