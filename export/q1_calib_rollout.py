#!/usr/bin/env python3
"""Multi-prompt calibration capture for prompt-ROBUST int16 ranges. For each diverse prompt's
context (calibctx/ctx_*.raw from sd.cpp), run the torch q1_0 flux 4-step rollout on the GPU and
dump every chunk's per-step INPUT activations at DUMP_STEPS. The converter's --input_list then
sets each int16 activation range as the max over ALL (prompts x steps) -> robust for any prompt.
cos/sin/pe are prompt-independent (shared, dumped once). Writes calibsamples/ + cal_<chunk>.txt.
Run on the box GPU (full GPU is free): DEVICE=cuda, ~fast per prompt."""
import os, glob, math, numpy as np, torch
os.environ.setdefault("BONSAI_GGUF", os.path.expanduser("~/bonsai-export/bonsai_image_4b-q1_0.gguf"))
from real_dit import build_real_dit
from seeded_dit import Prologue, Epilogue

EXP = os.path.expanduser("~/bonsai-export"); OUT = f"{EXP}/calibsamples"; os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("DEVICE", "cuda")
DUMP_STEPS = {int(s) for s in os.environ.get("DUMP_STEPS", "0 3").split()}   # capture step extremes


def flux_sigmas(n, L):
    a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
    if L > 4300: mu = a2 * L + b2
    else: m2, m1 = a2 * L + b2, a1 * L + b1; a = (m2 - m1) / 190.0; mu = a * n + (m2 - 200.0 * a)
    em = math.exp(mu); return [em / (em + (1.0 / (1.0 - (1.0 - 1.0 / n) * i / (n - 1)) - 1.0)) for i in range(n)] + [0.0]
def temb_of(sg):
    t = sg * 1000.0; f = np.exp(-math.log(10000) * np.arange(128) / 128); a = t * f
    return np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)
def patchA(raw):
    hw = int(round((raw.size / 128) ** 0.5)); return raw.reshape(128, hw, hw).transpose(1, 2, 0).reshape(hw * hw, 128).astype(np.float32)


SIG = flux_sigmas(4, 1024)
pe = np.fromfile(f"{EXP}/q1dump512_s7/pe.raw", np.float32).reshape(1536, 64, 2, 2)
cos = torch.from_numpy(pe[:, :, 0, 0].copy()).to(DEV); sin = torch.from_numpy(pe[:, :, 1, 0].copy()).to(DEV)
COS = f"{OUT}/cos.raw"; SIN = f"{OUT}/sin.raw"
np.ascontiguousarray(pe[:, :, 0, 0].copy(), np.float32).tofile(COS)
np.ascontiguousarray(pe[:, :, 1, 0].copy(), np.float32).tofile(SIN)

dit = build_real_dit(device=DEV, dtype=torch.float32).eval()
pro = Prologue(dit).eval(); epi = Epilogue(dit).eval()
ctxs = sorted(glob.glob(f"{EXP}/calibctx/ctx_*.raw"))
print(f"{len(ctxs)} prompts, dump steps {sorted(DUMP_STEPS)}", flush=True)

lines = {"pro": [], "epi": [], **{f"dbl{j}": [] for j in range(5)}, **{f"sgl{j}": [] for j in range(20)}}
A = lambda i: f"serving_default_args_{i}"
def dump(sub, nm, t):
    d = f"{OUT}/{sub}"; os.makedirs(d, exist_ok=True); p = f"{d}/{nm}.raw"
    np.ascontiguousarray(t.detach().float().cpu().numpy(), np.float32).tofile(p); return p

for pi, cf in enumerate(ctxs):
    ctx = torch.from_numpy(np.fromfile(cf, np.float32).reshape(1, 512, 7680)).to(DEV)
    x = torch.from_numpy(patchA(np.random.default_rng(1000 + pi).standard_normal(128 * 1024).astype(np.float32))).to(DEV)  # [1024,128]
    for s in range(4):
        temb = torch.from_numpy(temb_of(SIG[s])[None]).to(DEV)
        with torch.no_grad():
            img, txt, imgmod, txtmod, smod, sv = pro(x[None], ctx, temb)
            if s in DUMP_STEPS:
                tg = f"p{pi}s{s}"
                lp = dump("pro", f"{tg}_lat", x[None]); tp = dump("pro", f"{tg}_temb", temb)
                lines["pro"].append(f"{A(0)}:={lp} {A(1)}:={cf} {A(2)}:={tp}")
                imP = dump("mods", f"{tg}_imgmod", imgmod); txP = dump("mods", f"{tg}_txtmod", txtmod)
                smP = dump("mods", f"{tg}_smod", smod); svP = dump("mods", f"{tg}_sv", sv)
            i2, t2 = img, txt
            for j in range(5):
                if s in DUMP_STEPS:
                    ip = dump(f"dbl{j}", f"{tg}_img", i2); tp2 = dump(f"dbl{j}", f"{tg}_txt", t2)
                    lines[f"dbl{j}"].append(f"{A(0)}:={ip} {A(1)}:={tp2} {A(2)}:={COS} {A(3)}:={SIN} {A(4)}:={imP} {A(5)}:={txP}")
                i2, t2 = dit.doubles[j](i2, t2, cos, sin, imgmod, txtmod)
            xm = torch.cat([t2, i2], dim=1)
            for j in range(20):
                if s in DUMP_STEPS:
                    xp = dump(f"sgl{j}", f"{tg}_x", xm)
                    lines[f"sgl{j}"].append(f"{A(0)}:={xp} {A(1)}:={COS} {A(2)}:={SIN} {A(3)}:={smP}")
                xm = dit.singles[j](xm, cos, sin, smod)
            if s in DUMP_STEPS:
                ep = dump("epi", f"{tg}_imgtok", xm[:, 512:])
                lines["epi"].append(f"{A(0)}:={ep} {A(1)}:={svP}")
            vel = epi(xm[:, 512:], sv)
            x = x + (SIG[s + 1] - SIG[s]) * vel[0]
    print(f"prompt {pi} done", flush=True)

for ch, ls in lines.items():
    open(f"{OUT}/cal_{ch}.txt", "w").write("\n".join(ls) + "\n")
print(f"calib lists written ({len(ctxs) * len(DUMP_STEPS)} samples/chunk)", flush=True)
