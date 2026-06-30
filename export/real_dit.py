#!/usr/bin/env python3
"""Load REAL FLUX.2 Klein 4B weights from flux-2-klein-4b-Q4_0.gguf into a full
KleinDiT(5,20) — all 27 sub-modules (img_in, txt_in, time_in, dmod_img/txt, smod,
5 doubles, 20 singles, finalmod, final_lin). Reuses the gguf parser/dequant from
quality_real_weights.py. gguf dims are [in,out] (row-major) == torch Linear [out,in]."""
import os
import struct
import numpy as np
import torch
from flux_full import KleinDiT

# Default = the (vanilla, Q4_0) Klein gguf on the box. Override with BONSAI_GGUF to
# load the BINARY q1_0 Bonsai model (bonsai_image_4b-q1_0.gguf) — no PTQ, exact weights.
GGUF = os.environ.get("BONSAI_GGUF", os.path.expanduser("~/bonsai-export/flux-2-klein-4b-Q4_0.gguf"))


def parse_gguf(path):
    f = open(path, "rb")
    def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
    def rstr(): return f.read(rd("Q")).decode("utf-8", "replace")
    SIZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    assert f.read(4) == b"GGUF"; rd("I"); nt = rd("Q"); nkv = rd("Q")
    align = 32
    for _ in range(nkv):
        k = rstr(); vt = rd("I")
        if vt == 9:
            et = rd("I"); cnt = rd("Q")
            for _ in range(cnt):
                rstr() if et == 8 else f.read(SIZ[et])
        else:
            v = rstr() if vt == 8 else struct.unpack(
                "<" + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[vt],
                f.read(SIZ[vt]))[0]
            if k == "general.alignment": align = v
    infos = {}
    for _ in range(nt):
        name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; typ = rd("I"); off = rd("Q")
        infos[name] = (dims, typ, off)
    pos = f.tell(); data_start = (pos + align - 1) // align * align
    f.seek(0); blob = f.read(); f.close()
    return infos, data_start, blob


def deq(blob, ds, info):
    dims, typ, off = info
    ne = int(np.prod(dims)); base = ds + off
    if typ == 0:
        return np.frombuffer(blob, np.float32, count=ne, offset=base).copy()
    if typ == 30:  # bf16
        u = np.frombuffer(blob, np.uint16, count=ne, offset=base).astype(np.uint32) << 16
        return u.view(np.float32).copy()
    if typ == 2:   # q4_0
        nb = ne // 32
        a = np.frombuffer(blob, np.uint8, count=nb * 18, offset=base).reshape(nb, 18)
        d = a[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
        qs = a[:, 2:]
        lo = (qs & 0x0F).astype(np.int32) - 8; hi = (qs >> 4).astype(np.int32) - 8
        out = np.empty((nb, 32), np.float32); out[:, :16] = lo * d; out[:, 16:] = hi * d
        return out.reshape(-1)
    if typ == 41:  # q1_0 (BINARY): block=128, fp16 d=mean(|x|)/block, w = bit?+d:-d
        nb = ne // 128  # ggml-quants.c:377 dequantize_row_q1_0; bit j -> byte j//8, bit j%8 (LSB-first)
        a = np.frombuffer(blob, np.uint8, count=nb * 18, offset=base).reshape(nb, 18)
        d = a[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
        bits = np.unpackbits(a[:, 2:], axis=1, bitorder="little")  # (nb,128)
        return (np.where(bits == 1, 1.0, -1.0).astype(np.float32) * d).reshape(-1)
    raise ValueError(f"ggml type {typ}")


def build_real_dit(gguf=GGUF, device="cpu", dtype=torch.float32):
    """Load real weights IN PLACE (no full state_dict copy) so peak ~= model size, not 2x.
    fp32 full model is ~15.5 GB (fits the 28 GB box / 24 GB GPU); the old sd-dict path
    peaked ~31 GB and OOM'd. device='cuda' loads straight onto the GPU (low CPU peak)."""
    infos, ds, blob = parse_gguf(gguf)

    def W(name):
        dims, typ, _ = infos[name]
        flat = deq(blob, ds, infos[name])
        t = torch.from_numpy(flat.reshape(dims[1], dims[0]).copy()) if len(dims) == 2 \
            else torch.from_numpy(flat.copy())
        return t.to(device=device, dtype=dtype)

    dit = KleinDiT(5, 20).to(device=device, dtype=dtype).eval()
    slots = dict(dit.named_parameters()); slots.update(dict(dit.named_buffers()))
    done = set()

    def put(key, name):
        assert key in slots, f"no such param/buffer: {key}"
        with torch.no_grad():
            slots[key].data = W(name)               # replace; previous tensor freed next iter
        done.add(key)

    for k, n in {
        "img_in.weight": "img_in.weight",
        "txt_in.weight": "txt_in.weight",
        "time_in_in.weight": "time_in.in_layer.weight",
        "time_in_out.weight": "time_in.out_layer.weight",
        "dmod_img.weight": "double_stream_modulation_img.lin.weight",
        "dmod_txt.weight": "double_stream_modulation_txt.lin.weight",
        "smod.weight": "single_stream_modulation.lin.weight",
        "finalmod.weight": "final_layer.adaLN_modulation.1.weight",
        "final_lin.weight": "final_layer.linear.weight",
    }.items():
        put(k, n)
    for i in range(5):
        p, q = f"double_blocks.{i}.", f"doubles.{i}."
        put(q + "img_qkv.weight", p + "img_attn.qkv.weight")
        put(q + "img_proj.weight", p + "img_attn.proj.weight")
        put(q + "img_qn", p + "img_attn.norm.query_norm.scale")
        put(q + "img_kn", p + "img_attn.norm.key_norm.scale")
        put(q + "img_mlp_in.weight", p + "img_mlp.0.weight")
        put(q + "img_mlp_out.weight", p + "img_mlp.2.weight")
        put(q + "txt_qkv.weight", p + "txt_attn.qkv.weight")
        put(q + "txt_proj.weight", p + "txt_attn.proj.weight")
        put(q + "txt_qn", p + "txt_attn.norm.query_norm.scale")
        put(q + "txt_kn", p + "txt_attn.norm.key_norm.scale")
        put(q + "txt_mlp_in.weight", p + "txt_mlp.0.weight")
        put(q + "txt_mlp_out.weight", p + "txt_mlp.2.weight")
    for j in range(20):
        p, q = f"single_blocks.{j}.", f"singles.{j}."
        put(q + "linear1.weight", p + "linear1.weight")
        put(q + "linear2.weight", p + "linear2.weight")
        put(q + "qn", p + "norm.query_norm.scale")
        put(q + "kn", p + "norm.key_norm.scale")
    missing = set(slots) - done
    assert not missing, f"unassigned params/buffers: {sorted(missing)[:8]}"
    return dit


if __name__ == "__main__":
    from seeded_dit import val_inputs
    dit = build_real_dit()
    n = sum(p.numel() for p in dit.parameters())
    print(f"real KleinDiT loaded: {n/1e9:.2f}B params")
    with torch.no_grad():
        o = dit(*val_inputs(0))
    print(f"forward OK: out={tuple(o.shape)} mean={float(o.mean()):.4f} std={float(o.std()):.4f}")
