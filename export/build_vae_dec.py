#!/usr/bin/env python3
"""flux2-vae DECODER -> HTP-friendly torch -> tflite. Swaps the mid-block attention for a
short-named custom module (diffusers' deep names overflow the qnn lib-gen symbol length)."""
import os, numpy as np, torch, torch.nn as nn
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
import litert_torch
from diffusers import AutoencoderKL
from diffusers.models.upsampling import Upsample2D
from diffusers.models.attention_processor import AttnProcessor
from safetensors.torch import load_file

def htp_upsample(self, x, *a, **k):
    B,C,H,W = x.shape
    x = x.reshape(B,C,H,1,W,1).expand(B,C,H,2,W,2).reshape(B,C,2*H,2*W)
    return self.conv(x)
Upsample2D.forward = htp_upsample

class MA(nn.Module):   # short-named single-head VAE self-attention (HTP-friendly, decomposed)
    def __init__(s, c, groups=32):
        super().__init__(); s.gn=nn.GroupNorm(groups,c,eps=1e-6); s.q=nn.Linear(c,c); s.k=nn.Linear(c,c); s.v=nn.Linear(c,c); s.o=nn.Linear(c,c); s.c=c
    def forward(s, x, *a, **k):
        B,C,H,W = x.shape; r=x
        h = s.gn(x).reshape(B,C,H*W).transpose(1,2)
        q,k,v = s.q(h), s.k(h), s.v(h)
        a = torch.softmax((q @ k.transpose(-1,-2)) * (C**-0.5), dim=-1)
        h = (a @ v); h = s.o(h).transpose(1,2).reshape(B,C,H,W)
        return r + h

vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=32,
    block_out_channels=[128,256,512,512], down_block_types=["DownEncoderBlock2D"]*4,
    up_block_types=["UpDecoderBlock2D"]*4, layers_per_block=2, norm_num_groups=32).eval()
vae.set_attn_processor(AttnProcessor())
sd = load_file("flux2-vae.safetensors")
miss,unexp = vae.load_state_dict(sd, strict=False)
print(f"decoder missing: {len([k for k in miss if k.startswith(('decoder','post_quant'))])}")

# swap mid attention -> MA, copy weights
att = vae.decoder.mid_block.attentions[0]
ma = MA(512).eval()
ma.gn.load_state_dict(att.group_norm.state_dict())
ma.q.load_state_dict(att.to_q.state_dict()); ma.k.load_state_dict(att.to_k.state_dict())
ma.v.load_state_dict(att.to_v.state_dict()); ma.o.load_state_dict(att.to_out[0].state_dict())
# parity check the swap
xt = torch.randn(1,512,8,8)
with torch.no_grad():
    ref = att(xt); got = ma(xt)
    c = float((ref.flatten()@got.flatten())/(ref.norm()*got.norm()+1e-9))
print(f"attention swap parity cos={c:.5f} (want ~1.0)")
vae.decoder.mid_block.attentions[0] = ma

class Dec(nn.Module):
    def __init__(s, vae): super().__init__(); s.vae=vae
    def forward(s, z): return s.vae.decoder(s.vae.post_quant_conv(z))
m = Dec(vae).eval()
x = torch.randn(1,32,64,64)
with torch.no_grad(): o=m(x); print(f"decode {tuple(x.shape)}->{tuple(o.shape)} range[{float(o.min()):.2f},{float(o.max()):.2f}]")
litert_torch.convert(m, sample_args=(x,)).export("vae_dec_fp32.tflite")
print(f"exported vae_dec_fp32.tflite {os.path.getsize('vae_dec_fp32.tflite')/1e6:.0f}MB")
