# Metrics

Device: Galaxy S25+ (SM-S936B), Snapdragon 8 Elite, Hexagon V79 NPU, 12 GB RAM.
Settings: 512×512, 4 denoise steps (matching the iOS port). Render = `scripts/bonsai_render.sh`.

## Timing – iOS sample-app prompt battery, on the NPU (4 steps)

| image | prompt | encode (CPU) | DiT (NPU) | decode (CPU) | total |
| --- | --- | --- | --- | --- | --- |
| bonsai.png | A bonsai tree in a quiet ceramic studio, soft morning light, shallow depth of field | 19 s | 62 s | 45 s | 126 s |
| whale.png | A massive humpback whale breaching beside a tiny fishing boat, dramatic ocean spray | 20 s | 64 s | 42 s | 126 s |
| jellyfish.png | A bioluminescent jellyfish ballet in dark ocean depths, ethereal and otherworldly | 20 s | 66 s | 45 s | 131 s |
| cabin.png | A cozy mountain cabin in winter storm, smoke from chimney, warm windows, romantic landscape | 20 s | 68 s | 47 s | 135 s |
| sailor.png | A weathered sailor in oilskin coat, salt spray on his beard, golden hour photography | 23 s | 67 s | 48 s | 138 s |

The DiT runs at ~16 s per step. The per-block NPU execute itself is ~0.15 s; the rest of each step
is loading the context binaries from storage as the chain streams through them. The encode and decode
are on the CPU. These were run one at a time from a cool start – the SoC was cooled to ~45 °C between
renders, on battery – so the encode and decode times stay tight, without the throttling drift of a
back-to-back batch.

## The three compute paths (DiT, 512×512, 4 steps)

| path | result |
| --- | --- |
| CPU (Snapdragon 8 Elite, 8 threads) | works – ~114 s per step (measured), so ~8–9 min for a full image |
| GPU (Adreno 830 v2) | partial – OpenCL rendered 256²; no GPU path finished 512² (the Vulkan path faults at every size) |
| NPU (Hexagon V79) | works – ~16 s per step, ~140 s for a full image |

The NPU is the only one of the three that both completes a 512² render and does it quickly. The CPU is
the reliable fallback but an order of magnitude slower on the transformer.

The GPU is not a flat failure: an OpenCL build of the diffusion path rendered a 256² image (the apple)
on the Adreno, but pushing it to 512² crashed during denoising. The Vulkan path is worse – it never runs
the diffusion at all. That Vulkan failure was diagnosed, not assumed. `sd-cli-vk --backend diffusion=vulkan0` runs the Qwen3
text-encode on the GPU fine, then aborts with `vk::DeviceLostError: vk::Queue::submit` on the first
diffusion step. Qualcomm's KGSL counters (`/sys/class/kgsl/kgsl-3d0/`) pin the cause: each attempt
increments `gpufaults` by one, attributed to `sd-cli-vk` in `gpufault_procs`, with `pagefaults` staying
at zero – a genuine GPU hang/reset, not a memory fault. It reproduces from a cold start (not thermal)
and at every resolution tested (512², 448², 384², 256², 128²), so it is not a capacity or
dispatch-timeout limit. It is a ggml-vulkan / Adreno-830 incompatibility on the diffusion graph
specifically – the encoder, on the same GPU, is fine. Leading suspect, not chased to the shader: the
transformer's weight types are q1_0 (1-bit) + bf16, where the working encoder is ordinary K-quants.

## Memory (peak, sampled on-device)

- Peak process RSS: ~4980 MB.
- System memory in use at peak (MemAvailable dip): ~4.6 GB.
- Model weights held by the CPU build: 4169 MB (encoder 3207, transformer 866, VAE 97).
- DiT NPU residency: ~1.6 GB (pro + 5 doubles + epilogue resident; singles streamed).

## Bundle (~10.7 GB)

- Qwen3-4B encoder gguf (4-bit UD-Q4_K_XL): 2.55 GB
- Bonsai DiT q1_0 gguf (CPU stages): 0.87 GB
- flux2 VAE safetensors: 0.32 GB
- 27 DiT V79 context binaries: 6.89 GB (pro 335 MB, 5×double 240 MB, 20×single 250 MB, epi 19 MB)
- sd-cli + QNN runtime + runner: 0.15 GB

## The in-app context-merge experiment (2026-06-29)

The in-app chain stalls because an `untrusted_app` leaks a FastRPC queue resource on every
`contextCreateFromBinary`, capping a run at about ten loads. The fix we tried was to fold the 27
graphs into fewer contexts (fewer loads). It is blocked by a separate, opaque DSP limit on how
many single-stream graphs one context may hold. Measured on the V79 with a standalone create probe
(`runner/ctx_probe.cpp`, shell user, each context created alone):

| context | size | creates? |
| --- | --- | --- |
| 1 single | 246 MB | yes |
| 2 singles | 493 MB | yes |
| 3 singles | 739 MB | yes |
| 4 singles | 986 MB | no (`err 0x3ea`) |
| 6 singles | 1479 MB | no |
| R = prologue + 5 doubles + epilogue (mixed int8/fp16) | 1563 MB | yes |

The cap is not size – R is larger than the failing 4- and 6-single contexts and loads fine. It is
the count of fp16 single-blocks per context; the doubles are int8-weight and pack denser. Three per
context means seven contexts for the twenty singles, which cannot be made both few enough (to stay
under the ~10-load FastRPC leak) and resident enough (without crossing the ~6.6 GB at which the
phone OOM-reboots). So the merge does not open the in-app path; the adb render stays the deliverable.

## Image-quality findings (2026-06-29)

Three pipeline choices, settled by eye against the iPhone port:

- Encoder: 4-bit Qwen3 (`UD-Q4_K_XL`), not 3-bit. On a clean studio subject (the bonsai) the
  two are a wash, but on richer prompts – a portrait, a dynamic scene – the 4-bit encoder gives
  visibly better prompt fidelity. Worth the extra ~0.5 GB of bundle.
- The bottom-edge watermark on portrait prompts is seed-dependent. "Photorealistic portrait"
  prompts can hallucinate a garbled caption/watermark band along the bottom (a stock-photo training
  prior; the model is guidance-distilled, so there is no negative-prompt lever at `cfg 1`). It is
  tied to the initial noise, not baked in – re-rolling the latent removes it cleanly. No crop needed.
- Percentile-99.99 activation calibration on the doubles makes it worse, not sharper. The int16
  "softness" is not fixed by clipping the activation outliers – they carry signal. The real lever for
  that is rotation-based requantisation (LRQ-DiT / Hadamard), which we have not built.
