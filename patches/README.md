# sd-cli – the NPU-split patch

The CPU stages of the pipeline – the Qwen3 text-encode and the flux2 VAE decode –
run on `sd-cli`, a build of stable-diffusion.cpp. The 1-bit (`q1_0`) Bonsai support
comes from [Juste-Leo2's fork](https://github.com/Juste-Leo2/stable-diffusion.cpp);
on top of that, `npu-split.patch` adds three small hooks (about 40 lines, all in
`src/stable-diffusion.cpp`) so the encode and decode can run as separate stages on
either side of the NPU diffusion transformer:

- `SD_DUMP_CTX=<path>` – after the text-encode, write the conditioning tensor to
  `<path>`, where the QNN runner picks it up.
- `SD_ENCODE_ONLY=1` – exit immediately after dumping the context (an encode-only run).
- `SD_LOAD_LATENT=<path>` – skip the CPU diffusion entirely, load the NPU's output
  latent from `<path>`, and decode it.

The hooks are inert unless their environment variable is set, so a patched `sd-cli`
still behaves exactly like upstream for an ordinary end-to-end run.

## Build

```sh
git clone https://github.com/Juste-Leo2/stable-diffusion.cpp
cd stable-diffusion.cpp
git checkout ddcad62          # the bonsai_dev commit this patch was cut against
git apply /path/to/npu-split.patch
# then build sd-cli per upstream's instructions, cross-compiled for arm64 Android (NDK)
```

`scripts/bonsai_render.sh` drives the three stages in order: `SD_ENCODE_ONLY` to dump
the context, the QNN runner (`qnn_chain512`) for the diffusion transformer, then
`SD_LOAD_LATENT` to decode the NPU's latent.
