#!/system/bin/sh
# Fully on-device Bonsai text->image: text-encode (Qwen3, 8 threads) -> NPU DiT (V79, resident small blocks) -> VAE decode. NO Mac.
DEV=/data/local/tmp/edge; SD=$DEV/sd; M=$DEV/models; AOT=$DEV/aot; IO=$DEV/q1run/nio; BINS=$DEV/q1run/bins
PROMPT="${1:-a red apple on a wooden table}"; OUT="${2:-$SD/out.png}"; STEPS="${3:-4}"
MM="--diffusion-model $M/bonsai_image_4b-q1_0.gguf --vae $M/flux2-vae.safetensors --llm $M/Qwen3-4B-UD-Q4_K_XL.gguf --threads 8 --cfg-scale 1 --width 512 --height 512"
T0=$(date +%s)
echo "[1/3] text-encode: \"$PROMPT\""
LD_LIBRARY_PATH=$SD SD_DUMP_CTX=$IO/context.raw SD_ENCODE_ONLY=1 $SD/sd-cli $MM --steps 1 --prompt "$PROMPT" --output $SD/none.png 2>/dev/null
T1=$(date +%s); echo "  encode $((T1-T0))s"
echo "[2/3] NPU DiT (V79, $STEPS steps, resident)"
cd $DEV/q1run && LD_LIBRARY_PATH=$AOT ADSP_LIBRARY_PATH=$AOT $DEV/q1run/qnn_chain512 $BINS $IO $STEPS 1 2>/dev/null | tail -1
T2=$(date +%s); echo "  dit $((T2-T1))s"
echo "[3/3] VAE decode -> $OUT"
LD_LIBRARY_PATH=$SD SD_LOAD_LATENT=$IO/final_chain.raw $SD/sd-cli $MM --steps 1 --prompt "$PROMPT" --output $OUT 2>/dev/null
T3=$(date +%s); echo "  decode $((T3-T2))s"
echo "=== TOTAL $((T3-T0))s -> $OUT ==="
