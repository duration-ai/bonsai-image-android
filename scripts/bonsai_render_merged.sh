#!/system/bin/sh
# Gate A: fully on-device render using the MERGED 3-context DiT (resident=2).
# TE (Qwen3, CPU) -> NPU DiT merged (V79, 3 contexts) -> VAE decode (CPU). Shell user (no FastRPC wall).
DEV=/data/local/tmp/edge; SD=$DEV/sd; M=$DEV/models; AOT=$DEV/aot; IO=$DEV/q1run/nio; BINS=/sdcard/bonsai/bins
PROMPT="${1:-a red apple on a wooden table}"; OUT="${2:-$SD/out_merged.png}"; STEPS="${3:-3}"; RES="${4:-3}"
MM="--diffusion-model $M/bonsai_image_4b-q1_0.gguf --vae $M/flux2-vae.safetensors --llm $M/Qwen3-4B-UD-Q3_K_XL.gguf --threads 8 --cfg-scale 1 --width 512 --height 512"
T0=$(date +%s)
echo "[1/3] text-encode: \"$PROMPT\""
LD_LIBRARY_PATH=$SD SD_DUMP_CTX=$IO/context.raw SD_ENCODE_ONLY=1 $SD/sd-cli $MM --steps 1 --prompt "$PROMPT" --output $SD/none.png 2>/dev/null
T1=$(date +%s); echo "  encode $((T1-T0))s"
echo "[2/3] NPU DiT MERGED (V79, $STEPS steps, resident=$RES, R+3 singles ctx)"
cd $DEV/q1run && LD_LIBRARY_PATH=$AOT ADSP_LIBRARY_PATH=$AOT $DEV/q1run/qnn_chain512_merged $BINS $IO $STEPS $RES 2>&1 | grep -E "merged|loadctx.ctxcreate|step |chain|DONE|not found|fail|ctx " | tail -12
T2=$(date +%s); echo "  dit $((T2-T1))s"
echo "[3/3] VAE decode -> $OUT"
LD_LIBRARY_PATH=$SD SD_LOAD_LATENT=$IO/final_chain.raw $SD/sd-cli $MM --steps 1 --prompt "$PROMPT" --output $OUT 2>/dev/null
T3=$(date +%s); echo "  decode $((T3-T2))s"
echo "=== TOTAL $((T3-T0))s -> $OUT === temp: $(cat /sys/class/thermal/thermal_zone0/temp)"
