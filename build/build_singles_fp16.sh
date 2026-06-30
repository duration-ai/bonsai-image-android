#!/usr/bin/env bash
# Rebuild the 20 singles as fp16 with the LN-overflow FIX (single_block.py layer_norm_safe,
# 1/4096 prescale). Confirmed on sgl0: finite + cos 0.9999 on the step-3 overflow input that
# NaN'd before. fp16 = shippable size (263 MB). Doubles+epi stay int16, pro stays fp16. TMUX.
cd ~/bonsai-export || exit 1
for s in $(seq -f "sgl%g" 0 19); do
  echo "===== $s $(date +%H:%M:%S) ====="
  CUDA_VISIBLE_DEVICES="" ./venv/bin/python export_one_q1.py "$s" >/dev/null 2>&1 || { echo "[$s] EXPORT FAIL"; continue; }
  bash build_fp.sh "$s" "q1chunks/${s}_q1_fp32.tflite" 2>&1 | tail -1
  rm -f "q1chunks/${s}_q1_fp32.tflite" "q1chunks/cal_${s}"*
done
echo "===== SINGLES FP16 LN-FIXED DONE $(date +%H:%M:%S) ====="
n=0; for s in $(seq -f "sgl%g" 0 19); do test -s ${s}_fpctx/${s}_v79fp.bin && n=$((n+1)); done; echo "singles: $n/20"
