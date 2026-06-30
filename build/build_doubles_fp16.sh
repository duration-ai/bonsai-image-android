#!/usr/bin/env bash
# Rebuild the 5 DOUBLE blocks as fp16 with the LN-overflow FIX (dit_block.py layer_norm now
# uses _LNP=1/4096 instead of 1/256). Sharpening pass: int16 doubles -> fp16 doubles to close
# the 0.933->~0.99 gap. fp16 = ~263 MB/chunk (shippable). pro+singles already fp16, epi int16.
cd ~/bonsai-export || exit 1
for s in dbl0 dbl1 dbl2 dbl3 dbl4; do
  echo "===== $s $(date +%H:%M:%S) ====="
  CUDA_VISIBLE_DEVICES="" ./venv/bin/python export_one_q1.py "$s" 2>&1 | tail -1 || { echo "[$s] EXPORT FAIL"; continue; }
  bash build_fp.sh "$s" "q1chunks/${s}_q1_fp32.tflite" 2>&1 | tail -1
  rm -f "q1chunks/${s}_q1_fp32.tflite" "q1chunks/cal_${s}"*
done
echo "===== DOUBLES FP16 LN-FIXED DONE $(date +%H:%M:%S) ====="
n=0; for s in dbl0 dbl1 dbl2 dbl3 dbl4; do test -s ${s}_fpctx/${s}_v79fp.bin && n=$((n+1)); done; echo "doubles: $n/5"
