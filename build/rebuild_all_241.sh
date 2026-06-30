#!/usr/bin/env bash
# Rebuild ALL 27 Bonsai DiT context bins with QAIRT 2.41 (fixes the in-app 2.46 dspqueue stall).
# Recipe UNCHANGED from the banked 0.933 set; only the backend libQnnHtp -> 2.41.
#   pro + sgl0..19 : fp16  (--float_bitwidth 16, LN-safe via current single_block.py)
#   dbl0..4 + epi  : int16 (w8a16 + multi-prompt calibsamples calib)
# Re-exports each chunk fresh (export_one_q1.py -> current LN-fixed source) then builds.
# Output: bins241/<spec>q1.bin (the exact names qnn_chain512 loads). RUN IN TMUX.
source ~/qnn-env-241.sh >/dev/null 2>&1
source ~/qnn-venv310/bin/activate 2>/dev/null
cd ~/bonsai-export || exit 1
R0="$R"; C=$HOME/bonsai-export/calibsamples
OUT=~/bonsai-export/bins241; mkdir -p "$OUT"
SINGLES="pro $(seq -f 'sgl%g' 0 19)"
INT16="dbl0 dbl1 dbl2 dbl3 dbl4 epi"

build_fp() {   # $1=spec  (fp16, no calib)
  local N="$1" T="q1chunks/$1_q1_fp32.tflite"
  qnn-tflite-converter -i "$T" -o "${N}b.cpp" --float_bitwidth 16 --float_bias_bitwidth 16 >/tmp/${N}_c.log 2>&1 || { echo "[$N] CONVERT FAIL"; return 1; }
  qnn-model-lib-generator -c "${N}b.cpp" -b "${N}b.bin" -t x86_64-linux-clang -l "${N}b" -o "${N}b_libs" >/tmp/${N}_l.log 2>&1 || { echo "[$N] LIB FAIL"; return 1; }
  qnn-context-binary-generator --model "${N}b_libs/x86_64-linux-clang/lib${N}b.so" --backend "$R0/lib/x86_64-linux-clang/libQnnHtp.so" --config_file ctx_config.json --output_dir "${N}b_ctx" --binary_file "${N}q1" >/tmp/${N}_x.log 2>&1
  rm -f "${N}b.cpp" "${N}b.bin"; rm -rf "${N}b_libs"
  [ -f "${N}b_ctx/${N}q1.bin" ] && { mv -f "${N}b_ctx/${N}q1.bin" "$OUT/${N}q1.bin"; rm -rf "${N}b_ctx"; return 0; } || { echo "[$N] NO BIN"; grep -iE "error|1002|flat_from" /tmp/${N}_x.log|head -3; return 1; }
}
build_i16() {  # $1=spec  (int16 w8a16 + calib)
  local N="$1" T="q1chunks/$1_q1_fp32.tflite" CAL="$C/cal_$1.txt"
  test -s "$CAL" || { echo "[$N] NO CALIB $CAL"; return 1; }
  qnn-tflite-converter -i "$T" -o "${N}b.cpp" --input_list "$CAL" --weights_bitwidth 8 --use_per_row_quantization --keep_weights_quantized --act_bitwidth 16 >/tmp/${N}_c.log 2>&1 || { echo "[$N] CONVERT FAIL"; tail -2 /tmp/${N}_c.log; return 1; }
  qnn-model-lib-generator -c "${N}b.cpp" -b "${N}b.bin" -t x86_64-linux-clang -l "${N}b" -o "${N}b_libs" >/tmp/${N}_l.log 2>&1 || { echo "[$N] LIB FAIL"; return 1; }
  qnn-context-binary-generator --model "${N}b_libs/x86_64-linux-clang/lib${N}b.so" --backend "$R0/lib/x86_64-linux-clang/libQnnHtp.so" --config_file ctx_config.json --output_dir "${N}b_ctx" --binary_file "${N}q1" >/tmp/${N}_x.log 2>&1
  rm -f "${N}b.cpp" "${N}b.bin"; rm -rf "${N}b_libs"
  [ -f "${N}b_ctx/${N}q1.bin" ] && { mv -f "${N}b_ctx/${N}q1.bin" "$OUT/${N}q1.bin"; rm -rf "${N}b_ctx"; return 0; } || { echo "[$N] NO BIN"; grep -iE "error|1002|flat_from" /tmp/${N}_x.log|head -3; return 1; }
}

echo "##### REBUILD 27 @2.41 START $(date +%H:%M:%S) #####"
for s in $SINGLES; do
  echo "===== [fp16] $s $(date +%H:%M:%S) ====="
  CUDA_VISIBLE_DEVICES="" ./venv/bin/python export_one_q1.py "$s" >/tmp/${s}_e.log 2>&1 || { echo "[$s] EXPORT FAIL"; tail -2 /tmp/${s}_e.log; continue; }
  build_fp "$s" && echo "[$s] OK $(stat -c %s $OUT/${s}q1.bin)"
  rm -f "q1chunks/${s}_q1_fp32.tflite" "q1chunks/cal_${s}"* 2>/dev/null
done
for s in $INT16; do
  echo "===== [int16] $s $(date +%H:%M:%S) ====="
  CUDA_VISIBLE_DEVICES="" ./venv/bin/python export_one_q1.py "$s" >/tmp/${s}_e.log 2>&1 || { echo "[$s] EXPORT FAIL"; tail -2 /tmp/${s}_e.log; continue; }
  build_i16 "$s" && echo "[$s] OK $(stat -c %s $OUT/${s}q1.bin)"
  rm -f "q1chunks/${s}_q1_fp32.tflite" 2>/dev/null
done
echo "##### DONE $(date +%H:%M:%S) — bins: $(ls -1 $OUT/*q1.bin 2>/dev/null | wc -l)/27 #####"
ls -1 $OUT/*q1.bin 2>/dev/null | sed 's#.*/##' | sort | tr '\n' ' '; echo
