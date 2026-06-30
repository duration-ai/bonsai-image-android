#!/usr/bin/env bash
# Rebuild all 27 with int8 weights + int16 acts + MULTI-PROMPT calib (calibsamples/cal_<chunk>.txt
# from q1_calib_rollout.py) -> prompt-ROBUST int16 ranges (max over 12 prompts x 2 steps).
# Same clean recipe as build_clean_all; only the calib set is broader. Overwrites bins. TMUX.
cd ~/bonsai-export || exit 1
source ~/qnn-env.sh 2>/dev/null; source ~/qnn-venv310/bin/activate 2>/dev/null
R0="$R"; C=$HOME/bonsai-export/calibsamples
build() {
  qnn-tflite-converter -i "$2" -o "$1.cpp" --input_list "$3" --weights_bitwidth 8 --use_per_row_quantization --keep_weights_quantized --act_bitwidth 16 >/dev/null 2>&1
  qnn-model-lib-generator -c "$1.cpp" -b "$1.bin" -t x86_64-linux-clang -l "$1" -o "$1_libs" >/dev/null 2>&1
  qnn-context-binary-generator --model "$1_libs/x86_64-linux-clang/lib$1.so" --backend "$R0/lib/x86_64-linux-clang/libQnnHtp.so" --config_file ctx_config.json --output_dir "$1_ctx" --binary_file "$1_v79" >/dev/null 2>&1
  rm -f "$1.cpp" "$1.bin"; rm -rf "$1_libs"
}
SPECS="pro dbl0 dbl1 dbl2 dbl3 dbl4 sgl0 sgl1 sgl2 sgl3 sgl4 sgl5 sgl6 sgl7 sgl8 sgl9 sgl10 sgl11 sgl12 sgl13 sgl14 sgl15 sgl16 sgl17 sgl18 sgl19 epi"
for s in $SPECS; do
  echo "===== $s $(date +%H:%M:%S) ====="
  CUDA_VISIBLE_DEVICES="" ./venv/bin/python export_one_q1.py "$s" >/dev/null 2>&1 || { echo "[$s] EXPORT FAIL"; continue; }
  test -s "$C/cal_${s}.txt" || { echo "[$s] NO CALIB ($C/cal_${s}.txt)"; continue; }
  build "${s}q1" "q1chunks/${s}_q1_fp32.tflite" "$C/cal_${s}.txt"
  sz=$(stat -c %s ${s}q1_ctx/${s}q1_v79.bin 2>/dev/null); echo "[$s] ${sz:-FAILED}"
  rm -f "q1chunks/${s}_q1_fp32.tflite" "q1chunks/cal_${s}.txt" "q1chunks/cal_${s}_a"*.raw
done
echo "===== ALL DONE $(date +%H:%M:%S) ====="
echo "bins: $(ls -1 *q1_ctx/*q1_v79.bin 2>/dev/null | wc -l)/27"
