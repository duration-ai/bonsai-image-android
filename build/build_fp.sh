#!/usr/bin/env bash
# Build ONE fp32 tflite -> V79 pure-fp16 ctx binary (weights+acts fp16, no calib).
source ~/qnn-env.sh || true
source ~/qnn-venv310/bin/activate || true
set -eo pipefail
cd ~/bonsai-export
N="$1"; TFL="$2"
qnn-tflite-converter -i "$TFL" -o "${N}fp.cpp" --float_bitwidth 16 >/dev/null 2>&1
qnn-model-lib-generator -c "${N}fp.cpp" -b "${N}fp.bin" -t x86_64-linux-clang -l "${N}fp" -o "${N}fp_libs" >/dev/null 2>&1
qnn-context-binary-generator --model "${N}fp_libs/x86_64-linux-clang/lib${N}fp.so" \
    --backend "$R/lib/x86_64-linux-clang/libQnnHtp.so" --config_file ctx_config.json \
    --output_dir "${N}_fpctx" --binary_file "${N}_v79fp" >/dev/null 2>&1
if [ -f "${N}_fpctx/${N}_v79fp.bin" ]; then echo "[$N] DONE $(stat -c %s ${N}_fpctx/${N}_v79fp.bin)"; else echo "[$N] FAILED"; fi
rm -f "${N}fp.cpp" "${N}fp.bin"; rm -rf "${N}fp_libs"
