#!/usr/bin/env bash
# Cross-compile the persistent-context QNN runners for the S25+ (aarch64).
# Needs the QNN headers (copy your QAIRT install's ~/qairt/.../include/QNN to /tmp/qnn_include).
set -e
NDK=$HOME/Library/Android/sdk/ndk/27.1.12297006
TOOL=$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin
INC=${1:-/tmp/qnn_include}
for src in qnn_runner qnn_chain; do
  $TOOL/aarch64-linux-android31-clang++ -O2 -std=c++17 -I"$INC" "$(dirname "$0")/$src.cpp" -o "$(dirname "$0")/$src" -ldl -llog
  echo "built $src"
done
# Run on device: push to /data/local/tmp/edge/dit alongside the *_v79.bin + input raws,
# LD_LIBRARY_PATH=<aot>:<bridge> ADSP_LIBRARY_PATH="<aot>;/system/lib/rfsa/adsp;..."
#   ./qnn_chain /data/local/tmp/edge/dit <steps>     (forces TURBO_PLUS, ~1.07 s/step)
