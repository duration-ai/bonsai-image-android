#!/usr/bin/env bash
# Re-pack the 20 single-stream blocks into 3 contexts (7+7+6), each well under the 2 GiB
# per-context-binary cap that killed the 10-per-context split (10 singles = 2.41 GiB,
# contextCreateFromBinary err 0x3ea even alone; R at 1.53 GiB loads fine).
#   sglAq1 = sgl0-6  (7, ~1.68 GiB)   sglBq1 = sgl7-13 (7)   sglCq1 = sgl14-19 (6, ~1.44 GiB)
# R (pro+5dbl+epi, 1.53 GiB) is unchanged — reuse the existing binsMerged/Rq1.bin.
# The runner keeps the first N singles contexts resident (resident arg = 2+N) to cut
# contextCreateFromBinary calls under the ~10 untrusted_app FastRPC leak limit. RUN IN TMUX.
set -o pipefail
source ~/qnn-env-241.sh >/dev/null 2>&1
source ~/qnn-venv310/bin/activate 2>/dev/null
cd ~/bonsai-export || exit 1
R0="$R"
OUT=~/bonsai-export/binsMerged; mkdir -p "$OUT"
LIBROOT=~/bonsai-export/mergedlibs; rm -rf "$LIBROOT"; mkdir -p "$LIBROOT"

export_block(){ CUDA_VISIBLE_DEVICES="" ./venv/bin/python export_one_q1.py "$1" >/tmp/$1_e.log 2>&1 || { echo "[$1] EXPORT FAIL" >&2; tail -4 /tmp/$1_e.log >&2; return 1; }; }
lib_fp(){ local N="$1" T="q1chunks/$1_q1_fp32.tflite" D="$LIBROOT/${1}b"
  qnn-tflite-converter -i "$T" -o "${D}.cpp" --float_bitwidth 16 --float_bias_bitwidth 16 >/tmp/${N}_c.log 2>&1 || { echo "[$N] CONVERT FAIL" >&2; return 1; }
  qnn-model-lib-generator -c "${D}.cpp" -b "${D}.bin" -t x86_64-linux-clang -l "${N}b" -o "${D}_libs" >/tmp/${N}_l.log 2>&1 || { echo "[$N] LIB FAIL" >&2; return 1; }
  echo "${D}_libs/x86_64-linux-clang/lib${N}b.so"; }
merge(){ qnn-context-binary-generator --model "$2" --backend "$R0/lib/x86_64-linux-clang/libQnnHtp.so" --config_file ctx_config.json --output_dir "$OUT" --binary_file "$1" >/tmp/merge_$1.log 2>&1
  [ -f "$OUT/$1.bin" ] && echo "[$1] OK $(stat -c %s "$OUT/$1.bin") bytes" || { echo "[$1] MERGE FAIL"; grep -iE "error|fail" /tmp/merge_$1.log | head; return 1; }; }
build_group(){ local out="$1"; shift; local sos="" s so
  for s in "$@"; do echo "  [export $s $(date +%H:%M:%S)]"; export_block "$s" || return 1
    so=$(lib_fp "$s") || return 1; sos="${sos:+$sos,}$so"; rm -f "q1chunks/${s}_q1_fp32.tflite" 2>/dev/null; done
  echo "  [merge $out ($(awk -F, '{print NF}' <<<"$sos") graphs) $(date +%H:%M:%S)]"; merge "$out" "$sos"; }

echo "##### REPACK SINGLES 7+7+6 START $(date +%H:%M:%S) #####"
build_group sglAq1 sgl0 sgl1 sgl2 sgl3 sgl4 sgl5 sgl6        || { echo ABORT A; exit 1; }
rm -rf "$LIBROOT"/*
build_group sglBq1 sgl7 sgl8 sgl9 sgl10 sgl11 sgl12 sgl13    || { echo ABORT B; exit 1; }
rm -rf "$LIBROOT"/*
build_group sglCq1 sgl14 sgl15 sgl16 sgl17 sgl18 sgl19       || { echo ABORT C; exit 1; }
rm -rf "$LIBROOT"
echo "##### DONE $(date +%H:%M:%S) #####"
for b in Rq1 sglAq1 sglBq1 sglCq1; do [ -f "$OUT/$b.bin" ] && printf "%s: %s bytes (%.2f GiB)\n" "$b" "$(stat -c %s $OUT/$b.bin)" "$(awk "BEGIN{print $(stat -c %s $OUT/$b.bin)/1073741824}")"; done