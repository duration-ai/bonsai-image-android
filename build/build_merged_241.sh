#!/usr/bin/env bash
# Build the 3 MERGED V79 context binaries for the in-app (tappable) path.
#   Rq1.bin    = pro + dbl0..4 + epi   (7 graphs, ~1.6 GB)  -> resident
#   sglAq1.bin = sgl0..sgl9            (10 fp16 graphs, ~2.5 GB) -> streamed
#   sglBq1.bin = sgl10..sgl19          (10 fp16 graphs, ~2.5 GB) -> streamed
# Why: the all-separate-bins in-app chain does ~67 contextCreateFromBinary calls over a
# 3-step run; an untrusted_app leaks a FastRPC dspqueue resource per create and stalls at
# ~10. Folding 27 graphs into 3 contexts => 1 (R, resident) + 2/step*3 = 7 creates, under
# the limit, at ~4 GB peak (R + one singles ctx). Per-block recipe is IDENTICAL to
# rebuild_all_241.sh (fp16 for pro+singles, int16 w8a16+calib for doubles+epi); only the
# packaging differs. Graph names stay <spec>b (prob, dbl0b.., sgl0b.., epib). RUN IN TMUX.
set -o pipefail
source ~/qnn-env-241.sh >/dev/null 2>&1
source ~/qnn-venv310/bin/activate 2>/dev/null
cd ~/bonsai-export || exit 1
R0="$R"; C=$HOME/bonsai-export/calibsamples
OUT=~/bonsai-export/binsMerged; mkdir -p "$OUT"
LIBROOT=~/bonsai-export/mergedlibs; rm -rf "$LIBROOT"; mkdir -p "$LIBROOT"

export_block() { # $1=spec -> q1chunks/<spec>_q1_fp32.tflite
  CUDA_VISIBLE_DEVICES="" ./venv/bin/python export_one_q1.py "$1" >/tmp/$1_e.log 2>&1 \
    || { echo "[$1] EXPORT FAIL" >&2; tail -4 /tmp/$1_e.log >&2; return 1; }
}
lib_fp() {  # $1=spec -> echoes abs .so path (graph name = <spec>b)
  local N="$1" T="q1chunks/$1_q1_fp32.tflite" D="$LIBROOT/${1}b"
  qnn-tflite-converter -i "$T" -o "${D}.cpp" --float_bitwidth 16 --float_bias_bitwidth 16 >/tmp/${N}_c.log 2>&1 || { echo "[$N] CONVERT FAIL" >&2; return 1; }
  qnn-model-lib-generator -c "${D}.cpp" -b "${D}.bin" -t x86_64-linux-clang -l "${N}b" -o "${D}_libs" >/tmp/${N}_l.log 2>&1 || { echo "[$N] LIB FAIL" >&2; return 1; }
  echo "${D}_libs/x86_64-linux-clang/lib${N}b.so"
}
lib_i16() { # $1=spec -> echoes abs .so path; int16 w8a16 + multi-prompt calib
  local N="$1" T="q1chunks/$1_q1_fp32.tflite" CAL="$C/cal_$1.txt" D="$LIBROOT/${1}b"
  test -s "$CAL" || { echo "[$N] NO CALIB $CAL" >&2; return 1; }
  qnn-tflite-converter -i "$T" -o "${D}.cpp" --input_list "$CAL" --weights_bitwidth 8 --use_per_row_quantization --keep_weights_quantized --act_bitwidth 16 >/tmp/${N}_c.log 2>&1 || { echo "[$N] CONVERT FAIL" >&2; tail -2 /tmp/${N}_c.log >&2; return 1; }
  qnn-model-lib-generator -c "${D}.cpp" -b "${D}.bin" -t x86_64-linux-clang -l "${N}b" -o "${D}_libs" >/tmp/${N}_l.log 2>&1 || { echo "[$N] LIB FAIL" >&2; return 1; }
  echo "${D}_libs/x86_64-linux-clang/lib${N}b.so"
}
merge() {  # $1=outname (no ext)  $2=comma list of .so
  qnn-context-binary-generator --model "$2" --backend "$R0/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file ctx_config.json --output_dir "$OUT" --binary_file "$1" >/tmp/merge_$1.log 2>&1
  [ -f "$OUT/$1.bin" ] && echo "[$1] OK $(stat -c %s "$OUT/$1.bin") bytes" \
    || { echo "[$1] MERGE FAIL"; grep -iE "error|fail|not found|undefined" /tmp/merge_$1.log | head -6; return 1; }
}

build_group() { # $1=outname  $2.. = spec:qt pairs (qt = fp | i16)
  local out="$1"; shift; local sos="" pair s qt so
  for pair in "$@"; do
    s="${pair%%:*}"; qt="${pair##*:}"
    echo "  [export $s ($qt) $(date +%H:%M:%S)]"; export_block "$s" || return 1
    if [ "$qt" = "i16" ]; then so=$(lib_i16 "$s") || return 1; else so=$(lib_fp "$s") || return 1; fi
    sos="${sos:+$sos,}$so"
    rm -f "q1chunks/${s}_q1_fp32.tflite" 2>/dev/null || true
  done
  echo "  [merge $out ($(awk -F, '{print NF}' <<<"$sos") graphs) $(date +%H:%M:%S)]"
  merge "$out" "$sos"
}

echo "##### BUILD MERGED @2.41 START $(date +%H:%M:%S) #####"
build_group sglAq1 $(for j in $(seq 0 9);  do echo sgl$j:fp; done) || { echo ABORT sglA; exit 1; }
rm -rf "$LIBROOT"/sgl*_libs "$LIBROOT"/sgl*.cpp "$LIBROOT"/sgl*.bin 2>/dev/null || true
build_group sglBq1 $(for j in $(seq 10 19); do echo sgl$j:fp; done) || { echo ABORT sglB; exit 1; }
rm -rf "$LIBROOT"/sgl*_libs "$LIBROOT"/sgl*.cpp "$LIBROOT"/sgl*.bin 2>/dev/null || true
build_group Rq1 pro:fp dbl0:i16 dbl1:i16 dbl2:i16 dbl3:i16 dbl4:i16 epi:i16 || { echo ABORT R; exit 1; }
rm -rf "$LIBROOT" 2>/dev/null || true

echo "##### DONE $(date +%H:%M:%S) #####"
for b in Rq1 sglAq1 sglBq1; do
  [ -f "$OUT/$b.bin" ] || { echo "MISSING $b"; continue; }
  qnn-context-binary-utility --context_binary "$OUT/$b.bin" --json_file /tmp/$b.json >/dev/null 2>&1
  echo "=== $b.bin $(stat -c %s "$OUT/$b.bin") bytes ==="; grep -iE "\"graphName\"" /tmp/$b.json | tr -d ' '
done
echo "total: $(du -sh "$OUT" | cut -f1)"
