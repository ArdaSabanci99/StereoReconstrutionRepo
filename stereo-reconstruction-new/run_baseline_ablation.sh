#!/usr/bin/env bash
# Baseline ablation (C1): fix the method (custom SGM), scale and ndisp; vary the
# stereo baseline by anchoring the left view at 001 and widening the right view.
# Shows how dense-matching quality behaves as the baseline grows.
# Common settings so ONLY the baseline changes. SGM caps D at 256, and disparity
# grows ~linearly with baseline, so we run at scale 0.20 (320px wide) / ndisp 252.
set -u
DATA="../data/SampleSet/MVS Data"; GT="$DATA/Points/stl/stl001_total.ply"
SCALE=0.20; NDISP=252
OUT="results/scene1/study"; mkdir -p "$OUT"
TAB="$OUT/ablation_baseline_sgm.md"
echo "| Pair | Baseline (mm) | Match ms | Points | Chamfer@20 | Prec@2mm | Rec@2mm |" >  "$TAB"
echo "|------|--------------:|---------:|-------:|-----------:|---------:|--------:|" >> "$TAB"
for R in 2 3 4; do
  L="$OUT/log_baseline_1_${R}.txt"
  ./build/pipeline "$DATA" 1 1 "$R" --scale "$SCALE" --ndisp "$NDISP" --window 5 \
      --method sgm --eval-ply "$GT"  --test-gt-pose > "$L" 2>&1
  cp "results/scene1/matching/view_001_$(printf %03d $R)_disparity.png" \
     "$OUT/disp_baseline_1_${R}.png" 2>/dev/null
  BASE=$(grep -m1 "Baseline:" "$L" | grep -oE "[0-9.]+" | head -1)
  MS=$(grep -m1 "\[matching\] time" "$L" | grep -oE "[0-9.]+" | head -1)
  PTS=$(grep -m1 "disparityToCloud" "$L" | grep -oE "[0-9]+ points" | grep -oE "[0-9]+")
  CH=$(grep -m1 "chamfer@" "$L" | grep -oE "[0-9.]+" | tail -1)
  PR=$(grep -m1 "precision@" "$L" | grep -oE "[0-9.]+ %" | grep -oE "[0-9.]+")
  RE=$(grep -m1 "recall@" "$L" | grep -oE "[0-9.]+ %" | grep -oE "[0-9.]+")
  printf "| 1-%s | %s | %s | %s | %s | %s | %s |\n" "$R" "${BASE:-?}" "${MS:-?}" "${PTS:-?}" "${CH:-?}" "${PR:-?}" "${RE:-?}" >> "$TAB"
  echo ">>> pair 1-$R  baseline=${BASE} chamfer=${CH}"
done
echo "=== TABLE ==="; cat "$TAB"
