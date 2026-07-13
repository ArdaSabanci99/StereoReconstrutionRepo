#!/usr/bin/env bash
# Stage-3 dense-matching study (Contribution C1).
# Sweeps every cost function / method on a fixed DTU pair, records runtime and
# 3D accuracy (DTU point-cloud eval), and collects the disparity figure for each.
# Usage: ./run_matching_study.sh <scene_id> <leftV> <rightV> <stlNNN> [scale] [ndisp] [window]
set -u
DATA="../data/SampleSet/MVS Data"
SCENE="${1:-1}"; L="${2:-1}"; R="${3:-2}"; STL="${4:-001}"
SCALE="${5:-0.25}"; NDISP="${6:-208}"; WIN="${7:-5}"   # ndisp multiple of 16 for bm/sgbm
GT="$DATA/Points/stl/stl${STL}_total.ply"
OUT="results/scene${SCENE}/study"
mkdir -p "$OUT"
TABLE="$OUT/study_pair_${L}_${R}.md"

echo "| Method | Impl | Match ms | Points | Acc.med | Acc.@20 | Comp.@20 | Chamfer@20 | Prec@2mm | Rec@2mm |" >  "$TABLE"
echo "|--------|------|---------:|-------:|--------:|--------:|---------:|-----------:|---------:|--------:|" >> "$TABLE"

for M in sad ssd ncc census sgm bm sgbm; do
  case "$M" in bm|sgbm) IMPL="OpenCV";; *) IMPL="custom";; esac
  LOG="$OUT/log_${M}_${L}_${R}.txt"
  echo ">>> $M ($IMPL)"
  ./build/pipeline "$DATA" "$SCENE" "$L" "$R" --scale "$SCALE" --ndisp "$NDISP" \
      --window "$WIN" --method "$M" --light 0 --eval-ply "$GT" > "$LOG" 2>&1
  # keep the disparity figure per method
  cp "results/scene${SCENE}/matching/view_$(printf %03d $L)_$(printf %03d $R)_disparity.png" \
     "$OUT/disp_${M}_${L}_${R}.png" 2>/dev/null
  # parse
  MS=$(grep -m1 "\[matching\] time" "$LOG" | grep -oE "[0-9.]+" | head -1)
  PTS=$(grep -m1 "disparityToCloud" "$LOG" | grep -oE "[0-9]+ points" | grep -oE "[0-9]+")
  AMED=$(awk '/accuracy \(ours/{f=1} f&&/median/{print $2; exit}' "$LOG")
  A20=$(awk '/accuracy \(ours/{f=1} f&&/mean@/{print $2; exit}' "$LOG")
  C20=$(awk '/completeness \(GT/{f=1} f&&/mean@/{print $2; exit}' "$LOG")
  CHAM=$(grep -m1 "chamfer@" "$LOG" | grep -oE "[0-9.]+" | tail -1)
  PREC=$(grep -m1 "precision@" "$LOG" | grep -oE "[0-9.]+ %" | grep -oE "[0-9.]+")
  REC=$(grep -m1 "recall@" "$LOG" | grep -oE "[0-9.]+ %" | grep -oE "[0-9.]+")
  echo "| $M | $IMPL | ${MS:-?} | ${PTS:-?} | ${AMED:-?} | ${A20:-?} | ${C20:-?} | ${CHAM:-?} | ${PREC:-?} | ${REC:-?} |" >> "$TABLE"
done
echo "=== TABLE ==="; cat "$TABLE"
