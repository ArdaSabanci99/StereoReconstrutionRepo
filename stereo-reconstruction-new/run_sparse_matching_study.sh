#!/usr/bin/env bash
# Sparse-matching stage study: custom vs OpenCV 8-point/RANSAC, and OpenCV vs custom SIFT,
# against exact DTU ground-truth pose. Sweeps scene, view-pair separation, backend combo, and
# RANSAC Sampson threshold. Same pattern as run_matching_study.sh: the C++ binary
# (eval_sparse_matching) prints numbers to stdout, this script greps them into a markdown table.
#
# WARNING: custom SIFT detection timing has been observed to vary wildly (seconds to tens of
# minutes per image, apparently load-dependent -- see notes/sparse_matching_analysis.md).
# Default COMBOS below only exercises the two OpenCV-SIFT backends; opt into "custom" explicitly
# once you've timed a single run on an otherwise-idle machine.
#
# Usage: ./run_sparse_matching_study.sh <scenes> <pairs> [combos] [ransac_thresholds]
#   <scenes>            comma-separated scene ids, e.g. "1,6"
#   <pairs>             comma-separated "L-R" pairs, e.g. "1-2,1-6,1-11,1-21"
#                        OR "adjacent:N"  -> N adjacent pairs starting at 1 (1-2,2-3,...)
#                        OR "skip:K:N"    -> N pairs with separation K (1-(1+K),(1+K)-(1+2K),...)
#   [combos]            comma-separated sift:fmatrix combos, default "opencv:opencv,opencv:custom"
#                        (fmatrix "opencv"|"custom" controls --opencv; sift "opencv"|"custom"
#                        controls --custom-sift; note "custom:opencv" is not reachable -- the
#                        pipeline only exposes 3 combos, see eval_sparse_matching.cpp usage)
#   [ransac_thresholds] comma-separated RANSAC Sampson thresholds (px), default "1.0"
set -u
DATA="../../Data/DTU_MVS/SampleSet/MVS Data"
SCENES="${1:-1,6}"
PAIRS_SPEC="${2:-adjacent:5}"
COMBOS="${3:-opencv:opencv,opencv:custom}"
THRESHOLDS="${4:-1.0}"

expand_pairs() {
  local spec="$1"
  if [[ "$spec" == adjacent:* ]]; then
    local n="${spec#adjacent:}"
    seq 1 "$n" | awk '{printf "%d-%d,", $1, $1+1}' | sed 's/,$//'
  elif [[ "$spec" == skip:* ]]; then
    local rest="${spec#skip:}"; local k="${rest%%:*}"; local n="${rest##*:}"
    awk -v k="$k" -v n="$n" 'BEGIN{s=1; for(i=0;i<n;i++){printf "%d-%d,", s, s+k; s=s+k}}' | sed 's/,$//'
  else
    echo "$spec"
  fi
}
PAIRS=$(expand_pairs "$PAIRS_SPEC")

for SCENE in ${SCENES//,/ }; do
  OUT="results/scene${SCENE}/sparse_matching_study"
  mkdir -p "$OUT"
  TABLE="$OUT/study.md"

  echo "| L | R | SIFT | FMatrix | RANSACpx | n_matches | n_inliers | inlier% | n_pose_inl | sampson_est_px | rot_err_deg | t_err_deg | F_frob_diff | sampson_gt_px | sift_ms | fmatrix_ms |" >  "$TABLE"
  echo "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"                                                                                              >> "$TABLE"

  for PAIR in ${PAIRS//,/ }; do
    L="${PAIR%-*}"; R="${PAIR#*-}"
    for COMBO in ${COMBOS//,/ }; do
      SIFT="${COMBO%%:*}"; FM="${COMBO##*:}"
      if [[ "$SIFT" == "custom" && "$FM" == "opencv" ]]; then
        echo "!!! skipping $COMBO: custom SIFT + OpenCV fmatrix is not reachable via --opencv/--custom-sift (see eval_sparse_matching.cpp usage) -- would silently run as opencv:opencv"
        continue
      fi
      FLAGS=()
      [[ "$FM"   == "opencv" ]] && FLAGS+=(--opencv)
      [[ "$SIFT" == "custom" ]] && FLAGS+=(--custom-sift)
      for THR in ${THRESHOLDS//,/ }; do
        LOG="$OUT/log_${L}_${R}_${SIFT}_${FM}_${THR}.txt"
        echo ">>> scene $SCENE  pair $L-$R  sift=$SIFT fmatrix=$FM  ransac=$THR"
        ./build/eval_sparse_matching "$DATA" "$SCENE" "$L" "$R" "${FLAGS[@]}" --ransac "$THR" > "$LOG" 2>&1

        NM=$(grep -m1 "Matches after ratio test" "$LOG" | grep -oE "[0-9]+" | head -1)
        NI=$(grep -m1 "RANSAC inliers" "$LOG" | grep -oE "[0-9]+" | head -1)
        NPI=$(grep -m1 "Pose cheirality inliers" "$LOG" | grep -oE "[0-9]+" | head -1)
        SAMP_EST=$(grep -m1 "Mean Sampson error (inliers)" "$LOG" | grep -oE "[0-9.]+" | head -1)
        ROT=$(grep -m1 "Rotation error" "$LOG" | grep -oE "[0-9.]+" | head -1)
        TERR=$(grep -m1 "Translation direction error" "$LOG" | grep -oE "[0-9.]+" | head -1)
        FDIFF=$(grep -m1 "Frobenius diff" "$LOG" | grep -oE "[0-9.]+" | head -1)
        SAMP_GT=$(grep -m1 "vs F_gt" "$LOG" | grep -oE "[0-9.]+" | head -1)
        SIFT_MS=$(grep -m1 "SIFT stage" "$LOG" | grep -oE "[0-9.]+" | head -1)
        FM_MS=$(grep -m1 "F-matrix stage" "$LOG" | grep -oE "[0-9.]+" | head -1)
        INLPCT=$(awk -v n="${NM:-0}" -v i="${NI:-0}" 'BEGIN{ if(n>0) printf "%.1f", 100*i/n; else print "0.0"}')

        echo "| $L | $R | $SIFT | $FM | $THR | ${NM:-?} | ${NI:-?} | ${INLPCT:-?} | ${NPI:-?} | ${SAMP_EST:-?} | ${ROT:-?} | ${TERR:-?} | ${FDIFF:-?} | ${SAMP_GT:-?} | ${SIFT_MS:-?} | ${FM_MS:-?} |" >> "$TABLE"
      done
    done
  done
  echo "=== scene $SCENE table ==="; cat "$TABLE"
done
