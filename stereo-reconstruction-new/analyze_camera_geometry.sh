#!/usr/bin/env bash
# Calibration-only camera-center / baseline geometry report -- no scene images needed, so this
# runs instantly. Use it to pick which view pairs to feed into run_sparse_matching_study.sh
# before spending time on the much slower SIFT/RANSAC sweep.
#
# DTU calibration is per physical camera position and is reused across all scans (see
# DTUDataLoader::decomposeProjectionMatrix, which takes no sceneId) -- the cameras themselves
# are the same regardless of which object/scene was scanned, so output is NOT scene-scoped.
#
# Columns: L, R        = view IDs of the pair
#          sep         = R - L (view-index separation)
#          baseline_mm = total distance between the two camera centers
#          horiz_mm    = that displacement along the LEFT camera's image-horizontal axis
#          vert_mm     = that displacement along the LEFT camera's image-vertical axis
#          depth_mm    = that displacement along the LEFT camera's viewing direction
#
# Usage: ./analyze_camera_geometry.sh [num_views] [--adjacent-only]
set -u
DATA="../../Data/DTU_MVS/SampleSet/MVS Data"
NUM_VIEWS="${1:-49}"
MODE="${2:-}"

OUT="results/camera_geometry"
mkdir -p "$OUT"
TABLE="$OUT/geometry.md"
CENTERS="$OUT/centers.md"

./build/camera_geometry_report "$DATA" "$NUM_VIEWS" $MODE > "$TABLE"
N=$(($(wc -l < "$TABLE") - 2))
echo "Saved $N pairs to $TABLE"
cat "$TABLE"

./build/camera_geometry_report "$DATA" "$NUM_VIEWS" --centers > "$CENTERS"
echo
echo "Saved $(($(wc -l < "$CENTERS") - 2)) camera centers to $CENTERS"
echo "Plot with: python3 scripts/plot_camera_geometry.py $CENTERS"
