#pragma once
#include "Eigen.h"
#include <vector>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// DTU point-cloud evaluation (Stage 5)
//
// Compares a reconstructed cloud against the DTU structured-light reference
// cloud (Points/stl/stlNNN_total.ply). Both must be in the same DTU world frame
// (the pipeline already transforms the reconstruction to world space), so no
// registration is performed here.
//
//   accuracy      mean over OUR points of distance to nearest GT point
//                 (how correct the points we produced are)
//   completeness  mean over GT points of distance to nearest OUR point
//                 (how much of the reference surface we covered)
//   chamfer       (accuracy + completeness) / 2          [capped variants]
//   precision@τ   % of OUR points within τ mm of GT
//   recall@τ      % of GT  points within τ mm of OUR cloud
//
// Distances are in millimetres (DTU world units). Mean values are reported both
// raw and capped at maxdist to stay robust to gross outliers, alongside medians.
// ─────────────────────────────────────────────────────────────────────────────
struct DtuEvalResult {
    double tau         = 2.0;    // recall/precision threshold (mm)
    double maxdist     = 20.0;   // outlier cap for the mean (mm)

    double accuracy_mean      = 0;  // ours → GT, raw mean
    double accuracy_median    = 0;
    double accuracy_capped    = 0;  // ours → GT, mean of min(d, maxdist)
    double precision_tau      = 0;  // % of ours within tau of GT

    double completeness_mean  = 0;  // GT → ours, raw mean
    double completeness_median= 0;
    double completeness_capped= 0;  // GT → ours, mean of min(d, maxdist)
    double recall_tau         = 0;  // % of GT within tau of ours

    double chamfer            = 0;  // (accuracy_capped + completeness_capped)/2

    size_t n_ours = 0, n_gt = 0;

    // Bounding boxes for a frame-overlap sanity check
    Eigen::Vector3f ours_min, ours_max, gt_min, gt_max;
    bool  frames_overlap = false;
};

// Robust PLY loader: reads only x,y,z. Handles ascii and binary_little_endian
// (binary_big_endian byte-swapped), float or double coordinates, and any extra
// per-vertex properties (normals, colour, …) which are skipped.
std::vector<Eigen::Vector3f> loadPlyXYZ(const std::string& path);

DtuEvalResult evaluateCloudVsReference(const std::vector<Eigen::Vector3f>& ours,
                                       const std::vector<Eigen::Vector3f>& gt,
                                       double tau = 2.0, double maxdist = 20.0);

void printDtuEval(const DtuEvalResult& r, const std::string& label = "");
