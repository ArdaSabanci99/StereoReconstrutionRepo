#pragma once
#include "depth.h"
#include <vector>

// Result of a single ICP alignment (Member 4 — C2 challenge)
struct ICPResult {
    cv::Mat R;            // 3x3 rotation
    cv::Mat t;            // 3x1 translation
    double  rms_error  = 0;
    int     iterations = 0;
};

// Iterative Closest Point: aligns source into the target frame.
// Returns T = (R, t) such that R * source + t ≈ target.
ICPResult alignICP(const PointCloud& source,
                   const PointCloud& target,
                   int    max_iter  = 50,
                   double tolerance = 1e-6);

// Fuse multiple overlapping point clouds via sequential pairwise ICP.
// clouds[1] → clouds[0], clouds[2] → merged, ...
PointCloud fusePointClouds(const std::vector<PointCloud>& clouds,
                           int max_iter = 50);
