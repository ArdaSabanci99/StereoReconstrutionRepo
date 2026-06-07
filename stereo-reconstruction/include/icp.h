#pragma once
#include "PointCloud.h"   // Eigen-based PointCloud
#include <vector>

// Alignment result — same Matrix4f convention as ICPOptimizer / Exercise 5
struct ICPResult {
    Eigen::Matrix4f pose;          // 4x4 rigid transform (homogeneous coords)
    double          rms_error  = 0;
    int             iterations = 0;
};

// Align source into the target frame.
// Uses CeresICPOptimizer or LinearICPOptimizer (toggle in icp.cpp).
ICPResult alignICP(const PointCloud& source,
                   const PointCloud& target,
                   int    max_iter  = 50,
                   double tolerance = 1e-6);

// Fuse multiple overlapping point clouds via sequential pairwise ICP.
// clouds[1]→clouds[0], clouds[2]→merged, ...
PointCloud fusePointClouds(const std::vector<PointCloud>& clouds,
                           int max_iter = 50);
