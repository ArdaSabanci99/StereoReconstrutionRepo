#pragma once
#include "PointCloud.h"
#include "Eigen.h"

struct ICPResult {
    Matrix4f pose;
    int      iterations;
    double   rms_error;
};

// Align source to target using ICP.
// Returns the transform T such that T * source ≈ target.
ICPResult alignICP(const PointCloud& source, const PointCloud& target,
                   int max_iter = 20, double tolerance = 1e-4);

// Sequentially fuse multiple clouds into one using ICP.
PointCloud fusePointClouds(const std::vector<PointCloud>& clouds,
                            int max_iter = 20,
                            float voxel_size = 5.0f);   // mm

// Load .ply file into PointCloud
PointCloud loadPointCloudPLY(const std::string& path);

// Voxel downsampling (keep one point per voxel cell)
PointCloud voxelDownsample(const PointCloud& cloud, float cell_size);
