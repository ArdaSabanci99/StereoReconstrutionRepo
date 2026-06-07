#include "icp.h"
#include "ICPOptimizer.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// ── ICP alignment (Member 4 — C2 challenge) ───────────────────────────────
//
// Uses the same CeresICPOptimizer / LinearICPOptimizer infrastructure as
// Exercise 5 (ICPOptimizer.h). Toggle USE_LINEAR_ICP and USE_POINT_TO_PLANE
// to switch between variants.

#define USE_LINEAR_ICP     0   // 0 = Ceres, 1 = linear closed-form
#define USE_POINT_TO_PLANE 0   // 0 = point-to-point, 1 = point-to-plane

ICPResult alignICP(const PointCloud& source, const PointCloud& target,
                   int max_iter, double /*tolerance*/) {
    ICPResult result;

    ICPOptimizer* optimizer = nullptr;
    if (USE_LINEAR_ICP)
        optimizer = new LinearICPOptimizer();
    else
        optimizer = new CeresICPOptimizer();

    optimizer->setMatchingMaxDistance(0.05f);
    optimizer->usePointToPlaneConstraints(USE_POINT_TO_PLANE);
    optimizer->setNbOfIterations(max_iter);

    Matrix4f pose = Matrix4f::Identity();
    optimizer->estimatePose(source, target, pose);
    delete optimizer;

    // Extract R and t from the 4x4 pose matrix
    cv::Mat R_cv(3, 3, CV_64F), t_cv(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R_cv.at<double>(i,j) = pose(i,j);
    for (int i = 0; i < 3; ++i)
        t_cv.at<double>(i,0) = pose(i,3);

    result.R          = R_cv;
    result.t          = t_cv;
    result.rms_error  = 0;   // TODO: compute final RMS after alignment
    result.iterations = max_iter;
    return result;
}

// ── Sequential multi-cloud fusion (Member 4 — C2 challenge) ──────────────

PointCloud fusePointClouds(const std::vector<PointCloud>& clouds, int max_iter) {
    if (clouds.empty()) return PointCloud{};
    if (clouds.size() == 1) return clouds[0];

    PointCloud merged = clouds[0];

    for (size_t i = 1; i < clouds.size(); ++i) {
        std::cout << "[ICP] Fusing cloud " << i+1 << "/" << clouds.size() << "\n";
        ICPResult align = alignICP(clouds[i], merged, max_iter);

        // TODO-FUSE: Transform clouds[i] by align.R, align.t and append to merged
        // for (const auto& p : clouds[i].points) {
        //     cv::Mat pt = (cv::Mat_<double>(3,1) << p.x(), p.y(), p.z());
        //     cv::Mat pt_aligned = align.R * pt + align.t;
        //     merged.points.emplace_back(
        //         (float)pt_aligned.at<double>(0),
        //         (float)pt_aligned.at<double>(1),
        //         (float)pt_aligned.at<double>(2));
        // }
        // merged.colors.insert(merged.colors.end(),
        //                      clouds[i].colors.begin(), clouds[i].colors.end());

        // TODO-FUSE-2: Optional voxel downsampling to reduce density artefacts
        //   Grid-subsample: divide space into voxels of side voxel_size,
        //   keep the centroid of each occupied voxel.
    }

    return merged;
}

#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: icp <cloud1.ply> <cloud2.ply> [cloud3.ply ...]\n"
                  << "  Outputs results/fused.ply\n";
        return 1;
    }

    // TODO: Load .ply files, fuse, save
    // std::vector<PointCloud> clouds;
    // for (int i = 1; i < argc; ++i)
    //     clouds.push_back(loadPointCloud(argv[i]));
    // PointCloud fused = fusePointClouds(clouds);
    // savePointCloud(fused, "results/fused.ply");

    std::cout << "[ICP] Standalone not yet fully implemented.\n";
    return 0;
}
#endif
