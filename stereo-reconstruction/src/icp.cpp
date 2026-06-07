#include "icp.h"
#include "ICPOptimizer.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// Toggle between Ceres (non-linear) and linear closed-form ICP
#define USE_LINEAR_ICP     0   // 0 = Ceres, 1 = linear (Procrustes)
#define USE_POINT_TO_PLANE 0   // 0 = point-to-point, 1 = point-to-plane

// ── ICP alignment (Member 4 — C2 challenge) ───────────────────────────────

ICPResult alignICP(const PointCloud& source, const PointCloud& target,
                   int max_iter, double /*tolerance*/) {
    ICPResult result;
    result.pose = Matrix4f::Identity();

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

    result.pose       = pose;
    result.iterations = max_iter;
    // TODO: compute final RMS error after applying pose to source and
    //       finding nearest-neighbor distances to target
    result.rms_error  = 0;
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

        // TODO-FUSE-1: Apply align.pose to each point in clouds[i] and append
        //   for (const auto& p : clouds[i].points) {
        //       Eigen::Vector4f ph(p.x(), p.y(), p.z(), 1.0f);
        //       Eigen::Vector4f pt = align.pose * ph;
        //       merged.points.emplace_back(pt.x(), pt.y(), pt.z());
        //   }
        //   merged.colors.insert(merged.colors.end(),
        //                        clouds[i].colors.begin(),
        //                        clouds[i].colors.end());

        // TODO-FUSE-2: Voxel downsampling — keep one point per cell of side
        //   voxel_size to avoid density artefacts after merging.
    }

    return merged;
}

#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: icp <cloud1.ply> <cloud2.ply> [cloud3.ply ...]\n"
                  << "  Outputs results/fused.ply and results/fused.off\n";
        return 1;
    }

    // TODO: Load PLY files → fusePointClouds → save
    // std::vector<PointCloud> clouds;
    // for (int i = 1; i < argc; ++i)
    //     clouds.push_back(loadPointCloud(argv[i]));  // TODO: implement loader
    // PointCloud fused = fusePointClouds(clouds);
    // savePointCloud   (fused, "results/fused.ply");
    // savePointCloudOFF(fused, "results/fused.off");

    std::cout << "[ICP] Standalone not yet fully implemented.\n";
    return 0;
}
#endif
