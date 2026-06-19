#include "icp.h"
#include "ICPOptimizer.h"
#include "depth.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// ICP alignment using the exercise ICPOptimizer
// ─────────────────────────────────────────────────────────────────────────────
ICPResult alignICP(const PointCloud& source, const PointCloud& target,
                   int max_iter, double /*tolerance*/) {
    ICPResult result;
    result.pose = Matrix4f::Identity();

#ifdef USE_LINEAR_ICP
    LinearICPOptimizer optimizer;
#else
    CeresICPOptimizer optimizer;
#endif
    optimizer.setMatchingMaxDistance(0.05f);
    optimizer.usePointToPlaneConstraints(!target.normals.empty());
    optimizer.setNbOfIterations(max_iter);

    Matrix4f pose = Matrix4f::Identity();
    optimizer.estimatePose(source, target, pose);

    result.pose       = pose;
    result.iterations = max_iter;

    // Compute final RMS error
    double sum_sq = 0; int cnt = 0;
    for (const auto& p : source.points) {
        Eigen::Vector4f ph(p.x(), p.y(), p.z(), 1.0f);
        Eigen::Vector3f pt = (pose * ph).head<3>();
        // Find nearest in target (brute-force, only for diagnostics)
        float min_d = std::numeric_limits<float>::max();
        for (const auto& q : target.points) min_d = std::min(min_d, (pt-q).squaredNorm());
        sum_sq += min_d; ++cnt;
    }
    result.rms_error = (cnt > 0) ? std::sqrt(sum_sq / cnt) : 0.0;
    std::cout << "[ICP] RMS = " << result.rms_error << "\n";
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Voxel downsampling
// ─────────────────────────────────────────────────────────────────────────────
PointCloud voxelDownsample(const PointCloud& cloud, float cell) {
    std::unordered_map<uint64_t, std::pair<Eigen::Vector3f, std::vector<Vector4uc>>> cells;
    bool has_color = !cloud.colors.empty();

    for (size_t i = 0; i < cloud.points.size(); ++i) {
        const auto& p = cloud.points[i];
        int ix = (int)std::floor(p.x() / cell);
        int iy = (int)std::floor(p.y() / cell);
        int iz = (int)std::floor(p.z() / cell);
        uint64_t key = ((uint64_t)(ix & 0x1FFFFF))
                     | ((uint64_t)(iy & 0x1FFFFF) << 21)
                     | ((uint64_t)(iz & 0x1FFFFF) << 42);
        auto& entry = cells[key];
        entry.first += p;
        if (has_color && i < cloud.colors.size())
            entry.second.push_back(cloud.colors[i]);
        else
            entry.second.push_back({200,200,200,255});
    }

    PointCloud out;
    for (auto& [k, v] : cells) {
        int n = (int)v.second.size();
        out.points.push_back(v.first / (float)n);
        Vector4uc c = {0,0,0,255};
        for (auto& col : v.second) {
            c[0] += col[0]/n; c[1] += col[1]/n; c[2] += col[2]/n;
        }
        out.colors.push_back(c);
    }
    std::cout << "[ICP] Voxel downsampled: " << cloud.points.size()
              << " → " << out.points.size() << "\n";
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequential multi-cloud fusion
// ─────────────────────────────────────────────────────────────────────────────
PointCloud fusePointClouds(const std::vector<PointCloud>& clouds,
                            int max_iter, float voxel_size) {
    if (clouds.empty()) return {};
    if (clouds.size() == 1) return clouds[0];

    PointCloud merged = voxelDownsample(clouds[0], voxel_size);

    for (size_t i = 1; i < clouds.size(); ++i) {
        std::cout << "[ICP] Fusing cloud " << i+1 << "/" << clouds.size() << "\n";
        ICPResult align = alignICP(clouds[i], merged, max_iter);

        // Apply transform and append
        for (size_t j = 0; j < clouds[i].points.size(); ++j) {
            const auto& p = clouds[i].points[j];
            Eigen::Vector4f ph(p.x(), p.y(), p.z(), 1.0f);
            Eigen::Vector3f pt = (align.pose * ph).head<3>();
            merged.points.push_back(pt);
            if (!clouds[i].colors.empty() && j < clouds[i].colors.size())
                merged.colors.push_back(clouds[i].colors[j]);
            else
                merged.colors.push_back({200,200,200,255});
        }

        // Voxel downsample to control memory
        merged = voxelDownsample(merged, voxel_size);
    }
    return merged;
}

// ─────────────────────────────────────────────────────────────────────────────
// PLY loader (ASCII, coloured)
// ─────────────────────────────────────────────────────────────────────────────
PointCloud loadPointCloudPLY(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "[ICP] Cannot open " << path << "\n"; return {}; }

    std::string line;
    int n_verts = 0;
    bool has_color = false, header_done = false;
    while (std::getline(f, line)) {
        if (line.find("element vertex") != std::string::npos)
            std::istringstream(line.substr(15)) >> n_verts;
        if (line.find("red") != std::string::npos) has_color = true;
        if (line == "end_header") { header_done = true; break; }
    }
    if (!header_done || n_verts <= 0) {
        std::cerr << "[ICP] Bad PLY header in " << path << "\n"; return {};
    }

    PointCloud cloud;
    cloud.points.reserve(n_verts);
    for (int i = 0; i < n_verts; ++i) {
        float x, y, z;
        f >> x >> y >> z;
        cloud.points.emplace_back(x, y, z);
        if (has_color) {
            int r, g, b; f >> r >> g >> b;
            cloud.colors.push_back({(uchar)r,(uchar)g,(uchar)b,255});
        }
    }
    std::cout << "[ICP] Loaded " << cloud.points.size() << " pts from " << path << "\n";
    return cloud;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone
// ─────────────────────────────────────────────────────────────────────────────
#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: icp_fuse <cloud1.ply> <cloud2.ply> [cloud3.ply ...]\n";
        return 1;
    }
    std::vector<PointCloud> clouds;
    for (int i = 1; i < argc; ++i)
        clouds.push_back(loadPointCloudPLY(argv[i]));

    PointCloud fused = fusePointClouds(clouds);

    std::string save = "results/icp_fused";
    fs::create_directories(save);
    savePointCloud(fused, save + "/fused.ply");
    savePointCloudOFF(fused, save + "/fused.off");
    std::cout << "Fused cloud: " << fused.points.size() << " points\n";
    return 0;
}
#endif
