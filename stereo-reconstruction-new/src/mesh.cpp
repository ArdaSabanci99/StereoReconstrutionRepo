#include "depth.h"
#include "DataLoader.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;


/**
 * @brief Mesh structure containing vertices, colors, and faces for 3D representation. 
 */
struct Mesh {
    std::vector<cv::Point3f> verts;
    std::vector<cv::Vec3b>   colors;
    std::vector<cv::Vec3i>   faces;
};

/**
 * @brief Triangulates a depth map into a mesh, optionally using color information.
 *        Each 2×2 pixel block of valid depth values yields 2 triangles:
 * 
 *        (y,x) --- (y,x+1)
 *          |  \        |
 *          |    \      |
 *        (y+1,x)-(y+1,x+1)
 * 
 * @param[in] depth Depth map 
 * @param[in] calib Calibration data
 * @param[in] color Colour image (optional, must match depth size)
 * @param[in] depth_jump_thresh Threshold for depth discontinuity to avoid creating triangles across large depth jumps.
 * @return Mesh 
 */
Mesh triangulateDepthMap(const cv::Mat& depth,
                          const CalibData& calib,
                          const cv::Mat& color,
                          float depth_jump_thresh = 0.05f) {
    const float fx = (float)calib.K0.at<double>(0, 0);
    const float fy = (float)calib.K0.at<double>(1, 1);
    const float cx = (float)calib.K0.at<double>(0, 2);
    const float cy = (float)calib.K0.at<double>(1, 2);

    const int H = depth.rows, W = depth.cols;
    const bool has_color = !color.empty() && color.rows == H && color.cols == W;

    // Map from pixel (y*W+x) → vertex index, -1 if invalid
    std::vector<int> idx_map(H * W, -1);
    Mesh mesh;
    mesh.verts.reserve(H * W / 4);
    mesh.colors.reserve(H * W / 4);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float Z = depth.at<float>(y, x);
            if (Z <= 0.0f) continue;
            idx_map[y * W + x] = (int)mesh.verts.size();
            mesh.verts.emplace_back((x - cx) * Z / fx,
                                    (y - cy) * Z / fy,
                                    Z);
            if (has_color)
                mesh.colors.push_back(color.at<cv::Vec3b>(y, x));
            else
                mesh.colors.push_back({200, 200, 200});
        }
    }

    mesh.faces.reserve(H * W / 2);

    for (int y = 0; y < H - 1; ++y) {
        for (int x = 0; x < W - 1; ++x) {
            int i00 = idx_map[ y    * W +  x   ];
            int i10 = idx_map[ y    * W + (x+1)];
            int i01 = idx_map[(y+1) * W +  x   ];
            int i11 = idx_map[(y+1) * W + (x+1)];

            // Triangle 1: top-left, top-right, bottom-left
            if (i00 >= 0 && i10 >= 0 && i01 >= 0) {
                float dmax = std::max({
                    std::abs(mesh.verts[i00].z - mesh.verts[i10].z),
                    std::abs(mesh.verts[i00].z - mesh.verts[i01].z),
                    std::abs(mesh.verts[i10].z - mesh.verts[i01].z)
                });
                if (dmax <= depth_jump_thresh)
                    mesh.faces.push_back({i00, i10, i01});
            }

            // Triangle 2: top-right, bottom-right, bottom-left
            if (i10 >= 0 && i11 >= 0 && i01 >= 0) {
                float dmax = std::max({
                    std::abs(mesh.verts[i10].z - mesh.verts[i11].z),
                    std::abs(mesh.verts[i10].z - mesh.verts[i01].z),
                    std::abs(mesh.verts[i11].z - mesh.verts[i01].z)
                });
                if (dmax <= depth_jump_thresh)
                    mesh.faces.push_back({i10, i11, i01});
            }
        }
    }

    std::cout << "[mesh] Triangulated: " << mesh.verts.size() << " vertices, "
              << mesh.faces.size() << " faces\n";
    return mesh;
}


/**
 * @brief Save a mesh to an OBJ file, optionally including vertex colors.
 *        Vertices: "v X Y Z"
 *        Per-vertex color as comment (MeshLab extension): "v X Y Z R G B" where R,G,B in [0,1]
 *        Faces:    "f v1 v2 v3"  (1-indexed)
 * 
 * @param mesh 
 * @param path 
 */
void saveMeshOBJ(const Mesh& mesh, const std::string& path) {
    std::ofstream f(path);
    if (!f) { std::cerr << "[mesh] Cannot open " << path << "\n"; return; }

    const bool has_color = !mesh.colors.empty();

    for (size_t i = 0; i < mesh.verts.size(); ++i) {
        const auto& v = mesh.verts[i];
        f << "v " << v.x << " " << v.y << " " << v.z;
        if (has_color) {
            const auto& c = mesh.colors[i];
            f << " " << c[2]/255.0f << " " << c[1]/255.0f << " " << c[0]/255.0f;
        }
        f << "\n";
    }

    for (const auto& fc : mesh.faces)
        f << "f " << fc[0]+1 << " " << fc[1]+1 << " " << fc[2]+1 << "\n";

    std::cout << "[mesh] Saved OBJ (" << mesh.verts.size() << " verts, "
              << mesh.faces.size() << " faces) → " << path << "\n";
}

/**
 * @brief Laplacian smoothing of a mesh. 
 *        Each vertex is moved towards the mean of its neighbors.
 *        v_i_new = (1 - lambda) * v_i + lambda * mean(neighbors(v_i)) 
 * 
 * @param mesh Mesh to smooth. Only geometry (verts) is modified; colors stay fixed.
 * @param iters Applies the smoothing operation this many times.
 * @param lambda Controls the strength of smoothing.
 */
void laplacianSmooth(Mesh& mesh, int iters, float lambda = 0.5f) {
    if (iters <= 0) return;
    const int N = (int)mesh.verts.size();

    // Build vertex adjacency list from face connectivity
    std::vector<std::set<int>> adj(N);
    for (const auto& fc : mesh.faces) {
        adj[fc[0]].insert(fc[1]); adj[fc[0]].insert(fc[2]);
        adj[fc[1]].insert(fc[0]); adj[fc[1]].insert(fc[2]);
        adj[fc[2]].insert(fc[0]); adj[fc[2]].insert(fc[1]);
    }

    for (int it = 0; it < iters; ++it) {
        std::vector<cv::Point3f> new_verts = mesh.verts;
        for (int i = 0; i < N; ++i) {
            if (adj[i].empty()) continue;
            cv::Point3f mean(0.0f, 0.0f, 0.0f);
            for (int j : adj[i]) mean += mesh.verts[j];
            mean *= (1.0f / (float)adj[i].size());
            new_verts[i] = mesh.verts[i] * (1.0f - lambda) + mean * lambda;
        }
        mesh.verts = std::move(new_verts);
    }

    std::cout << "[mesh] Laplacian smoothing done (" << iters
              << " iters, lambda=" << lambda << ")\n";
}

// ── Main ──────────────────────────────────────────────────────────────────
//
// Usage:
//   mesh <data_path> <scene_id> <left_view_id> <right_view_id>
//        [--light <id>] [--smooth <n>] [--lambda <f>] [--jump <f>]
//        [--out <dir>]
//
// Loads the pre-computed disparity from results/<scene_id>/matching/,
// converts to depth, triangulates it, optionally smooths, and saves OBJ.

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: mesh <data_path> <scene_id> <left_view_id> <right_view_id> [options]\n"
                  << "  --light  <id>     light condition (default: 0)\n"
                  << "  --smooth <n>      Laplacian smoothing iterations (default: 0)\n"
                  << "  --lambda <f>      smoothing strength in [0,1] (default: 0.5)\n"
                  << "  --jump   <f>      depth discontinuity threshold in metres (default: 0.05)\n"
                  << "  --out    <dir>    output directory (default: results/scene<id>/mesh/)\n";
        return 1;
    }

    fs::path dataPath(argv[1]);
    const std::string sceneId(argv[2]);
    const std::string viewLeftId  = padViewId(std::stoi(argv[3]));
    const std::string viewRightId = padViewId(std::stoi(argv[4]));

    std::string lightId = "0";
    int smooth_n = 0;
    float lambda = 0.5f;
    float jump_thr = 0.05f;
    std::string out_dir = "results/scene" + sceneId + "/mesh";

    for (int i = 5; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--smooth" && i+1 < argc) smooth_n  = std::stoi(argv[++i]);
        else if (a == "--lambda" && i+1 < argc) lambda = std::stof(argv[++i]);
        else if (a == "--jump" && i+1 < argc) jump_thr = std::stof(argv[++i]);
        else if (a == "--out" && i+1 < argc) out_dir = argv[++i];
    }

    // Load calibration
    DTUDataLoader loader(dataPath.string());
    std::string calib_path = "results/scene" + sceneId + "/sparse_matching/calib_"
                         + viewLeftId + "_" + viewRightId + ".yaml";
    CalibData calib = loadCalibData(calib_path);
    if (!calib.hasIntrinsics()) {
        std::cerr << "[mesh]: Missing intrinsics — run sparse_matching first.\n";
        return 1;
    }

    // Load pre-computed disparity saved by pipeline/matching stage
    std::string disp_path = "results/scene" + sceneId + "/matching/view_"
                            + viewLeftId + "_" + viewRightId + "_raw.png";
    cv::Mat disp = loadDisparity(disp_path);
    if (disp.empty()) {
        std::cerr << "[mesh] Could not load disparity: " << disp_path
                  << "\nRun the pipeline or matching stage first.\n";
        return 1;
    }

    // Load rectified image for vertex colors
    cv::Mat color = cv::imread("results/scene" + sceneId + "/rectification/view_" + viewLeftId + ".png");

    // Convert disparity → depth
    cv::Mat depth = disparityToDepth(disp, calib);

    // Triangulate
    Mesh mesh = triangulateDepthMap(depth, calib, color, jump_thr);

    // Optionally smooth
    if (smooth_n > 0)
        laplacianSmooth(mesh, smooth_n, lambda);

    // Save
    fs::create_directories(out_dir);
    std::string obj_path = out_dir + "/views_" + viewLeftId + "_" + viewRightId + ".obj";
    saveMeshOBJ(mesh, obj_path);

    std::cout << "[mesh] Done. Output in " << out_dir << "\n";
    return 0;
}
