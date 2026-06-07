#include "rectification.h"
#include "matching.h"
#include "depth.h"
#include <iostream>
#include <string>

static void printUsage(const char* name) {
    std::cerr << "Usage: " << name << " <scene_dir> [options]\n"
              << "  --method  bm|sgbm|sad|ssd|ncc|sgm   (default: sgbm)\n"
              << "  --manual-rect                    (use manual rectification)\n"
              << "  --window  <size>                 (default: 5)\n"
              << "  --disps   <num_disparities>      (default: from calib.txt)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { printUsage(argv[0]); return 1; }

    fs::path scene(argv[1]);
    MatchParams mp;
    mp.method      = MatchMethod::OPENCV_SGBM;
    mp.window_size = 5;
    bool manual_rect   = false;
    bool disps_override = false;

    for (int i = 2; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--manual-rect")      manual_rect = true;
        else if (a == "--method" && i+1 < argc) {
            std::string m(argv[++i]);
            if      (m == "bm")   mp.method = MatchMethod::OPENCV_BM;
            else if (m == "sgbm") mp.method = MatchMethod::OPENCV_SGBM;
            else if (m == "sad")  mp.method = MatchMethod::MANUAL_SAD;
            else if (m == "ssd")  mp.method = MatchMethod::MANUAL_SSD;
            else if (m == "ncc")  mp.method = MatchMethod::MANUAL_NCC;
            else if (m == "sgm")  mp.method = MatchMethod::MANUAL_SGM;
        }
        else if (a == "--window" && i+1 < argc) mp.window_size = std::stoi(argv[++i]);
        else if (a == "--disps"  && i+1 < argc) {
            mp.num_disparities = std::stoi(argv[++i]);
            disps_override = true;
        }
    }

    // ── 1. Load data ───────────────────────────────────────────
    std::cout << "=== Loading scene: " << scene << " ===\n";
    CalibData calib = loadCalib(scene);

    // Use ndisp from calib.txt unless overridden
    if (!disps_override)
        mp.num_disparities = calib.ndisp;

    cv::Mat left  = cv::imread((scene / "im0.png").string());
    cv::Mat right = cv::imread((scene / "im1.png").string());
    if (left.empty() || right.empty()) {
        std::cerr << "Could not load images.\n"; return 1;
    }
    printMatInfo("left",  left);
    printMatInfo("right", right);

    // ── 2. Rectification ───────────────────────────────────────
    std::cout << "\n=== Rectification ===\n";
    auto rect = manual_rect ? rectifyManual(left, right, calib)
                            : rectifyOpenCV(left, right, calib);

    fs::create_directories("results");
    cv::imwrite("results/left_rect.png",  rect.left_rect);
    cv::imwrite("results/right_rect.png", rect.right_rect);

    // ── 3. Stereo matching ─────────────────────────────────────
    std::cout << "\n=== Stereo Matching (ndisp=" << mp.num_disparities
              << " window=" << mp.window_size << ") ===\n";
    cv::Mat disp = computeDisparity(rect.left_rect, rect.right_rect, mp);

    cv::Mat disp_vis;
    cv::normalize(disp, disp_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);
    cv::imwrite("results/disparity.png", disp_vis);
    saveDisparity(disp, "results/disparity_raw.png");

    // ── 4. Depth + point cloud ─────────────────────────────────
    std::cout << "\n=== Depth & Point Cloud ===\n";
    cv::Mat depth = disparityToDepth(disp, calib);
    auto cloud    = depthToPointCloud(depth, calib, rect.left_rect);
    savePointCloud(cloud, "results/pointcloud.ply");

    std::cout << "\n=== Pipeline complete. Results in results/ ===\n";
    return 0;
}
