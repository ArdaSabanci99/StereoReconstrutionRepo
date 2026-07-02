#include "DataLoader.h"
#include "sparse_matching.h"
#include "rectification.h"
#include "matching.h"
#include "depth.h"
#include "evaluation.h"
#include "dtu_eval.h"
#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

static void printUsage(const char* name) {
    std::cerr
        << "Usage: " << name
        << " <data_path> <scene_id> <left_id> <right_id> [options]\n"
        << "Dense matching:\n"
        << "  --method    sad|ssd|ncc|census|sgm|bm|sgbm   (default: sgm)\n"
        << "  --window    <size>                             (default: 5)\n"
        << "  --ndisp     <num_disparities>                  (default: 64)\n"
        << "  --no-subpixel\n"
        << "  --no-lr\n"
        << "  --no-median\n"
        << "Pipeline:\n"
        << "  --manual-rect                                  (Loop-Zhang rectification)\n"
        << "  --scale     <factor>                           (default: 0.5)\n"
        << "  --gt        <path_to.pfm>                      (evaluate if given)\n"
        << "Sparse matching:\n"
        << "  --opencv_sm                                    (use OpenCV sparse matching; default: manual)\n"
        << "  --sm-features  <N>   SIFT max keypoints, 0=unlimited (default: 0)\n"
        << "  --sm-octaves   <N>   SIFT octave layers (default: 3)\n"
        << "  --sm-contrast  <F>   SIFT contrast threshold (default: 0.04)\n"
        << "  --sm-edge      <F>   SIFT edge threshold (default: 10.0)\n"
        << "  --sm-sigma     <F>   SIFT Gaussian sigma (default: 1.6)\n"
        << "  --sm-ratio     <F>   Lowe ratio test threshold (default: 0.75)\n"
        << "  --sm-ransac    <F>   RANSAC Sampson distance threshold px (default: 1.0)\n"
        << "  --sm-chirality <F>   recoverPose max depth (default: 25.0)\n"
        << "Evaluation:\n"
        << "  --gt        <path_to.pfm>                      (disparity eval, Middlebury)\n"  //TODO: Check whether it is being used in current setup 
        << "  --eval-ply  <reference.ply>                    (DTU point-cloud eval)\n" 
}

int main(int argc, char** argv) {
    if (argc < 5) { printUsage(argv[0]); return 1; }

    fs::path dataPath(argv[1]);
    const std::string sceneId(argv[2]);
    const std::string viewL = padViewId(std::stoi(argv[3]));
    const std::string viewR = padViewId(std::stoi(argv[4]));

    MatchParams mp;
    mp.method = MatchMethod::MANUAL_SGM;
    bool manual_rect = false;
    double scale = 0.5;

    std::string gt_path;
    bool opencv_sm = false;
    SparseMatchParams smParams;

    float max_depth_mm = 2000.0f;
    std::string gt_path;
    std::string eval_ply_path;

    for (int i = 5; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--manual-rect")                 manual_rect = true;
        else if (a == "--no-subpixel")                 mp.subpixel = false;
        else if (a == "--no-lr")                       mp.lr_check = false;
        else if (a == "--no-median")                   mp.median_filter = false;
        else if (a == "--method" && i+1 < argc) {
            std::string m(argv[++i]);
            if      (m == "sad")   mp.method = MatchMethod::MANUAL_SAD;
            else if (m == "ssd")   mp.method = MatchMethod::MANUAL_SSD;
            else if (m == "ncc")   mp.method = MatchMethod::MANUAL_NCC;
            else if (m == "census")mp.method = MatchMethod::MANUAL_CENSUS;
            else if (m == "sgm")   mp.method = MatchMethod::MANUAL_SGM;
            else if (m == "bm")    mp.method = MatchMethod::OPENCV_BM;
            else if (m == "sgbm")  mp.method = MatchMethod::OPENCV_SGBM;
        }

        else if (a == "--window"       && i+1 < argc) mp.window_size               = std::stoi(argv[++i]);
        else if (a == "--ndisp"        && i+1 < argc) mp.num_disparities           = std::stoi(argv[++i]);
        else if (a == "--scale"        && i+1 < argc) scale                        = std::stod(argv[++i]);
        else if (a == "--gt"           && i+1 < argc) gt_path                      = argv[++i];
        else if (a == "--opencv_sm")                   opencv_sm                   = true;
        else if (a == "--sm-features"  && i+1 < argc) smParams.sift_features        = std::stoi(argv[++i]);
        else if (a == "--sm-octaves"   && i+1 < argc) smParams.sift_octave_layers   = std::stoi(argv[++i]);
        else if (a == "--sm-contrast"  && i+1 < argc) smParams.sift_contrast_thresh = std::stod(argv[++i]);
        else if (a == "--sm-edge"      && i+1 < argc) smParams.sift_edge_thresh     = std::stod(argv[++i]);
        else if (a == "--sm-sigma"     && i+1 < argc) smParams.sift_sigma           = std::stod(argv[++i]);
        else if (a == "--sm-ratio"     && i+1 < argc) smParams.ratio_threshold      = std::stof(argv[++i]);
        else if (a == "--sm-ransac"    && i+1 < argc) smParams.ransac_threshold     = std::stod(argv[++i]);
        else if (a == "--sm-chirality" && i+1 < argc) smParams.chirality_depth      = std::stod(argv[++i]);

        else if (a == "--gt"      && i+1 < argc) gt_path            = argv[++i];
        else if (a == "--eval-ply"&& i+1 < argc) eval_ply_path      = argv[++i];
        else if (a == "--zmax"    && i+1 < argc) max_depth_mm       = std::stof(argv[++i]);
    }

    // ── 1. Load images & calibration ─────────────────────────────────────
    std::cout << "\n=== Loading scene " << sceneId << " views "
              << viewL << "–" << viewR << " ===\n";
    DTUDataLoader loader(dataPath.string());
    CalibData calib = loader.loadCalibIntrinsics(viewL, viewR);
    cv::Mat imgL = loader.loadImage(sceneId, viewL);
    cv::Mat imgR = loader.loadImage(sceneId, viewR);
    if (imgL.empty() || imgR.empty()) {
        std::cerr << "Could not load images.\n"; return 1;
    }
    printMatInfo("Left Image",  imgL);
    printMatInfo("Right Image", imgR);

    // Downscale
    if (scale != 1.0) {
        cv::resize(imgL, imgL, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(imgR, imgR, cv::Size(), scale, scale, cv::INTER_AREA);
        for (cv::Mat* K : {&calib.K0, &calib.K1}) {
            K->at<double>(0,0) *= scale; K->at<double>(1,1) *= scale;
            K->at<double>(0,2) = (K->at<double>(0,2)+0.5)*scale - 0.5;
            K->at<double>(1,2) = (K->at<double>(1,2)+0.5)*scale - 0.5;
        }
        std::cout << "Images downscaled to " << imgL.cols << "×" << imgL.rows << "\n";
    }

    // ── 2. Sparse matching ────────────────────────────────────────────────
    std::cout << "\n=== Sparse Matching ===\n";
    SparseMatchResult sparse = computeSparseMatches(imgL, imgR, calib, opencv_sm, smParams);

    if (calib.swapped) {
        std::swap(imgL, imgR);
    }
    std::string sparse_path = "results/scene" + sceneId + "/sparse_matching";
    fs::create_directories(sparse_path);
    saveCalibData(calib, sparse_path + "/calib_" + viewL + "_" + viewR + ".yaml");

    // ── 3. Rectification ─────────────────────────────────────────────────
    std::cout << "\n=== Rectification (" << (manual_rect ? "Loop-Zhang" : "OpenCV") << ") ===\n";
    RectifyResult rect;
    rect = manual_rect ? rectifyManual(imgL, imgR, calib) : rectifyOpenCV(imgL, imgR, calib);

    std::string rect_path = "results/scene" + sceneId + "/rectification";
    fs::create_directories(rect_path);
    cv::imwrite(rect_path + "/view_" + viewL + ".png", rect.left_rect);
    cv::imwrite(rect_path + "/view_" + viewR + ".png", rect.right_rect);

    // Side-by-side with epipolar lines
    cv::Mat sbs; cv::hconcat(rect.left_rect, rect.right_rect, sbs);
    for (int y = 0; y < sbs.rows; y += 40)
        cv::line(sbs, {0,y}, {sbs.cols,y}, {0,255,0}, 1);
    cv::imwrite(rect_path + "/epipolar_lines.png", sbs);

    // ── 4. Dense matching ─────────────────────────────────────────────────
    std::cout << "\n=== Dense Matching (ndisp=" << mp.num_disparities
              << " window=" << mp.window_size << ") ===\n";
    cv::Mat disp = computeDisparity(rect.left_rect, rect.right_rect, mp);

    cv::Mat disp_vis;
    cv::normalize(disp, disp_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);

    std::string match_path = "results/scene" + sceneId + "/matching";
    fs::create_directories(match_path);
    cv::imwrite(match_path + "/view_" + viewL + "_" + viewR + "_disparity.png", disp_vis);
    saveDisparity(disp, match_path + "/view_" + viewL + "_" + viewR + "_raw.png");

    // ── 5. Depth & point cloud ────────────────────────────────────────────
    std::cout << "\n=== Point Cloud ===\n";
    PointCloud cloud = disparityToCloud(disp, rect.Q, rect.left_rect,
                                        std::max(1.0f, (float)mp.min_disparity),
                                        max_depth_mm);

    // Load left camera world pose now that swapped flag is settled after sparse matching.
    loader.loadLeftExtrinsics(calib, viewL, viewR);

    // Transform to world space: X_world = R0^T * (R0_rect^T * X_rect - t0)
    {
        cv::Mat R0t       = calib.R0.t();
        cv::Mat R0_rect_t = rect.R0_rect.t();
        for (auto& p : cloud.points) {
            cv::Mat x = (cv::Mat_<double>(3,1) << p.x(), p.y(), p.z());
            cv::Mat w = R0t * (R0_rect_t * x - calib.t0);
            p = Eigen::Vector3f((float)w.at<double>(0),
                                (float)w.at<double>(1),
                                (float)w.at<double>(2));
        }
        std::cout << "[pipeline] Transformed " << cloud.points.size()
                  << " points to world space.\n";
    }

    std::string cloud_path = "results/scene" + sceneId + "/pointcloud";
    fs::create_directories(cloud_path);
    savePointCloud(cloud, cloud_path + "/views_" + viewL + "_" + viewR + ".ply");
    savePointCloudOFF(cloud, cloud_path + "/views_" + viewL + "_" + viewR + ".off");

    // ── 5b. DTU point-cloud evaluation (optional) ─────────────────────────
    if (!eval_ply_path.empty()) {
        std::cout << "\n=== DTU Point-Cloud Evaluation ===\n";
        std::vector<Eigen::Vector3f> gt = loadPlyXYZ(eval_ply_path);
        if (gt.empty()) {
            std::cerr << "[pipeline] Could not load reference cloud: " << eval_ply_path << "\n";
        } else {
            DtuEvalResult er = evaluateCloudVsReference(cloud.points, gt);
            printDtuEval(er, "scene" + sceneId + " views " + viewL + "-" + viewR
                             + " (" + (manual_rect ? "Loop-Zhang" : "OpenCV") + ")");
        }
    }

    // ── 6. Evaluation (optional) ──────────────────────────────────────────
    if (!gt_path.empty()) {
        std::cout << "\n=== Evaluation ===\n";
        cv::Mat gt = loadPFM(gt_path);
        if (!gt.empty()) {
            // Resize disparity to GT size if needed
            cv::Mat disp_eval = disp.clone();
            if (disp_eval.size() != gt.size())
                cv::resize(disp_eval, disp_eval, gt.size(), 0, 0, cv::INTER_NEAREST);
            EvalResult r = evaluateDisparity(disp_eval, gt);
            printEvalResult(r, "Scene " + sceneId + " views " + viewL + "-" + viewR);
        }
    }

    std::cout << "\n=== Pipeline complete. Results in results/scene"
              << sceneId << "/ ===\n";
    return 0;
}
