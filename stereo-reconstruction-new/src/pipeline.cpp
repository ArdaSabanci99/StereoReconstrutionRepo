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
#include <algorithm>

namespace fs = std::filesystem;

static void printUsage(const char* name) {
    std::cerr
        << "Usage: " << name
        << " <data_path> <scene_id> <left_id> <right_id> [options]\n"
        << "Dense matching:\n"
        << "  --method    sad|ssd|ncc|census|sgm|bm|sgbm   (default: sgm)\n"
        << "  --window    <size>                             (default: 5)\n"
        << "  --ndisp     <num_disparities>                  (default: 256 at scale 1.0)\n"
        << "  --min-disp  <minimum_disparity>                (default: scaled from 80)\n"
        << "  --no-subpixel\n"
        << "  --no-lr\n"
        << "  --no-median\n"
        << "Pipeline:\n"
        << "  --manual-rect | --rect-manual                  (Loop-Zhang closed-form rectification)\n"
        << "  --scale     <factor>                           (default: 0.5)\n"
        << "  --test-gt-pose                                 (skip sparse matching, use DTU GT pose;\n"
        << "                                                   also derives GT F, so combines with\n"
        << "                                                   --manual-rect)\n"
        << "Sparse matching:\n"
        << "  --sm-opencv                                    (use OpenCV sparse matching; default: manual)\n"
        << "  --sm-custom-sift                                (manual pipeline only: use our own SIFT\n"
        << "                                                    detector instead of cv::SIFT)\n"
        << "  --sm-features  <N>   SIFT max keypoints, 0=unlimited (default: 0)\n"
        << "  --sm-ratio     <F>   Lowe ratio test threshold (default: 0.75)\n"
        << "  --sm-ransac    <F>   RANSAC Sampson distance threshold px (default: 1.0)\n"
        << "                       (other SIFT/RANSAC tuning knobs live in the sparse_matching /\n"
        << "                        test_sparse_matching executables, not here)\n"
        << "Evaluation:\n"
        << "  --gt        <path_to.pfm>                      (disparity eval, Middlebury)\n"
        << "  --eval-ply  <reference.ply>                  (DTU point-cloud eval)\n";
}

int main(int argc, char** argv) {
    if (argc < 5) { printUsage(argv[0]); return 1; }

    fs::path data_path(argv[1]);
    const std::string scene_id(argv[2]);
    const std::string viewL = padViewId(std::stoi(argv[3]));
    const std::string viewR = padViewId(std::stoi(argv[4]));

    MatchParams mp;
    mp.method = MatchMethod::MANUAL_SGM;
    bool manual_rect = false;
    double scale = 0.5;

    bool opencv_sm = false;
    bool test_gt_pose = false;
    SparseMatchParams smParams;

    float max_depth_mm = 2000.0f;
    std::string gt_path;
    std::string eval_ply_path;

    for (int i = 5; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--manual-rect" || a == "--rect-manual") manual_rect = true;
        else if (a == "--no-subpixel") mp.subpixel = false;
        else if (a == "--no-lr") mp.lr_check = false;
        else if (a == "--no-median") mp.median_filter = false;
        else if (a == "--method" && i+1 < argc) {
            std::string m(argv[++i]);
            if (m == "sad") mp.method = MatchMethod::MANUAL_SAD;
            else if (m == "ssd") mp.method = MatchMethod::MANUAL_SSD;
            else if (m == "ncc") mp.method = MatchMethod::MANUAL_NCC;
            else if (m == "census") mp.method = MatchMethod::MANUAL_CENSUS;
            else if (m == "sgm") mp.method = MatchMethod::MANUAL_SGM;
            else if (m == "bm") mp.method = MatchMethod::OPENCV_BM;
            else if (m == "sgbm") mp.method = MatchMethod::OPENCV_SGBM;
        }

        else if (a == "--window" && i+1 < argc) mp.window_size = std::stoi(argv[++i]);
        else if (a == "--ndisp" && i+1 < argc) mp.num_disparities = std::stoi(argv[++i]);
        else if (a == "--min-disp" && i+1 < argc) mp.min_disparity = std::stoi(argv[++i]);
        else if (a == "--scale" && i+1 < argc) scale = std::stod(argv[++i]);
        else if (a == "--sm-opencv") opencv_sm = true;
        else if (a == "--sm-custom-sift") smParams.use_custom_sift = true;
        else if (a == "--test-gt-pose") test_gt_pose = true;
        else if ((a == "--sm-features" || a == "--n-features") && i+1 < argc)
            smParams.sift_features = std::stoi(argv[++i]);
        else if (a == "--sm-ratio" && i+1 < argc) smParams.ratio_threshold = std::stof(argv[++i]);
        else if (a == "--sm-ransac" && i+1 < argc) smParams.ransac_threshold = std::stod(argv[++i]);

        else if (a == "--gt" && i+1 < argc) gt_path = argv[++i];
        else if (a == "--eval-ply" && i+1 < argc) eval_ply_path = argv[++i];
        else if (a == "--zmax" && i+1 < argc) max_depth_mm = std::stof(argv[++i]);
        else {
            std::cerr << "Unknown or incomplete argument: " << a << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Treat defaults or CLI overrides as full-resolution values and always
    // scale them to the resolution used by the pipeline.
    try {
        configureDisparityRangeForScale(mp, scale);
    } catch (const std::exception& e) {
        std::cerr << "Invalid matching parameters: " << e.what() << "\n";
        return 1;
    }
    // ── 1. Load images & calibration ─────────────────────────────────────
    std::cout << "\n=== Loading scene " << scene_id << " views "
              << viewL << "–" << viewR << " ===\n";
    DTUDataLoader loader(data_path.string());
    CalibData calib = loader.loadCalibIntrinsics(viewL, viewR);
    cv::Mat imgL = loader.loadImage(scene_id, viewL);
    cv::Mat imgR = loader.loadImage(scene_id, viewR);
    if (imgL.empty() || imgR.empty()) {
        std::cerr << "Could not load images.\n"; return 1;
    }

    // ── 2. Sparse matching / GT pose ─────────────────────────────────────────────
    std::cout << "\n=== Sparse Matching ===\n";
    if (test_gt_pose) {
        std::cout << "[sparse matching]: Using GT. Skipping Sparse Matching.\n";
    } else if (opencv_sm) {
        std::cout << "[sparse matching]: Running OpenCV pipeline.\n";
    } else {
        std::cout << "[sparse matching]: Running custom 8-point/RANSAC with "
                  << (smParams.use_custom_sift ? "manual SIFT" : "OpenCV SIFT") << "\n";
    }

    if (test_gt_pose) {
        // Load GT relative pose directly from DTU calibration files — skip SIFT/RANSAC.
        loader.loadGTRelativePose(calib, viewL, viewR);
        {
            std::string sparse_path = "results/scene" + scene_id + "/sparse_matching";
            fs::create_directories(sparse_path);
            saveCalibData(calib, sparse_path + "/calib_debug_gt_" + viewL + "_" + viewR + ".yaml");
        }
    } else {
        SiftBackend sift_backend = (!opencv_sm && smParams.use_custom_sift) ? SiftBackend::Custom : SiftBackend::OpenCV;
        FMatrixBackend fmatrix_backend = opencv_sm ? FMatrixBackend::OpenCVFundamental : FMatrixBackend::CustomRansac;
        SparseMatchResult sparse = computeSparseMatches(imgL, imgR, calib, sift_backend, fmatrix_backend, smParams);

        std::string sparse_path = "results/scene" + scene_id + "/sparse_matching";
        fs::create_directories(sparse_path);
        saveCalibData(calib, sparse_path + "/calib_" + viewL + "_" + viewR + ".yaml");
    }

    if (calib.swapped) {
        std::swap(imgL, imgR);
    }

    if (scale != 1.0) {
        cv::resize(imgL, imgL, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(imgR, imgR, cv::Size(), scale, scale, cv::INTER_AREA);
        for (cv::Mat* K : {&calib.K0, &calib.K1}) {
            K->at<double>(0,0) *= scale; K->at<double>(1,1) *= scale;
            K->at<double>(0,2) = (K->at<double>(0,2)+0.5)*scale - 0.5;
            K->at<double>(1,2) = (K->at<double>(1,2)+0.5)*scale - 0.5;
        }

        if (!calib.F.empty()) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    double s = (r < 2 ? scale : 1.0) * (c < 2 ? scale : 1.0);
                    calib.F.at<double>(r,c) /= s;
                }
        }

        std::cout << "Images downscaled to " << imgL.cols << "×" << imgL.rows << "\n";
    }

    // ── 3. Rectification ─────────────────────────────────────────────────
    std::string rect_method_name = !manual_rect ? "OpenCV" : "Loop-Zhang";
    std::cout << "\n=== Rectification (" << rect_method_name << ") ===\n";
    RectifyResult rect;

    rect = !manual_rect ? rectifyOpenCV(imgL, imgR, calib)
                        : rectifyLoopZhang(imgL, imgR, calib);

    std::string rect_path = "results/scene" + scene_id + "/rectification";
    fs::create_directories(rect_path);
    cv::imwrite(rect_path + "/view_" + viewL + ".png", rect.left_rect);
    cv::imwrite(rect_path + "/view_" + viewR + ".png", rect.right_rect);
    cv::imwrite(rect_path + "/mask_" + viewL + ".png", rect.mask1);
    cv::imwrite(rect_path + "/mask_" + viewR + ".png", rect.mask2);

    cv::Mat side;
    cv::hconcat(rect.left_rect, rect.right_rect, side);

    for (int y = 0; y < side.rows; y += 40) {
        const bool validInLeft =
            cv::countNonZero(rect.mask1.row(y)) > 0;

        const bool validInRight =
            cv::countNonZero(rect.mask2.row(y)) > 0;

        if (validInLeft && validInRight) {
            cv::line(side,cv::Point(0, y), cv::Point(side.cols - 1, y), cv::Scalar(0, 255, 0), 1);
        }
    }
    cv::imwrite(rect_path + "/epipolar_lines_" + viewL + "_" + viewR + ".png", side);


    // ── 4. Dense matching ─────────────────────────────────────────────────
    std::cout << "\n=== Dense Matching (ndisp=" << mp.num_disparities
              << " window=" << mp.window_size << ") ===\n";
    cv::Mat disp = computeDisparity(rect.left_rect, rect.right_rect, mp, rect.mask1, rect.mask2);

    cv::Mat disp_vis(disp.size(), CV_8U, cv::Scalar(0));
    const float visMin = static_cast<float>(mp.min_disparity);
    const float visMax = static_cast<float>(mp.min_disparity + mp.num_disparities);
    const cv::Mat validDisp = disp >= visMin;
    disp.convertTo(disp_vis, CV_8U, 255.0f / (visMax - visMin),
                   -255.0f * visMin / (visMax - visMin));
    disp_vis.setTo(0, ~validDisp);
    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);
    disp_vis.setTo(cv::Scalar(0, 0, 0), ~validDisp);

    std::string match_path = "results/scene" + scene_id + "/matching";
    fs::create_directories(match_path);
    cv::imwrite(match_path + "/view_" + viewL + "_" + viewR + "_disparity.png", disp_vis);
    saveDisparity(disp, match_path + "/view_" + viewL + "_" + viewR + "_raw.png");

    // ── 5. Depth & point cloud ────────────────────────────────────────────
    std::cout << "\n=== Point Cloud ===\n";
    float min_disp = std::max(1.0f, (float)mp.min_disparity);
    PointCloud cloud;
    if (manual_rect) {
        // Loop-Zhang produces a general projective warp, not a calibrated
        // rectified camera pair — there is no valid Q matrix (see rectification.cpp).
        // Triangulate per-pixel via the rectified projection matrices instead.
        cloud = disparityToCloudViaP(disp, rect.P1, rect.P2, rect.left_rect,
                                  min_disp, max_depth_mm, rect.mask1, rect.mask2);
    } else {
        cloud = disparityToCloud(disp, rect.Q, rect.left_rect, min_disp, max_depth_mm);
    }

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

    std::string cloud_path = "results/scene" + scene_id + "/pointcloud";
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
            printDtuEval(er, "scene" + scene_id + " views " + viewL + "-" + viewR
                             + " (" + rect_method_name + ")");
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
            printEvalResult(r, "Scene " + scene_id + " views " + viewL + "-" + viewR);
        }
    }

    std::cout << "\n=== Pipeline complete. Results in results/scene"
              << scene_id << "/ ===\n";
    return 0;
}
