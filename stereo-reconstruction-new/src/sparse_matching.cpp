#include "sparse_matching.h"
#include "DataLoader.h"
#include "Eigen.h"
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <filesystem>
#include <random>
#include <numeric>
#include <cmath>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Sampson distance (first-order geometric error)
//   d = sqrt( (x2^T F x1)^2 / ( (Fx1)[0]^2 + (Fx1)[1]^2 + (F^T x2)[0]^2 + (F^T x2)[1]^2 ) )
//   sqrt is applied so the result is in pixels (same units as the RANSAC threshold)
// ─────────────────────────────────────────────────────────────────────────────
double sampsonDistance(const cv::Point2f & pt_left, const cv::Point2f & pt_right, const cv::Mat & F) {
    cv::Mat x_left = (cv::Mat_<double>(3,1) << pt_left.x, pt_left.y, 1.0);
    cv::Mat x_right = (cv::Mat_<double>(3,1) << pt_right.x, pt_right.y, 1.0);

    cv::Mat l_right  = F  * x_left;
    cv::Mat l_left = F.t() * x_right;
    double  num  = cv::Mat(x_right.t() * l_right).at<double>(0,0);
    double  den  = l_right.at<double>(0)*l_right.at<double>(0)
                 + l_right.at<double>(1)*l_right.at<double>(1)
                 + l_left.at<double>(0)*l_left.at<double>(0)
                 + l_left.at<double>(1)*l_left.at<double>(1);
    return std::sqrt((num * num) / (den + 1e-10));
}

    }






}



            }
        }

        }



}


SparseMatchResult computeSparseMatches(cv::Mat& imgLeft, cv::Mat& imgRight, CalibData & calib, bool run_opencv, const SparseMatchParams& params) {
    SparseMatchResult result;
    try {
        result = run_opencv ? computeSparseMatchesOpenCV(imgLeft, imgRight, calib, params) : computeSparseMatchesCustom(imgLeft, imgRight, calib, params);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("[sparse matching] Sparse matching failed: ") + e.what());
    }

    if (result.R.empty()) {
        throw std::runtime_error(std::string("[sparse matching] Pose recovery failed. Sparse matching (")
            + (run_opencv ? "OpenCV" : "manual implementation")
            + ") produces empty R.\n");
    }

    calib.R_rel = result.R;
    calib.t_rel = result.t * calib.baseline;  // result.t is unit-length; scale to metric here
    calib.F = result.F;

    calib.verifyLeftRightCameraOrder();

    std::cout << "[sparse matching]: Sparse matching (" << (run_opencv ? "OpenCV" : "Manual") << ") finished." << std::endl;

    return result;
};


SparseMatchResult computeSparseMatchesOpenCV(const cv::Mat& imgLeft, const cv::Mat& imgRight, const CalibData& calib, const SparseMatchParams& params) {
    SparseMatchResult result;

    // Convert to grayscale for SIFT
    cv::Mat grayL, grayR;
    cv::cvtColor(imgLeft,  grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgRight, grayR, cv::COLOR_BGR2GRAY);

    // 1. Keypoint Detection using SIFT
        // TODO: Add multiple descriptors (e.g. ORB)
    auto sift = cv::SIFT::create(params.sift_features, params.sift_octave_layers,
                                  params.sift_contrast_thresh, params.sift_edge_thresh,
                                  params.sift_sigma);

    std::vector<cv::KeyPoint> kpL, kpR;
    cv::Mat descL, descR;
    sift->detectAndCompute(grayL, cv::noArray(), kpL, descL);
    sift->detectAndCompute(grayR, cv::noArray(), kpR, descR);

    // 2. Match descriptors using BFMatches
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(descL, descR, knn, 2);

    // 3. Find good matches using Lowe's ratio test
    std::vector<float> matchDist;

    for (auto& m : knn) {
        if (m.size() >= 2 && m[0].distance < params.ratio_threshold * m[1].distance) {
            result.pts_left.push_back(kpL[m[0].queryIdx].pt);
            result.pts_right.push_back(kpR[m[0].trainIdx].pt);
            matchDist.push_back(m[0].distance);  // For substep
        }
    }
    result.n_matches = (int)result.pts_left.size();


    // 4. Estimate fundamental matrix with RANSAC.

    if (result.pts_left.size() < 8) {
        throw std::runtime_error(std::string("[sparse matching] Not enough points (<8) to run 8-point algorithm."));
    }
    result.F = cv::findFundamentalMat(result.pts_left, result.pts_right,
                                      cv::FM_RANSAC, params.ransac_threshold, 0.999, result.inlier_mask);

    if (result.F.empty()) {
        throw std::runtime_error(std::string("[sparse matching] RANSAC failed to find a valid model."));
    }

    // Collect RANSAC inlier pairs.
    std::vector<cv::Point2f> inl_left, inl_right;
    double total_err = 0.0;
    for (size_t i = 0; i < result.pts_left.size(); ++i)
        if (result.inlier_mask[i]) {
            cv::Point2f pt_inl_left = result.pts_left[i];
            cv::Point2f pt_inl_right = result.pts_right[i];
            inl_left.push_back(pt_inl_left);
            inl_right.push_back(pt_inl_right);
            
            // Compute mean Sampson epipolar error over RANSAC inliers
            total_err += sampsonDistance(pt_inl_left, pt_inl_right, result.F);
            result.n_inliers++;
        }

    result.mean_epipolar_error = result.n_inliers > 0 ? total_err / result.n_inliers : 0.0;

    // 5. Derive essential matrix E from F and intrinsics.
    // E = K1^T · F · K0  (see derivation: F relates pixel coords, E relates normalised coords)
    cv::Mat K0 = calib.K0, K1 = calib.K1;
    result.E = K1.t() * result.F * K0;


    // 6. Recover pose
    std::vector<cv::Point2f> inl_left_norm, inl_right_norm;
    cv::undistortPoints(inl_left,  inl_left_norm,  K0, cv::noArray()); // no known distortion coefficients -> noArray()
    cv::undistortPoints(inl_right, inl_right_norm, K1, cv::noArray());


    std::vector<uchar> pose_mask(inl_left.size(), 1);
    result.n_pose_inliers = cv::recoverPose(result.E, inl_left_norm, inl_right_norm,
        cv::Mat::eye(3, 3, CV_64F),
        result.R, result.t,
        params.chirality_depth,
        pose_mask
    );


    return result;
};

    }

    // ── 4. Mean epipolar error on inliers ─────────────────────────────────
    double err_sum = 0; int n = 0;
    for (int i = 0; i < result.n_matches; ++i) {
        if (!result.inlier_mask[i]) continue;
        err_sum += sampsonDistance(result.F, result.pts_left[i], result.pts_right[i]);
        ++n;
    }
    result.mean_epipolar_error = (n > 0) ? err_sum / n : 0.0;
    std::cout << "[sparse] Mean Sampson error = " << result.mean_epipolar_error << "\n";

    // ── 5. Essential matrix ────────────────────────────────────────────────
    result.E = calib.K1.t() * result.F * calib.K0;

    // ── 6. Recover R, t (cheirality test) ─────────────────────────────────
    std::vector<cv::Point2f> in_l, in_r;
    for (int i = 0; i < result.n_matches; ++i)
        if (result.inlier_mask[i]) {
            in_l.push_back(result.pts_left[i]);
            in_r.push_back(result.pts_right[i]);
        }

    recoverRtFromE(result.E, in_l, in_r, calib.K0, result.R, result.t);

    std::cout << "[sparse] R =\n" << result.R << "\n"
              << "[sparse] t =  " << result.t.t() << "\n";

    return result;
};

void saveMatchVisualization(const cv::Mat& imgLeft, const cv::Mat& imgRight,
                             const SparseMatchResult& result,
                             const std::string& saveDir,
                             const std::string& runName,
                             const std::string& viewL,
                             const std::string& viewR,
                             double rotErrDeg,
                             double transErrDeg,
                             bool inliersOnly) {
    fs::create_directories(saveDir);

    const std::string imgPath = saveDir + "/matches_" + runName + "_"
                                + viewL + "_" + viewR + ".png";

    // ── Visualization ────────────────────────────────────────────────────────
    // Collect the point pairs to draw, respecting inlier filter and cap.
    std::vector<cv::Point2f> drawL, drawR;
    for (size_t i = 0; i < result.pts_left.size(); ++i) {
        if (inliersOnly && !result.inlier_mask.empty() && !result.inlier_mask[i]) continue;
        drawL.push_back(result.pts_left[i]);
        drawR.push_back(result.pts_right[i]);
    }
    if (drawL.empty()) {
        std::cout << "[sparse_matching] No matches to visualize.\n";
    } else {
        // Draw manually for thickness control: images side-by-side, circles + connecting lines.
        cv::Mat vis;
        cv::hconcat(imgLeft, imgRight, vis);
        const int offset = imgLeft.cols;
        cv::RNG rng(0);
        for (size_t i = 0; i < drawL.size(); ++i) {
            cv::Scalar color(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
            cv::Point2f ptR(drawR[i].x + offset, drawR[i].y);
            cv::circle(vis, drawL[i], 4, color, 2);
            cv::circle(vis, ptR,      4, color, 2);
            cv::line(vis, drawL[i], ptR, color, 1);
        }
        cv::imwrite(imgPath, vis);
        std::cout << "[sparse_matching] Match visualisation saved to " << imgPath << "\n";
    }

    // ── CSV log ──────────────────────────────────────────────────────────────
    const std::string logPath  = saveDir + "/results_log.csv";
    const bool        logExists = fs::exists(logPath);

    // Skip if this exact (run_name, viewL, viewR) tuple is already recorded,
    // so a stable baseline is never accidentally overwritten.
    const std::string key = runName + "," + viewL + "," + viewR;
    if (logExists) {
        std::ifstream fin(logPath);
        std::string line;
        while (std::getline(fin, line)) {
            if (line.rfind(key, 0) == 0) {
                std::cout << "[sparse_matching] '" << runName << "' already in log; skipping.\n";
                return;
            }
        }
    }

    std::ofstream fout(logPath, std::ios::app);
    if (!logExists)
        fout << "run_name,view_L,view_R,inliers_only,max_matches,"
                "n_matches,n_inliers,n_pose_inliers,mean_epipolar_error_px,"
                "rot_err_deg,trans_err_deg\n";

    fout << key << ","
         << (inliersOnly ? 1 : 0) << ","
         << result.n_matches << ","
         << result.n_inliers << ","
         << result.n_pose_inliers << ","
         << std::fixed << std::setprecision(4) << result.mean_epipolar_error << ",";
    if (rotErrDeg >= 0.0)
        fout << std::fixed << std::setprecision(4) << rotErrDeg << ","
             << std::fixed << std::setprecision(4) << transErrDeg << "\n";
    else
        fout << "N/A,N/A\n";

    std::cout << "[sparse_matching] Results logged to " << logPath << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone executable
// ─────────────────────────────────────────────────────────────────────────────
#if !defined(PIPELINE_BUILD) && !defined(BUILDING_RECTIFICATION)
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: sparse_matching <data_path> <scene_id> <left_id> <right_id> [--light N]\n";
        return 1;
    }
    fs::path dataPath(argv[1]);
    const std::string sceneId(argv[2]);
    const std::string viewL = padViewId(std::stoi(argv[3]));
    const std::string viewR = padViewId(std::stoi(argv[4]));
    std::string lightId = "0";
    for (int i = 5; i < argc; ++i)
        if (std::string(argv[i]) == "--light" && i+1 < argc) lightId = argv[++i];

    DTUDataLoader loader(dataPath.string());
    CalibData calib = loader.loadCalib(viewL, viewR);
    cv::Mat imgL = loader.loadImage(sceneId, viewL, lightId);
    cv::Mat imgR = loader.loadImage(sceneId, viewR, lightId);
    if (imgL.empty() || imgR.empty()) { std::cerr << "Could not load images.\n"; return 1; }

    SparseMatchResult r = computeSparseMatches(imgL, imgR, calib);

    // Save match visualisation
    std::vector<cv::KeyPoint> kpL, kpR;
    std::vector<cv::DMatch>   good;
    // Re-detect for visualisation
    auto sift = cv::SIFT::create(3000);
    cv::Mat dL, dR;
    sift->detectAndCompute(imgL, cv::noArray(), kpL, dL);
    sift->detectAndCompute(imgR, cv::noArray(), kpR, dR);
    // Draw first 50 inlier matches approximately
    cv::Mat vis;
    if (!kpL.empty() && !kpR.empty()) {
        std::vector<cv::DMatch> dm;
        cv::BFMatcher bfm(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> knn;
        bfm.knnMatch(dL, dR, knn, 2);
        for (auto& m : knn)
            if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
                dm.push_back(m[0]);
        std::vector<cv::DMatch> subset(dm.begin(), dm.begin() + std::min((int)dm.size(), 50));
        cv::drawMatches(imgL, kpL, imgR, kpR, subset, vis);
        std::string savePath = "results/scene" + sceneId + "/sparse";
        fs::create_directories(savePath);
        cv::imwrite(savePath + "/matches_" + viewL + "_" + viewR + ".png", vis);
        std::cout << "Match visualisation saved.\n";
    }
    return 0;
}
#endif
