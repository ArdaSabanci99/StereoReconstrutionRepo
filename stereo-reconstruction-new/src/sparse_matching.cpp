#include "sparse_matching.h"
#include "DataLoader.h"
#include "Eigen.h"
#include "Ransac.h"
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <random>
#include <numeric>
#include <cmath>

namespace fs = std::filesystem;
constexpr double detTolerance = 1e-6;


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

std::vector<Eigen::Vector3d> normalizePoints(const std::vector<Eigen::Vector3d>& pts, const Eigen::Matrix3d & K) {
    Eigen::Matrix3d K_inv = K.inverse();
    std::vector<Eigen::Vector3d> pts_norm;

    for (const auto & pt : pts) {
        pts_norm.push_back(K_inv * pt);
    }

    return pts_norm;
}

Eigen::Vector3d triangulatePoint(const Eigen::Vector3d & pt_left_norm, const Eigen::Vector3d & pt_right_norm,  const Eigen::Matrix<double, 3, 4> & P_left, const Eigen::Matrix<double, 3, 4> & P_right) {
    Eigen::Matrix4d A;

    // Constraints from left view
        // u = p1.X / p3.X -> u . (p3.X) - (p1.X) = 0 -> (u.p3 - p1)X = 0 -> row in linear system AX = 0
        // v = p2.X / p3.X
    A.row(0) = pt_left_norm.x() * P_left.row(2) - P_left.row(0);
    A.row(1) = pt_left_norm.y() * P_left.row(2) - P_left.row(1);
    
    // Constraints from right view
    A.row(2) = pt_right_norm.x() * P_right.row(2) - P_right.row(0);
    A.row(3) = pt_right_norm.y() * P_right.row(2) - P_right.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

    // smallest singular vector minimizes AX=0
    Eigen::Vector4d X_hom = svd.matrixV().col(3);

    // normalizing by scale factor
    return X_hom.head<3>() / X_hom(3);
}

size_t recoverPose(const std::vector<Eigen::Vector3d>& pts_left_norm, const std::vector<Eigen::Vector3d>& pts_right_norm, 
                    const Eigen::Matrix3d& E,
                    Eigen::Matrix3d & R, Eigen::Vector3d & t,
                    double chirality_depth) {
    // decompose matrix E via SVD
    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if (V.determinant() < 0) {
        V.col(2) *= -1;  // Ensure a proper rotation matrix
    }

    if (U.determinant() < 0) {
        U.col(2) *= -1;  // Ensure a proper rotation matrix
    }

    // Find physically correct setup
        // R: both R = U W V^T & U W V^T satisfy the E
        // t: ambiguity in the direction (t, -t)
        // -> find combination of (R, t) that corresponds to stereo setup: actual 3D scene points are in front of both cameras (positive depth)
    
    Eigen::Matrix3d W;
    W << 0, -1, 0,
        1, 0, 0,
        0, 0, 1;

    Eigen::Matrix3d R1 = U * W * V.transpose();
    Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();

    if (std::abs(R1.determinant() - 1.0) >= detTolerance) {
        throw std::runtime_error("R1 is not a proper rotation matrix!");
    }
    if (std::abs(R2.determinant() - 1.0) >= detTolerance) {
        throw std::runtime_error("R2 is not a proper rotation matrix!");
    }

    Eigen::Vector3d t1 = U.col(2);
    Eigen::Vector3d t2 = -U.col(2);

    // Chirelity test
    std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> candidates = {
        {R1, t1},
        {R1, t2},
        {R2, t1},
        {R2, t2}
    };
    // projection matrix of the left camera
    Eigen::Matrix<double, 3, 4> P_left = Eigen::Matrix<double, 3, 4>::Zero();
    P_left.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    
    size_t best_inlier_count = 0;
    size_t best_candidate_idx = 0;
    for (size_t candidate_idx = 0; candidate_idx < candidates.size(); candidate_idx++) {
        const auto & [R_candidate, t_candidate] = candidates[candidate_idx];
        // projection matrix of the right camera in the left camera's coordinate system
        Eigen::Matrix<double, 3, 4> P_right;
        P_right.block<3, 3>(0, 0) = R_candidate;  // candidate rotation
        P_right.col(3) = t_candidate;  // candidate translation

        size_t cur_inlier_count = 0;
        
        for (size_t i = 0; i < pts_left_norm.size(); ++i) {
            Eigen::Vector3d X_left = triangulatePoint(pts_left_norm[i], pts_right_norm[i], P_left, P_right);
            double Z_left = X_left.z();
            
            Eigen::Vector3d X_right = R_candidate * X_left + t_candidate;
            double Z_right = X_right.z();
            
            // Check if both positives
            // Checking if smaller than chirality depth (if reliable)
                // For almost parallel rays
            if (Z_left > 0 && Z_right > 0 && Z_left < chirality_depth && Z_right < chirality_depth) {
                cur_inlier_count++;
            }
        }

        if (cur_inlier_count > best_inlier_count) {
            best_inlier_count = cur_inlier_count;
            best_candidate_idx = candidate_idx;
        }

        std::cout << "[sparse matching - recover pose]: Candidate " << candidate_idx + 1 << " has votes (n_inliers) " << cur_inlier_count << std::endl;

    }

    std::cout << "[sparse matching - recover pose]: Best Candidate " << best_candidate_idx + 1 << std::endl;
    
    std::tie(R, t) = candidates[best_candidate_idx];
    
    return best_inlier_count;

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

SparseMatchResult computeSparseMatchesCustom(const cv::Mat & imgLeft, const cv::Mat & imgRight, const CalibData & calib, const SparseMatchParams & params) {
    SparseMatchResult result;

    // Convert to grayscale for SIFT
    cv::Mat grayL, grayR;
    cv::cvtColor(imgLeft,  grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgRight, grayR, cv::COLOR_BGR2GRAY);

    // 1. Keypoint Detection using SIFT
        // TODO: implement manual SIFT
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

    for (auto & m : knn) {
        if (m.size() >= 2 && m[0].distance < params.ratio_threshold * m[1].distance) {
            result.pts_left.push_back(kpL[m[0].queryIdx].pt);
            result.pts_right.push_back(kpR[m[0].trainIdx].pt);
            matchDist.push_back(m[0].distance);  // For substep
        }
    }
    result.n_matches = (int)result.pts_left.size();


    // 4. Estimate fundamental matrix with RANSAC.
    Ransac ransac(params.custom_ransac_iters, params.ransac_threshold, 8);

    std::vector<Eigen::Vector3d> eigen_pts_left, eigen_pts_right;
    for (const auto & pt : result.pts_left) {
        eigen_pts_left.emplace_back(pt.x, pt.y, 1.0);
    }
    for (const auto & pt : result.pts_right) {
        eigen_pts_right.emplace_back(pt.x, pt.y, 1.0);
    }

    RansacResult ransac_result = ransac.findFundamentalMatrix(eigen_pts_left, eigen_pts_right);
    cv::eigen2cv(ransac_result.F, result.F);
    result.n_inliers = ransac_result.n_inliers;
    result.inlier_mask = ransac_result.inlier_mask;
    result.mean_epipolar_error = ransac_result.mean_epipolar_error;

    // Retrieve inliers
    std::vector<Eigen::Vector3d> inl_left, inl_right;
    for (size_t i = 0; i < eigen_pts_left.size(); i++) {
        if (result.inlier_mask[i]) {
            inl_left.push_back(eigen_pts_left[i]);
            inl_right.push_back(eigen_pts_right[i]);
        }
    }


    // 5. Derive essential matrix E from F and intrinsics.
    // E = K1^T · F · K0  (see derivation: F relates pixel coords, E relates normalised coords)
    cv::Mat K0 = calib.K0, K1 = calib.K1;
    Eigen::Matrix3d K0_eigen, K1_eigen;
    cv::cv2eigen(K0, K0_eigen);
    cv::cv2eigen(K1, K1_eigen);
    
    result.E = K1.t() * result.F * K0;  // is already cv::Mat
    


    // 6. Recover pose
    std::vector<Eigen::Vector3d> inl_left_norm = normalizePoints(inl_left, K0_eigen);
    std::vector<Eigen::Vector3d> inl_right_norm = normalizePoints(inl_right, K1_eigen);

    Eigen::Matrix3d E_eigen;
    cv::cv2eigen(result.E, E_eigen);
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    result.n_pose_inliers = recoverPose(inl_left_norm, inl_right_norm, E_eigen, R_eigen, t_eigen, params.chirality_depth);
    cv::eigen2cv(R_eigen, result.R);
    cv::eigen2cv(t_eigen, result.t);

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
