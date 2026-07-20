#include "sparse_matching.h"
#include "DataLoader.h"
#include "Eigen.h"
#include "Ransac.h"
#include "KeypointDetector.h"
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <random>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>

namespace fs = std::filesystem;


/**
 * @brief Handles detection of SIFT features and computation of descriptors using a custom SIFT implementation.
 * 
 * @param[in] gray Gray-scale image.
 * @param[out] keypoints Vector to store detected keypoints.
 * @param[out] descriptors 
 * @param[in] params 
 */
static void detectCustomSift(const cv::Mat & gray, std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors,
                              const SparseMatchParams & params) {
    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F);
    Eigen::MatrixXf img_eigen;
    cv::cv2eigen(gray_f, img_eigen);

    SIFT sift(params.sift_octave_layers, params.sift_sigma, params.sift_features,
              params.sift_contrast_thresh, params.sift_edge_thresh);
    std::vector<Keypoint> kps = sift.detect_features(img_eigen);
    
    // Transforming to OpenCV's KeyPoint for compatibility
    keypoints.clear();
    keypoints.reserve(kps.size());
    descriptors.create(static_cast<int>(kps.size()), 128, CV_32F);

    for (size_t i = 0; i < kps.size(); ++i) {
        const Keypoint & kp = kps[i];
        keypoints.emplace_back(kp.y, kp.x, static_cast<float>(kp.sigma), kp.orientation_deg);
        for (int d = 0; d < 128; ++d)
            descriptors.at<float>(static_cast<int>(i), d) = kp.descriptor[d];
    }
}

/**
 * @brief Runs a brute-force matching of SIFT descriptors between two images.
 *        Applies Lowe's ratio test to filter matches.
 * 
 * @param[in] desc_left Descriptors of the keypoints from left image. 
 * @param[in] desc_right Descriptors of the keypoints from right image. 
 * @param[in] kp_left Left image keypoints.
 * @param[in] kp_right Right image keypoints.
 * @param[in] params Sparse matching parameters.
 * @param[out] result Sparse matching result. 
 */
static void brute_force_match(const cv::Mat & desc_left, const cv::Mat & desc_right,
                               const std::vector<cv::KeyPoint> & kp_left, const std::vector<cv::KeyPoint> & kp_right,
                               const SparseMatchParams & params, SparseMatchResult & result) {
    for (int i = 0; i < desc_left.rows; ++i) {
        const float * query_desc = desc_left.ptr<float>(i);

        float best_dist_sq = std::numeric_limits<float>::max();
        float second_best_dist_sq = std::numeric_limits<float>::max();
        int best_idx = -1;

        // Scan every candidate in the right image
            // Tracking the best and second-best match
        for (int j = 0; j < desc_right.rows; ++j) {
            const float * candidate_desc = desc_right.ptr<float>(j);

            float dist_sq = 0.0f;
            for (int d = 0; d < desc_left.cols; ++d) {
                float diff = query_desc[d] - candidate_desc[d];
                dist_sq += diff * diff;
            }

            if (dist_sq < best_dist_sq) {
                second_best_dist_sq = best_dist_sq;
                best_dist_sq = dist_sq;
                best_idx = j;
            } else if (dist_sq < second_best_dist_sq) {
                second_best_dist_sq = dist_sq;
            }
        }

        // Lowe's ratio test
            // Accept if difference between first and second best candidate is big enough
        bool has_two_candidates = best_idx >= 0 && second_best_dist_sq < std::numeric_limits<float>::max();
        if (has_two_candidates && best_dist_sq < (params.ratio_threshold * params.ratio_threshold) * second_best_dist_sq) {
            result.pts_left.push_back(kp_left[i].pt);
            result.pts_right.push_back(kp_right[best_idx].pt);
        }
    }
}

SparseMatchResult computeSparseMatches(cv::Mat& left, cv::Mat& right, CalibData & calib,
                                        SiftBackend sift_backend, FMatrixBackend fmatrix_backend,
                                        const SparseMatchParams& params) {
    SparseMatchResult result;
    try {
        // Timing of SIFT detection and matching
        auto sift_start = std::chrono::steady_clock::now();
        result = detectSiftMatches(left, right, params, sift_backend);
        double sift_time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - sift_start
        ).count();

        auto fmatrix_start = std::chrono::steady_clock::now();

        // Timing of 8-point AG
        if (fmatrix_backend == FMatrixBackend::OpenCVFundamental) {
            computeSparseMatchesOpenCV(result, calib, params);
        } else {
            computeSparseMatchesCustom(result, calib, params);
        }
        double fmatrix_time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fmatrix_start
        ).count();

        result.sift_time_ms = sift_time_ms;
        result.fmatrix_time_ms = fmatrix_time_ms;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("[sparse matching] Sparse matching failed: ") + e.what());
    }

    if (result.R.empty()) {
        throw std::runtime_error("[sparse matching] Pose recovery failed: produced empty R.\n");
    }

    calib.R_rel = result.R;
    calib.t_rel = result.t * calib.baseline;  // result.t is unit-length; scaling to metric
    calib.F = result.F;

    calib.verifyLeftRightCameraOrder();

    std::cout << "[sparse_matching] Matches after ratio test: " << result.n_matches << "\n";
    std::cout << "[sparse_matching] RANSAC inliers: " << result.n_inliers << "/" << result.n_matches << "\n";
    std::cout << "[sparse_matching] Mean Sampson error (inliers): " << result.mean_epipolar_error << " px\n";
    std::cout << "[sparse_matching] Pose cheirality inliers: " << result.n_pose_inliers
              << " / RANSAC inliers: " << result.n_inliers << "\n";
    std::cout << "[sparse_matching] Sparse matching finished (SIFT="
              << (sift_backend == SiftBackend::Custom ? "custom" : "OpenCV")
              << ", estimator=" << (fmatrix_backend == FMatrixBackend::CustomRansac ? "custom RANSAC" : "OpenCV") << ").\n";

    return result;
};

void computeSparseMatchesOpenCV(SparseMatchResult& result, const CalibData& calib, const SparseMatchParams& params) {
    if (result.pts_left.size() < 8) {
        throw std::runtime_error(std::string("[sparse matching] Not enough points (<8) to run 8-point algorithm."));
    }
    result.F = cv::findFundamentalMat(result.pts_left, result.pts_right,
                                      cv::USAC_FM_8PTS, params.ransac_threshold, 0.999, result.inlier_mask);

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

    // TODO: refit F from all inliers (minimal 8-point sample is high-variance) -- verify this closes the opencv:opencv pose-recovery gap seen in run_sparse_matching_study.sh.
    result.F = cv::findFundamentalMat(inl_left, inl_right, cv::FM_8POINT);

    // Derive essential matrix E from F and intrinsics
        // E = K1^T · F · K0
    cv::Mat K0 = calib.K0, K1 = calib.K1;
    result.E = K1.t() * result.F * K0;


    // Recover pose
    std::vector<cv::Point2f> inl_left_norm, inl_right_norm;
    cv::undistortPoints(inl_left,  inl_left_norm,  K0, cv::noArray()); // No known distortion coefficients -> noArray()
    cv::undistortPoints(inl_right, inl_right_norm, K1, cv::noArray());


    std::vector<uchar> pose_mask(inl_left.size(), 1);
    result.n_pose_inliers = cv::recoverPose(result.E, inl_left_norm, inl_right_norm,
        cv::Mat::eye(3, 3, CV_64F),
        result.R, result.t,
        params.chirality_depth,
        pose_mask
    );
};


void computeSparseMatchesCustom(SparseMatchResult& result, const CalibData & calib, const SparseMatchParams & params) {
    if (result.pts_left.size() < 8) {
        throw std::runtime_error(std::string("[sparse matching] Not enough points (<8) to run 8-point algorithm."));
    }

    // Estimate fundamental matrix with custom RANSAC + 8-point AG
    Ransac ransac(params.custom_ransac_iters, params.ransac_threshold);

    // Transforming OpenCV points to Eigen for custom RANSAC
    std::vector<Eigen::Vector3d> eigen_pts_left, eigen_pts_right;
    for (const auto & pt : result.pts_left) {
        eigen_pts_left.emplace_back(pt.x, pt.y, 1.0);
    }
    for (const auto & pt : result.pts_right) {
        eigen_pts_right.emplace_back(pt.x, pt.y, 1.0);
    }

    // Run custom RANSAC to estimate the fundamental matrix
    RansacResult ransac_result = ransac.findFundamentalMatrix(eigen_pts_left, eigen_pts_right);
    cv::eigen2cv(ransac_result.F, result.F);  // Transforming to OpenCV for compatibility between OpenCV & Custom Sparse Matching
    
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

    // Derive essential matrix E from F and intrinsics.
        // E = K1^T · F · K0
    Eigen::Matrix3d K0_eigen, K1_eigen, F_eigen;
    cv::cv2eigen(calib.K0, K0_eigen);
    cv::cv2eigen(calib.K1, K1_eigen);
    cv::cv2eigen(result.F, F_eigen);

    Eigen::Matrix3d E_eigen = K1_eigen.transpose() * F_eigen * K0_eigen;
    cv::eigen2cv(E_eigen, result.E);

    // 6. Recover pose
    std::vector<Eigen::Vector3d> inl_left_norm = normalizePoints(inl_left, K0_eigen);
    std::vector<Eigen::Vector3d> inl_right_norm = normalizePoints(inl_right, K1_eigen);

    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    
    result.n_pose_inliers = recoverPose(inl_left_norm, inl_right_norm, E_eigen, R_eigen, t_eigen, params.chirality_depth);
    cv::eigen2cv(R_eigen, result.R);
    cv::eigen2cv(t_eigen, result.t);
};


SparseMatchResult detectSiftMatches(const cv::Mat & img_left, const cv::Mat & img_right,
                                     const SparseMatchParams & params, SiftBackend sift_backend) {
    SparseMatchResult result;

    cv::Mat gray_l, gray_r;
    cv::cvtColor(img_left,  gray_l, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_right, gray_r, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> kp_l, kp_r;
    cv::Mat desc_l, desc_r;

    // In both cases: detect SIFT features + brute-force matcher
    if (sift_backend == SiftBackend::Custom) {  // Custom SIFT Implementation
        detectCustomSift(gray_l, kp_l, desc_l, params);
        detectCustomSift(gray_r, kp_r, desc_r, params);
        std::cout << "[sparse_matching] Custom SIFT keypoints: left=" << kp_l.size()
                  << " right=" << kp_r.size() << "\n";

        brute_force_match(desc_l, desc_r, kp_l, kp_r, params, result);
    } else {  // OpenCV SIFT Implementation
        auto sift = cv::SIFT::create(params.sift_features, params.sift_octave_layers,
                                      params.sift_contrast_thresh, params.sift_edge_thresh,
                                      params.sift_sigma);
        sift->detectAndCompute(gray_l, cv::noArray(), kp_l, desc_l);
        sift->detectAndCompute(gray_r, cv::noArray(), kp_r, desc_r);

        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(desc_l, desc_r, knn, 2);

        for (auto & m : knn) {
            if (m.size() >= 2 && m[0].distance < params.ratio_threshold * m[1].distance) {
                result.pts_left.push_back(kp_l[m[0].queryIdx].pt);
                result.pts_right.push_back(kp_r[m[0].trainIdx].pt);
            }
        }
    }
    result.n_matches = (int)result.pts_left.size();

    return result;
}

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

size_t recoverPose(const std::vector<Eigen::Vector3d>& pts_left_norm, const std::vector<Eigen::Vector3d>& pts_right_norm,
                    const Eigen::Matrix3d& E,
                    Eigen::Matrix3d & R, Eigen::Vector3d & t,
                    double chirality_depth) {
    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // U, V must be orthogonal matrices
        // Ensure a proper rotation matrix
    if (V.determinant() < 0) {
        V.col(2) *= -1;
    }

    if (U.determinant() < 0) {
        U.col(2) *= -1;
    }

    // Find physically correct setup
        // R: both R = U W V^T & U W V^T satisfy the E
        // t: ambiguity in the direction (t, -t)
        // Need find combination of (R, t) that corresponds to stereo setup
    
    Eigen::Matrix3d W;
    W << 0, -1, 0,
        1, 0, 0,
        0, 0, 1;

    Eigen::Matrix3d R1 = U * W * V.transpose();
    Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();

    Eigen::Vector3d t1 = U.col(2);
    Eigen::Vector3d t2 = -U.col(2);

    // Count number of points in front of both cameras for each candidate (R, t)
    std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> candidates = {
        {R1, t1},
        {R1, t2},
        {R2, t1},
        {R2, t2}
    };
    // Left camera stays fixed
        // E gives the right camera's rotation and translation relative to it
    Eigen::Matrix<double, 3, 4> P_left = Eigen::Matrix<double, 3, 4>::Zero();
    P_left.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    
    size_t best_inlier_count = 0;
    size_t best_candidate_idx = 0;
    for (size_t candidate_idx = 0; candidate_idx < candidates.size(); candidate_idx++) {
        const auto & [R_candidate, t_candidate] = candidates[candidate_idx];
        // Projection matrix of the right camera in the left camera's coordinate system
        Eigen::Matrix<double, 3, 4> P_right;

        P_right.block<3, 3>(0, 0) = R_candidate;  // Candidate rotation
        P_right.col(3) = t_candidate;  // Candidate translation

        size_t cur_inlier_count = 0;

        for (size_t i = 0; i < pts_left_norm.size(); ++i) {
            // Triangulate the point in the left camera's frame
            Eigen::Vector3d X_left = triangulatePoint(pts_left_norm[i], pts_right_norm[i], P_left, P_right);
            double Z_left = X_left.z();

            // Move it into the right camera's frame using this candidate's R, t
            Eigen::Vector3d X_right = R_candidate * X_left + t_candidate;
            double Z_right = X_right.z();

            // Valid point is in front of both cameras
            // Rule out out near-parallel rays with unreliable depth
            if (Z_left > 0 && Z_right > 0 && Z_left < chirality_depth && Z_right < chirality_depth) {
                cur_inlier_count++;
            }
        }

        if (cur_inlier_count > best_inlier_count) {
            best_inlier_count = cur_inlier_count;
            best_candidate_idx = candidate_idx;
        }

        std::cout << "[sparse matching]: Recover pose candidate " << candidate_idx + 1 << " has votes (n_inliers) " << cur_inlier_count << std::endl;
    }

    if (best_inlier_count == 0) {
        throw std::runtime_error("[sparse matching] Pose recovery failed: no candidate (R,t) has valid depth.");
    }

    std::cout << "[sparse matching]: Recover pose best candidate " << best_candidate_idx + 1 << std::endl;
    std::tie(R, t) = candidates[best_candidate_idx];

    return best_inlier_count;

}

std::vector<Eigen::Vector3d> normalizePoints(const std::vector<Eigen::Vector3d>& pts, const Eigen::Matrix3d & K) {
    Eigen::Matrix3d K_inv = K.inverse();
    std::vector<Eigen::Vector3d> pts_norm;

    for (const auto & pt : pts) {
        pts_norm.push_back(K_inv * pt);
    }

    return pts_norm;
}


// ─────────────────────────────────────────────────────────────────────────────
// Standalone executable
// ─────────────────────────────────────────────────────────────────────────────
#if !defined(PIPELINE_BUILD) && !defined(BUILDING_RECTIFICATION)
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr
            << "Usage: sparse_matching <data_path> <scene_id> <left_id> <right_id> [options]\n"
            << "  --opencv               (use OpenCV sparse matching; default: manual)\n"
            << "  --custom-sift          (manual pipeline only: use our own SIFT detector\n"
            << "                          instead of cv::SIFT for keypoint detection)\n"
            << "  --features  <N>        SIFT max keypoints, 0=unlimited (default: 0)\n"
            << "  --octaves   <N>        SIFT octave layers (default: 5)\n"
            << "  --contrast  <F>        SIFT contrast threshold (default: 0.04)\n"
            << "  --edge      <F>        SIFT edge threshold (default: 10.0)\n"
            << "  --sigma     <F>        SIFT Gaussian sigma (default: 1.6)\n"
            << "  --ratio     <F>        Lowe ratio test threshold (default: 0.75)\n"
            << "  --ransac    <F>        RANSAC Sampson distance threshold px (default: 1.0)\n"
            << "  --chirality <F>        recoverPose max depth (default: 25.0)\n";
        return 1;
    }

    fs::path data_path(argv[1]);
    const std::string scene_id(argv[2]);
    const std::string viewL = padViewId(std::stoi(argv[3]));
    const std::string viewR = padViewId(std::stoi(argv[4]));
    bool run_opencv = false;
    SparseMatchParams smParams;

    for (int i = 5; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--opencv") run_opencv = true;
        else if (a == "--custom-sift") smParams.use_custom_sift = true;
        else if (a == "--features" && i+1 < argc) smParams.sift_features = std::stoi(argv[++i]);
        else if (a == "--octaves" && i+1 < argc) smParams.sift_octave_layers = std::stoi(argv[++i]);
        else if (a == "--contrast" && i+1 < argc) smParams.sift_contrast_thresh = std::stod(argv[++i]);
        else if (a == "--edge" && i+1 < argc) smParams.sift_edge_thresh = std::stod(argv[++i]);
        else if (a == "--sigma" && i+1 < argc) smParams.sift_sigma = std::stod(argv[++i]);
        else if (a == "--ratio" && i+1 < argc) smParams.ratio_threshold = std::stof(argv[++i]);
        else if (a == "--ransac" && i+1 < argc) smParams.ransac_threshold = std::stod(argv[++i]);
        else if (a == "--chirality" && i+1 < argc) smParams.chirality_depth = std::stod(argv[++i]);
    }

    // Load calibration and images for the requested view pair.
    DTUDataLoader loader(data_path.string());
    CalibData calib = loader.loadCalibIntrinsics(viewL, viewR);
    cv::Mat imgL = loader.loadImage(scene_id, viewL);
    cv::Mat imgR = loader.loadImage(scene_id, viewR);

    if (imgL.empty() || imgR.empty()) {
        std::cerr << "Could not load images.\n";
        return 1;
    }

    SiftBackend sift_backend = (!run_opencv && smParams.use_custom_sift) ? SiftBackend::Custom : SiftBackend::OpenCV;
    FMatrixBackend fmatrix_backend = run_opencv ? FMatrixBackend::OpenCVFundamental : FMatrixBackend::CustomRansac;
    SparseMatchResult result = computeSparseMatches(imgL, imgR, calib, sift_backend, fmatrix_backend, smParams);

    std::string save_path = "results/scene" + scene_id + "/sparse_matching";
    fs::create_directories(save_path);

    saveCalibData(calib, save_path + "/calib_" + viewL + "_" + viewR + ".yaml");
    std::cout << "[sparse_matching] Calib saved to " << save_path << "\n";

    std::cout << "[sparse_matching] Matches after ratio test: " << result.n_matches << "\n";
    std::cout << "[sparse_matching] RANSAC inliers: " << result.n_inliers << "/" << result.n_matches << "\n";
    std::cout << "[sparse_matching] Mean Sampson error (inliers): " << result.mean_epipolar_error << " px\n";
    std::cout << "[sparse_matching] Pose cheirality inliers: " << result.n_pose_inliers
              << " / RANSAC inliers: " << result.n_inliers << "\n";

    return 0;
}
#endif
