#pragma once
#include "utils.h"
#include "Eigen.h"
#include <opencv2/opencv.hpp>
#include <vector>

// Keypoint detector backend, independent of the estimator below.
enum class SiftBackend { OpenCV, Custom };

// Fundamental-matrix estimator backend, independent of the SIFT backend above.
enum class FMatrixBackend { OpenCVFundamental, CustomRansac };

struct SparseMatchParams {
    // SIFT detection (OpenCV defaults)
    int    sift_features        = 0;     // 0 = no limit
    int    sift_octave_layers   = 5;
    double sift_contrast_thresh = 0.04;
    double sift_edge_thresh     = 10.0;
    double sift_sigma           = 1.6;
    // Descriptor matching
    float  ratio_threshold      = 0.75f;  // Lowe ratio test
    // RANSAC fundamental matrix
    double ransac_threshold     = 1.0;    // max Sampson distance (px)
    // recoverPose chirality check
    double chirality_depth      = 25.0;   // maxDepth passed to cv::recoverPose
    size_t custom_ransac_iters = 1000; // number of iterations
    // Only used with SiftBackend::Custom: swaps the SIFT detector itself (feature detection +
    // orientation + descriptor) for our own implementation (KeypointDetector.h) instead of cv::SIFT.
    bool   use_custom_sift      = false;
};

struct SparseMatchResult {
    std::vector<cv::Point2f> pts_left;
    std::vector<cv::Point2f> pts_right;
    std::vector<uchar>       inlier_mask;   // 1 = inlier after RANSAC

    cv::Mat F;   // 3×3 fundamental matrix (double)
    cv::Mat E;   // 3×3 essential matrix, derived as K1^T F K0 (double)
    cv::Mat R;   // 3×3 relative rotation  (double)
    cv::Mat t;   // 3×1 relative translation unit vector (double)

    int n_matches = 0;  // after Lowe ratio test
    int n_inliers = 0;  // after RANSAC (used for Sampson error)
    int n_pose_inliers = 0;  // after recoverPose chirality check
    double mean_epipolar_error = 0.0;

    
    
    
    double sift_time_ms    = 0.0;  // detectSiftMatches
    double fmatrix_time_ms = 0.0;  // computeSparseMatchesOpenCV/Custom
};

// Full sparse pipeline: SIFT → match → F → E → R,t. siftBackend and fmatrixBackend are
// independent axes. Canonicalizes calib (K0/K1/R_rel/t_rel/F) so t_rel.x < 0; does
// NOT swap left/right or result.R/t -- caller must swap those on calib.swapped (pipeline.cpp).
SparseMatchResult computeSparseMatches(cv::Mat& left,
                                        cv::Mat& right,
                                        CalibData& calib,
                                        SiftBackend siftBackend,
                                        FMatrixBackend fmatrixBackend,
                                        const SparseMatchParams& params = SparseMatchParams{});

// Backward-compatible overload for existing callers: opencv_sm=true -> OpenCV SIFT + OpenCV
// fundamental matrix; opencv_sm=false -> SIFT per params.use_custom_sift + custom RANSAC.
SparseMatchResult computeSparseMatches(cv::Mat& left,
                                        cv::Mat& right,
                                        CalibData& calib,
                                        bool opencv_sm,
                                        const SparseMatchParams& params = SparseMatchParams{});

// Interface to keypoint detection
    // Runs the requested SIFT backend (OpenCV or custom),
    // Matches descriptors, and applies Lowe's ratio test
SparseMatchResult detectSiftMatches(const cv::Mat& imgLeft, const cv::Mat& imgRight,
                                     const SparseMatchParams& params, SiftBackend siftBackend);

SparseMatchResult computeSparseMatchesOpenCV(SparseMatchResult result, const CalibData& calib,
                                              const SparseMatchParams& params);

SparseMatchResult computeSparseMatchesCustom(SparseMatchResult result, const CalibData& calib,
                                              const SparseMatchParams& params);

// Sampson distance (first-order geometric error)
//   d = sqrt( (x2^T F x1)^2 / ( (Fx1)[0]^2 + (Fx1)[1]^2 + (F^T x2)[0]^2 + (F^T x2)[1]^2 ) )
//   sqrt is applied so the result is in pixels (same units as the RANSAC threshold)
double sampsonDistance(const cv::Point2f& pt_left, const cv::Point2f& pt_right, const cv::Mat& F);



// Saves the RANSAC inlier point pairs to a YAML file for use by rectifyLoopZhang.
void saveSparseMatchInliers(const SparseMatchResult& result, const std::string& path);

// Renders matched point pairs and saves them as a PNG under saveDir. No CSV logging.
void saveMatchVisualization(const cv::Mat& imgLeft, const cv::Mat& imgRight,
                             const SparseMatchResult& result,
                             const std::string& saveDir,
                             const std::string& runName,
                             const std::string& viewL,
                             const std::string& viewR,
                             bool inliersOnly = true);



std::vector<Eigen::Vector3d> normalizePoints(const std::vector<Eigen::Vector3d>& pts, const Eigen::Matrix3d & K);

// pts are already normalized
size_t recoverPose (const std::vector<Eigen::Vector3d>& pts_left, const std::vector<Eigen::Vector3d>& pts_right, 
                                                            const Eigen::Matrix3d& E,
                                                            Eigen::Matrix3d & R, Eigen::Vector3d & t,
                                                            double chirality_depth);
Eigen::Vector3d triangulatePoint(const Eigen::Vector3d & pt_left_norm, const Eigen::Vector3d & pt_right_norm, const Eigen::Matrix<double, 3, 4> & P_left, const Eigen::Matrix<double, 3, 4> & P_right);

