#pragma once
#include "utils.h"
#include "Eigen.h"
#include <opencv2/opencv.hpp>
#include <vector>

struct SparseMatchParams {
    // SIFT detection (OpenCV defaults)
    int    sift_features        = 0;     // 0 = no limit
    int    sift_octave_layers   = 3;
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
};

struct SparseMatchResult {
    std::vector<cv::Point2f> pts_left;
    std::vector<cv::Point2f> pts_right;
    std::vector<uchar>       inlier_mask;   // 1 = inlier after RANSAC

    cv::Mat F;   // 3×3 fundamental matrix (double)
    cv::Mat E;   // 3×3 essential matrix   (double)
    cv::Mat R;   // 3×3 relative rotation  (double)
    cv::Mat t;   // 3×1 relative translation unit vector (double)

    int n_matches = 0;  // after Lowe ratio test
    int n_inliers = 0;  // after RANSAC (used for Sampson error)
    int n_pose_inliers = 0;  // after recoverPose chirality check
    double mean_epipolar_error = 0.0;
};

// Full sparse pipeline: SIFT → match → 8-point+RANSAC → E → R,t
// Interface, based on opencv_sm runs either computeSparseMatchesOpenCV or computeSparseMatchesCustom
// May swap left/right and calib.K0/K1 to ensure right camera is rightward (t[0] > 0).
SparseMatchResult computeSparseMatches(cv::Mat& left,
                                        cv::Mat& right,
                                        CalibData& calib,
                                        bool opencv_sm,
                                        const SparseMatchParams& params = SparseMatchParams{});

// OpenCV Baseline
SparseMatchResult computeSparseMatchesOpenCV(const cv::Mat& left,
                                        const cv::Mat& right,
                                        const CalibData& calib,
                                        const SparseMatchParams& params);

// Our custom implementation
SparseMatchResult computeSparseMatchesCustom(const cv::Mat& left,
                                        const cv::Mat& right,
                                        const CalibData& calib,
                                        const SparseMatchParams& params);

// Low-level: normalised 8-point algorithm (pure manual, no OpenCV findFundamentalMat)
// pts1, pts2 must be the same size; returns 3×3 F (double, rank-2 enforced)
cv::Mat estimateFundamentalManual(const std::vector<cv::Point2f>& pts1,
                                   const std::vector<cv::Point2f>& pts2);

// Approximate distance (px) from each point to its epipolar line in the other image.
double sampsonDistance(const cv::Point2f& pt_left, const cv::Point2f& pt_right, const cv::Mat& F);



void saveMatchVisualization(const cv::Mat& imgLeft, const cv::Mat& imgRight,
                             const SparseMatchResult& result,
                             const std::string& saveDir,
                             const std::string& runName,
                             const std::string& viewL,
                             const std::string& viewR,
                             double rotErrDeg = -1.0,
                             double transErrDeg = -1.0,
                             bool inliersOnly = true);



