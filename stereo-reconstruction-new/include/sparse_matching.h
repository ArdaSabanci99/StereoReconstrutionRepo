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

    int n_matches = 0;
    int n_inliers = 0;
    double mean_epipolar_error = 0.0;
};

// Full sparse pipeline: SIFT → match → 8-point+RANSAC → E → R,t
SparseMatchResult computeSparseMatches(const cv::Mat& left,
                                        const cv::Mat& right,
                                        const CalibData& calib);

// Low-level: normalised 8-point algorithm (pure manual, no OpenCV findFundamentalMat)
// pts1, pts2 must be the same size; returns 3×3 F (double, rank-2 enforced)
cv::Mat estimateFundamentalManual(const std::vector<cv::Point2f>& pts1,
                                   const std::vector<cv::Point2f>& pts2);
