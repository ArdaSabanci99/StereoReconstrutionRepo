#pragma once
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <vector>

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
