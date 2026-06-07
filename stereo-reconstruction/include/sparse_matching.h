#pragma once
#include <opencv2/opencv.hpp>
#include "utils.h"

// Result of sparse feature matching + pose recovery (Member 1)
struct SparseMatchResult {
    std::vector<cv::Point2f> pts_left;
    std::vector<cv::Point2f> pts_right;
    std::vector<uchar>       inlier_mask;
    cv::Mat F;   // 3x3 fundamental matrix (rank-2 enforced)
    cv::Mat E;   // 3x3 essential matrix
    cv::Mat R;   // 3x3 rotation  (camera 0 -> camera 1)
    cv::Mat t;   // 3x1 unit translation vector
};

// Detect features, match with ratio test, estimate F via 8-point + RANSAC,
// derive E from F and camera matrices, recover R and t.
SparseMatchResult computeSparseMatches(const cv::Mat& left, const cv::Mat& right,
                                        const CalibData& calib);
