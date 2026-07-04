#pragma once
#include "utils.h"
#include <opencv2/opencv.hpp>

struct RectifyResult {
    cv::Mat left_rect;   // warped left image
    cv::Mat right_rect;  // warped right image
    cv::Mat H1;          // homography applied to left
    cv::Mat H2;          // homography applied to right
    cv::Mat Q;           // disparity-to-depth 4×4 matrix
    cv::Mat R0_rect;     // rotation applied to left camera
    cv::Mat P1, P2;      // rectified projection matrices
};

// Loop-Zhang Rectification
// Uses pure uncalibrated geometry. F is estimated from sparse matching.
// cv_pts_l and cv_pts_r must be the filtered, verified INLIER matched points.
RectifyResult rectifyLoopZhang(const cv::Mat& left, const cv::Mat& right,
    const cv::Mat& F_cv,
    const std::vector<cv::Point2f>& cv_pts_l,
    const std::vector<cv::Point2f>& cv_pts_r);

// OpenCV baseline (for comparison / fallback)
RectifyResult rectifyOpenCV(const cv::Mat& left, const cv::Mat& right,
                              const CalibData& calib);
