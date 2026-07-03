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

// Manual Loop-Zhang rectification using F and K from calib.
// in_l / in_r are the RANSAC inlier correspondences from sparse matching,
// used to fit the affine component of H1 that minimises vertical disparity.
RectifyResult rectifyManual(const cv::Mat& left, const cv::Mat& right,
                              const CalibData& calib,
                              const std::vector<cv::Point2f>& in_l,
                              const std::vector<cv::Point2f>& in_r);

// OpenCV baseline (for comparison / fallback)
RectifyResult rectifyOpenCV(const cv::Mat& left, const cv::Mat& right,
                              const CalibData& calib);
