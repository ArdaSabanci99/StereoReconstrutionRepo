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

// Manual Loop-Zhang rectification using F + K from calib.
// F is estimated from sparse matching; K from calib.K0 / calib.K1.
RectifyResult rectifyManual(const cv::Mat& left, const cv::Mat& right,
                              const cv::Mat& F,
                              const CalibData& calib);

// OpenCV baseline (for comparison / fallback)
RectifyResult rectifyOpenCV(const cv::Mat& left, const cv::Mat& right,
                              const CalibData& calib);
