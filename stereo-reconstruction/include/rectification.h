#pragma once
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "sparse_matching.h"

struct RectifyResult {
    cv::Mat left_rect;
    cv::Mat right_rect;
    cv::Mat Q;          // disparity-to-depth mapping matrix (4x4)
    cv::Mat P1, P2;     // new projection matrices
};

// OpenCV-based rectification (baseline implementation)
RectifyResult rectifyOpenCV(const cv::Mat& left, const cv::Mat& right,
                             const CalibData& calib);

// Manual rectification via homography (to be implemented)
RectifyResult rectifyManual(const cv::Mat& left, const cv::Mat& right,
                             const CalibData& calib);
