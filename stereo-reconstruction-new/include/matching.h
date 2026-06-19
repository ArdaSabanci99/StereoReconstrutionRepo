#pragma once
#include <opencv2/opencv.hpp>

enum class MatchMethod {
    MANUAL_SAD,
    MANUAL_SSD,
    MANUAL_NCC,
    MANUAL_CENSUS,
    MANUAL_SGM,      // C1 challenge
    OPENCV_BM,
    OPENCV_SGBM,
};

struct MatchParams {
    MatchMethod method         = MatchMethod::MANUAL_SGM;
    int         window_size    = 5;
    int         num_disparities= 64;
    int         min_disparity  = 0;
    bool        subpixel       = true;   // parabola sub-pixel refinement
    bool        lr_check       = true;   // left-right consistency check
    bool        median_filter  = true;   // post-processing median filter
    // SGM penalties
    int P1 = 10;
    int P2 = 120;
};

// Compute disparity map from rectified grayscale pair.
// Returns CV_32F disparity (invalid pixels = -1).
cv::Mat computeDisparity(const cv::Mat& left_rect, const cv::Mat& right_rect,
                          const MatchParams& params);
