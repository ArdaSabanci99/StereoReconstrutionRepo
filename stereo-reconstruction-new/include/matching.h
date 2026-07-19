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
    int         min_disparity  = 80;   // DTU scan1: object ~550mm, f=720,B=61mm → d_min≈80
    bool        subpixel       = true;   // parabola sub-pixel refinement
    bool        lr_check       = true;   // left-right consistency check
    bool        median_filter  = false;  // post-processing median filter (off: cleaner cloud, higher precision)
    // SGM penalties
    int P1 = 10;
    int P2 = 120;
};

// Compute disparity map from rectified grayscale pair.
// Returns CV_32F disparity (invalid pixels = -1).
cv::Mat computeDisparity(const cv::Mat& left_rect, const cv::Mat& right_rect,
                          const MatchParams& params, const cv::Mat& mask1 = cv::Mat(), const cv::Mat& mask2 = cv::Mat());
