#pragma once
#include <opencv2/opencv.hpp>

enum class MatchMethod {
    OPENCV_BM,    // OpenCV BlockMatching (baseline)
    OPENCV_SGBM,  // OpenCV Semi-Global BM (baseline)
    MANUAL_SSD,   // Sum of Squared Differences (manual)
    MANUAL_SAD,   // Sum of Absolute Differences (manual)
    MANUAL_NCC,   // Normalized Cross-Correlation (manual)
    MANUAL_CENSUS, // Census transform + Hamming cost — Member 3
    MANUAL_SGM,    // Semi-Global Matching (uses Census as cost) — Member 3, C1
};

struct MatchParams {
    MatchMethod method  = MatchMethod::OPENCV_BM;
    int window_size     = 9;     // must be odd
    int min_disparity   = 0;
    int num_disparities = 64;    // must be divisible by 16
};

// Compute disparity map from rectified stereo pair
// Input:  8-bit or 16-bit grayscale rectified images
// Output: 32-bit float disparity map (invalid pixels = -1)
cv::Mat computeDisparity(const cv::Mat& left_rect, const cv::Mat& right_rect,
                          const MatchParams& params);
