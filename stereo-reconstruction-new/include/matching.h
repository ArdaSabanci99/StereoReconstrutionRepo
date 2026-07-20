#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Enumeration of cost functions for matching.
 */
enum class MatchMethod {
    MANUAL_SAD,
    MANUAL_SSD,
    MANUAL_NCC,
    MANUAL_CENSUS,
    MANUAL_SGM,
    OPENCV_BM,
    OPENCV_SGBM,
};

/**
 * @brief Parameters for stereo matching.
 */
struct MatchParams {
    MatchMethod method = MatchMethod::MANUAL_SGM;
    int window_size    = 5;
    int num_disparities= 256;
    int min_disparity  = 80;
    bool subpixel = true;           // Parabola sub-pixel refinement
    bool lr_check = true;           // Left-right consistency check
    bool median_filter = false;     // Post-processing median filter (off: cleaner cloud, higher precision)

    // SGM penalties
    int P1 = 10;
    int P2 = 120;
};


/**
 * @brief Scales the configured disparity bounds with image resolution to preserve the depth range.
 */
void configureDisparityRangeForScale(MatchParams& params, double scale);

/** @brief Validate and normalize matcher parameters before computation. */
void normalizeMatchParams(MatchParams& params);

/**
 * @brief Computes the disparity map from rectified grayscale stereo images.
 *
 * @param left_rect Left rectified image.
 * @param right_rect Right rectified image
 * @param params Matching parameters
 * @param mask1 Left validity mask (255 = valid content, 0 = black warp padding)
 * @param mask2 Right validity mask (255 = valid content, 0 = black warp padding)
 * @return cv::Mat Disparity map in CV_32F format, with invalid pixels set to -1.
 */
cv::Mat computeDisparity(const cv::Mat& left_rect, const cv::Mat& right_rect,
                          const MatchParams& params, const cv::Mat& mask1 = cv::Mat(), const cv::Mat& mask2 = cv::Mat());
