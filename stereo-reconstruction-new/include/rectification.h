#pragma once

#include "utils.h"
#include <opencv2/opencv.hpp>

/**
 * @brief Output of stereo rectification.
 *
 * The rectified images use the same output size and have horizontal epipolar
 * lines. A valid match is therefore searched on the same row in both images.
 */
struct RectifyResult {
    cv::Mat left_rect;   // Rectified left image.
    cv::Mat right_rect;  // Rectified right image.

    cv::Mat H1;          // Homography applied to the left image.
    cv::Mat H2;          // Homography applied to the right image.

    cv::Mat mask1;       // 255 for valid left-image pixels, 0 for warp padding.
    cv::Mat mask2;       // 255 for valid right-image pixels, 0 for warp padding.

    cv::Mat P1;          // Projection matrix for the rectified left image.
    cv::Mat P2;          // Projection matrix for the rectified right image.
    cv::Mat Q;           // OpenCV disparity-to-depth matrix; empty for Loop-Zhang.
    cv::Mat R0_rect;     // Rotation of the left camera used by calibrated rectification.
};

/**
 * @brief Rectify a calibrated stereo pair with OpenCV.
 *
 * OpenCV uses the camera intrinsics and relative pose to compute rectifying
 * rotations. The returned Q matrix can be used for calibrated depth recovery.
 * Validity masks are generated with the same remapping operation as the images.
 *
 * @param[in] left Left input image.
 * @param[in] right Right input image.
 * @param[in] calib Calibration containing K0, K1, R_rel, and t_rel.
 * @return Rectified images, masks, projection matrices, and Q.
 */
RectifyResult rectifyOpenCV(const cv::Mat& left, const cv::Mat& right,
                            const CalibData& calib);

/**
 * @brief Rectify a stereo pair with the Loop-Zhang projective method.
 *        Choose the best projective direction, build projective transforms, 
 *        align the epipolar rows, reduce shear distortion, then place both images on one shared canvas.
 *
 * @param[in] left Left input image.
 * @param[in] right Right input image.
 * @param[in] calib Calibration containing F, intrinsics, and relative pose.
 * @return Rectified images, masks, homographies, and projection matrices.
 */
RectifyResult rectifyLoopZhang(const cv::Mat& left, const cv::Mat& right,
                               const CalibData& calib);