#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

/**
 * @brief Struct to hold calibration data for stereo pairs. 
 */
struct CalibData {
    cv::Mat K0, K1;         // Intrinsic matrices
    cv::Mat R0, t0;         // Left camera absolute pose for transformation from camera space to world space (required for ICP multi-pair fusion)
    
    cv::Mat R_rel;          // Relative pose: (R1 * R0^T)
    cv::Mat t_rel;          // Metric translation

    cv::Mat F;              // Fundamental matrix

    double baseline = 0.0;
    bool swapped = false;   // Flag indicating if the left/right cameras were swapped to ensure left camera is geometrically to the left of the right camera

    // Utility functions to check if the calibration data is complete
    bool hasIntrinsics()       const { return !K0.empty() && !K1.empty() && baseline > 0.0; }
    bool hasRelativePose()     const { return !R_rel.empty() && !t_rel.empty(); }
    bool hasFundamentalMatrix() const { return !F.empty(); }

    /**
     * @brief Verifies that the left and right cameras are in the correct order based on the relative translation vector.
     *        If the fundamental matrix is present, it will also be transposed to maintain consistency with the swapped cameras.
     */
    void verifyLeftRightCameraOrder();
};

/**
 * @brief Saves the given calibration data to a YAML file.
 * 
 * @param calib Calibration data to be saved.
 * @param path Path to the output YAML file.
 */
void saveCalibData(const CalibData& calib, const std::string& path);

/**
 * @brief Loads calibration data from a YAML file.
 * 
 * @param path Path to the input YAML file.
 * @return CalibData 
 */
CalibData loadCalibData(const std::string& path);

/**
 * @brief Saves the disparity map to a PNG file. 
 *        The disparity values are scaled by 16 and stored as 16-bit unsigned integers.
 * 
 * @param disp Disparity map to be saved.
 * @param path Path to the output PNG file.
 */
void saveDisparity(const cv::Mat& disp, const std::string& path);

/**
 * @brief Loads a disparity map from a PNG file.
 *        The disparity values are expected to be stored as 16-bit unsigned integers and will be scaled by 16.
 * 
 * @param path Path to the input PNG file.
 * @return cv::Mat 
 */
cv::Mat loadDisparity(const std::string& path);

/**
 * @brief Pads a view ID with leading zeros.
 * 
 * @param id The view ID to be padded.
 * @return std::string The padded view ID.
 */
std::string padViewId(int id);

/**
 * @brief Prints information about a cv::Mat object.
 * 
 * @param name Name of the matrix (for display purposes).
 * @param m The cv::Mat object whose information is to be printed.
 */
void printMatInfo(const std::string& name, const cv::Mat& m);
