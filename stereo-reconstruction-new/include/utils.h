#pragma once
#include "Eigen.h"
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
 * @param[in] path Path to the input YAML file.
 * @return CalibData 
 */
CalibData loadCalibData(const std::string& path);

/**
 * @brief Saves the disparity map to a PNG file. 
 *        The disparity values are scaled by 16 and stored as 16-bit unsigned integers.
 * 
 * @param[in] disp Disparity map to be saved.
 * @param[in] path Path to the output PNG file.
 */
void saveDisparity(const cv::Mat& disp, const std::string& path);

/**
 * @brief Loads a disparity map from a PNG file.
 *        The disparity values are expected to be stored as 16-bit unsigned integers and will be scaled by 16.
 * 
 * @param[in] path Path to the input PNG file.
 * @return cv::Mat 
 */
cv::Mat loadDisparity(const std::string& path);

/**
 * @brief Pads a view ID with leading zeros.
 * 
 * @param[in] id The view ID to be padded.
 * @return std::string The padded view ID.
 */
std::string padViewId(int id);

/**
 * @brief Prints information about a cv::Mat object.
 *
 * @param[in] name Name of the matrix (for display purposes).
 * @param[in] m The cv::Mat object whose information is to be printed.
 */
void printMatInfo(const std::string& name, const cv::Mat& m);

/**
 * @brief Triangulates a 3D point from two views via DLT.
 *
 * @param[in] pt_left  Homogeneous point in the left view.
 * @param[in] pt_right Homogeneous point in the right view.
 * @param[in] P_left   Projection matrix of the left camera.
 * @param[in] P_right  Projection matrix of the right camera.
 * @return Eigen::Vector3d Triangulated 3D point.
 */
Eigen::Vector3d triangulatePoint(const Eigen::Vector3d& pt_left, const Eigen::Vector3d& pt_right,
                                  const Eigen::Matrix<double, 3, 4>& P_left, const Eigen::Matrix<double, 3, 4>& P_right);

/**
 * @brief Converts an Eigen::Matrix3d to a CV_64F cv::Mat.
 */
cv::Mat eigenToCv(const Eigen::Matrix3d& src);

/**
 * @brief Converts a 3x4 Eigen matrix (projection matrix) to a CV_64F cv::Mat.
 */
cv::Mat eigenToCv34(const Eigen::Matrix<double, 3, 4>& src);

/**
 * @brief Converts a 3x3 CV_64F cv::Mat to Eigen::Matrix3d.
 */
Eigen::Matrix3d cvToEigen3x3(const cv::Mat& src);

/**
 * @brief Converts a 3x1 CV_64F cv::Mat to Eigen::Vector3d.
 */
Eigen::Vector3d cvToEigenVec3(const cv::Mat& src);
