#pragma once
#include "Eigen.h"
#include <vector>
#include <cstdint>

/** 
 * @brief Structure to hold the result of the RANSAC fundamental matrix estimation.
 */
struct RansacResult {
    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
    std::vector<uint8_t> inlier_mask;   // Mask representing inliers (1) and outliers (0) after RANSAC
    int n_inliers = 0;
    double mean_epipolar_error = 0.0;   // Mean Sampson epipolar error over RANSAC inliers
};

class Ransac {
    size_t m_num_iters;     // Number of iterations to run
    double m_threshold;     // Sampson distance inlier threshold

    static constexpr int MIN_SAMPLE_SIZE = 8;     // Minimum points required for the 8-point algorithm
    static constexpr unsigned int RNG_SEED = 42;  // For reproducibility

    /**
     * @brief Rescales points for a better-conditioned 8-point solve (Hartley normalization).
     *        Rows of A are spanning orders of magnitude (quadratic terms in raw pixel coords vs. the constant 1).
     *        Normalization rescales the points so that the average distance from the origin is sqrt(2) and the centroid is at the origin.
     *        This improves numerical stability and accuracy of the fundamental matrix estimation.
     *        Needs to be undone on the resulting F matrix after estimation.
     *        
     * @param[in] pts Points to be normalized.
     * @param[out] normalized_pts Vector to store the normalized points.
     * @return Eigen::Matrix3d Transformation matrix for normalization.
     */
    static Eigen::Matrix3d doHartleyNormalization(const std::vector<Eigen::Vector3d>& pts, std::vector<Eigen::Vector3d>& normalized_pts);


    /** 
     * @brief Estimates the fundamental matrix using the 8-point algorithm.
     * 
     * @param[in] pts_left Points in the left image.
     * @param[in] pts_right Points in the right image.
     * @return Eigen::Matrix3d Estimated fundamental matrix.
     */
    static Eigen::Matrix3d estimateFundamentalMatrix(const std::vector<Eigen::Vector3d>& pts_left, const std::vector<Eigen::Vector3d>& pts_right);

    /**
     * @brief Computes the Sampson distance (px) between corresponding points and the fundamental matrix.
     * @param[in] pt_left Point in the left image.
     * @param[in] pt_right Point in the right image.
     * @param[in] F Fundamental matrix.
     * @return double Sampson distance (px).
     */
    static double sampsonDistance(const Eigen::Vector3d& pt_left, const Eigen::Vector3d& pt_right, const Eigen::Matrix3d& F);


public:
    /** 
     * @brief Constructor for the Ransac class.
     * 
     * @param num_iters Number of iterations to run.
     * @param threshold Sampson distance inlier threshold.
     */
    Ransac(size_t num_iters, double threshold)
        : m_num_iters(num_iters)
        , m_threshold(threshold)
        {}


    /** 
     * @brief Finds the fundamental matrix using RANSAC.
     * 
     * @param pts_left Points in the left image.
     * @param pts_right Points in the right image.
     * @return RansacResult Result containing the estimated fundamental matrix and inlier information.
     */
    RansacResult findFundamentalMatrix(const std::vector<Eigen::Vector3d>& pts_left, const std::vector<Eigen::Vector3d>& pts_right);

};