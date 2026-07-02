#pragma once
#include "Eigen.h"
#include <vector>
#include <cstdint>


struct RansacResult {
    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();  // 3×3 fundamental matrix (double, rank-2 enforced)
    std::vector<uint8_t> inlier_mask;   // 1 = inlier after RANSAC
    int n_inliers = 0;
    double mean_epipolar_error = 0.0;
};

class Ransac {
    size_t m_num_iters;  // number of iterations to run
    double m_threshold;  // Sampson distance inlier threshold
    int m_min_sample_size = 8;
    
    static constexpr unsigned int RNG_SEED = 42;  // For reproducibility

    // TODO: add confidence attribute

    Eigen::Matrix3d estimateFundamentalMatrix(const std::vector<Eigen::Vector3d>& pts_left, const std::vector<Eigen::Vector3d>& pts_right);

    // ─────────────────────────────────────────────────────────────────────────────
    // Sampson distance (first-order geometric error)
    //   d = sqrt( (x2^T F x1)^2 / ( (Fx1)[0]^2 + (Fx1)[1]^2 + (F^T x2)[0]^2 + (F^T x2)[1]^2 ) )
    //   sqrt is applied so the result is in pixels (same units as the RANSAC threshold)
    // ─────────────────────────────────────────────────────────────────────────────
    double sampsonDistance(const Eigen::Vector3d& pt_left, const Eigen::Vector3d& pt_right, const Eigen::Matrix3d& F);


public:
    Ransac(size_t num_iters, double threshold, int min_sample_size = 8)
        : m_num_iters(num_iters)
        , m_threshold(threshold)
        , m_min_sample_size(min_sample_size)
        {}


    RansacResult findFundamentalMatrix(const std::vector<Eigen::Vector2d>& pts_left, const std::vector<Eigen::Vector2d>& pts_right);
    RansacResult findFundamentalMatrix(const std::vector<Eigen::Vector3d>& pts_left, const std::vector<Eigen::Vector3d>& pts_right);

};