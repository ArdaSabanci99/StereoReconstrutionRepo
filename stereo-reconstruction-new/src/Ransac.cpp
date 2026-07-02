#include "Ransac.h"
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <algorithm>

double Ransac::sampsonDistance(const Eigen::Vector3d & pt_left, const Eigen::Vector3d & pt_right, const Eigen::Matrix3d & F) {
    double num  = pt_right.dot(F * pt_left);
    Eigen::Vector3d l_right = F * pt_left;  // epipolar line in right image
    Eigen::Vector3d l_left = F.transpose() * pt_right;  // epipolar line in left image

    double denom = l_right[0]*l_right[0] + l_right[1]*l_right[1]
                 + l_left[0] *l_left[0] + l_left[1] *l_left[1];
    
    return std::sqrt((num * num) / (denom + 1e-10));
}



Eigen::Matrix3d Ransac::estimateFundamentalMatrix(const std::vector<Eigen::Vector3d> & pts_left, const std::vector<Eigen::Vector3d> & pts_right) {
    size_t constraint_num = pts_left.size();  // each correspondence produces one constraint on F (9 unknowns))
    Eigen::MatrixXd A(constraint_num, 9);  // epipolar constrainst matrix
    // TODO: Hartley normalization (make the SVD more stable)
    // Create the linear system of equations for the 8-point algorithm
        // x'^T * F * x = 0
        // Af = 0; 
            // A = rows of Kroneecker products of point correspondences
                // a = (x'x, x'y, x', y'x, y'y, y', x, y, 1)
            // f = vectorized form of F (9x1): (f11, f12, f13, f21, f22, f23, f31, f32, f33)
        
    for (size_t i = 0; i < pts_left.size(); ++i) {
        Eigen::Vector3d pt_left = pts_left[i];
        Eigen::Vector3d pt_right = pts_right[i];

        Eigen::Matrix<double, 9, 1> a;
        a << pt_right.x() * pt_left.x(),
            pt_right.x() * pt_left.y(),
            pt_right.x(),
            pt_right.y() * pt_left.x(),
            pt_right.y() * pt_left.y(),
            pt_right.y(),
            pt_left.x(),
            pt_left.y(),
            1.0;

        A.row(i) = a.transpose();
    }
    // Solve: non-zero f minimizing ||Af|| subject to ||f|| = 1
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    // vector that best satisfies Af = 0
        // = singular vector corresponding to the smallest singular value
    Eigen::VectorXd f = svd.matrixV().col(8); 
    Eigen::Matrix3d F_estimated;
    F_estimated << f(0), f(1), f(2),
                   f(3), f(4), f(5),
                   f(6), f(7), f(8);

    // Enforcing rank 2
        // F_estimated is mathematical estimate -> need to enforce physical constraints
        // closest solution is obtained by setting the smallest singular value to zero
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_F(F_estimated, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd_F.singularValues();
    singular_values(2) = 0;  // set smallest singular value to zero

    Eigen::Matrix3d F_rank2 = svd_F.matrixU() * singular_values.asDiagonal() * svd_F.matrixV().transpose();
    
    return F_rank2;
};

RansacResult Ransac::findFundamentalMatrix(const std::vector<Eigen::Vector2d>& pts_left, const std::vector<Eigen::Vector2d>& pts_right) {
    std::vector<Eigen::Vector3d> pts_left_homogeneous, pts_right_homogeneous;
    for (const auto & pt : pts_left) {
        pts_left_homogeneous.push_back(pt.homogeneous());
    }
    for (const auto & pt : pts_right) {
        pts_right_homogeneous.push_back(pt.homogeneous());
    }

    return findFundamentalMatrix(pts_left_homogeneous, pts_right_homogeneous);
};

RansacResult Ransac::findFundamentalMatrix(const std::vector<Eigen::Vector3d>& pts_left, const std::vector<Eigen::Vector3d>& pts_right) {
    if (pts_left.size() < m_min_sample_size) {
        throw std::runtime_error(std::string("[sparse matching] Not enough points (<" + std::to_string(m_min_sample_size) + ")"));
    }

    if (pts_left.size() != pts_right.size()) {
        throw std::runtime_error(std::string("[sparse matching] Number of left and right points must be equal."));
    }

    RansacResult result;

    result.inlier_mask = std::vector<uint8_t>(pts_left.size(), 0);

    std::mt19937 rng(RNG_SEED);  // random number generator
    std::vector<size_t> indices(pts_left.size());
    std::iota(indices.begin(), indices.end(), 0); // initialize indices to 0, 1, ..., N-1

    // TODO: Add adaptive RANSAC based on inlier ratio and confidence
    // TODO: consider LO-RANSAC (local optimization: refit F from inliers within the loop, not just once at the end)
    for (size_t i = 0; i < m_num_iters; ++i) {
        // Randomly sample 8 points
            // TODO: No degeneracy check (near-collinear, near-duplicate points, near-planar scene, all points from small clustered region)
        std::shuffle(indices.begin(), indices.end(), rng);  // Randomly Shuffle indices
        std::vector<Eigen::Vector3d> sample_left, sample_right;
        for (size_t j = 0; j < m_min_sample_size; ++j) {
            sample_left.push_back(pts_left[indices[j]]);
            sample_right.push_back(pts_right[indices[j]]);
        }
        
        // Estimate F_rank2 from the 8 samples
        Eigen::Matrix3d F = estimateFundamentalMatrix(sample_left, sample_right);

        // Compute inliers based on the Sampson distance
        double inlier_count = 0;
        std::vector<uint8_t> inlier_mask(pts_left.size(), 0);
        double distance = 0.0;
        for (size_t k = 0; k < pts_left.size(); ++k) {
            distance = sampsonDistance(pts_left[k], pts_right[k], F);
            if (distance < m_threshold) {
                inlier_mask[k] = 1;
                ++inlier_count;
            }
        }
        
        if (inlier_count > result.n_inliers) {
            result.n_inliers = inlier_count;
            result.inlier_mask = inlier_mask;
        }
    }

    // RANSAC found inliers
    // user inliers to refit the F matrix
    std::vector<Eigen::Vector3d> inliers_left, inliers_right;
    for (size_t i = 0; i < pts_left.size(); ++i) {
        if (result.inlier_mask[i]) {
            inliers_left.push_back(pts_left[i]);
            inliers_right.push_back(pts_right[i]);
        }
    }
    result.F = estimateFundamentalMatrix(inliers_left, inliers_right);

    // Compute mean Sampson epipolar error over RANSAC inliers
    double total_err = 0.0;
    for (size_t i = 0; i < pts_left.size(); ++i)
        if (result.inlier_mask[i]) {
            total_err += sampsonDistance(pts_left[i], pts_right[i], result.F);
        }
    
    result.mean_epipolar_error = result.n_inliers > 0 ? total_err / result.n_inliers : 0.0;

    return result;
};