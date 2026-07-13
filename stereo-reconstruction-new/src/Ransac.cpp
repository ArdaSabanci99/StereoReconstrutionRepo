#include "Ransac.h"
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>

constexpr double kMinPerpDistPx = 3.0;  // degeneracy check: min pixel distance from a candidate line

// Rejects samples where all points lie on (or near) one line in an image
    // To  uniquely determine the fundamental matrix
static bool isNearCollinear(const std::vector<Eigen::Vector3d> & pts, double min_perp_dist_px) {
    const size_t n = pts.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            Eigen::Vector2d line_dir = pts[j].head<2>() - pts[i].head<2>();
            double line_len = line_dir.norm();
            if (line_len < 1e-9) continue;  // Duplicate points

            size_t num_on_line = 0;
            for (size_t k = 0; k < n; ++k) {
                if (k == i || k == j) continue;
                Eigen::Vector2d to_k = pts[k].head<2>() - pts[i].head<2>();
                double cross = line_dir.x() * to_k.y() - line_dir.y() * to_k.x();
                double perp_dist = std::abs(cross) / line_len;
                if (perp_dist < min_perp_dist_px) ++num_on_line;
            }

            if (num_on_line == n - 2)  // Every other point also lies on this line
                return true;
        }
    }
    return false;
}

double Ransac::sampsonDistance(const Eigen::Vector3d & pt_left, const Eigen::Vector3d & pt_right, const Eigen::Matrix3d & F) {
    double num  = pt_right.dot(F * pt_left);
    Eigen::Vector3d l_right = F * pt_left;  // epipolar line in right image
    Eigen::Vector3d l_left = F.transpose() * pt_right;  // epipolar line in left image

    double denom = l_right[0]*l_right[0] + l_right[1]*l_right[1]
                 + l_left[0] *l_left[0] + l_left[1] *l_left[1];
    
    return std::sqrt((num * num) / (denom + 1e-10));
}


Eigen::Matrix3d Ransac::doHartleyNormalization(const std::vector<Eigen::Vector3d> & pts, std::vector<Eigen::Vector3d> & normalized_pts) {
    // Move centroid to the origin
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (const auto & pt : pts) {
        centroid += pt.head<2>();
    }
    centroid /= static_cast<double>(pts.size());

    // Scale so the mean distance to the origin becomes sqrt(2)
    double mean_dist = 0.0;
    for (const auto & pt : pts) {
        mean_dist += (pt.head<2>() - centroid).norm();
    }
    mean_dist /= static_cast<double>(pts.size());
    double scale = std::sqrt(2.0) / mean_dist;

    Eigen::Matrix3d transform;
    transform << scale, 0,     -scale * centroid.x(),
                 0,     scale, -scale * centroid.y(),
                 0,     0,      1;

    normalized_pts.clear();
    normalized_pts.reserve(pts.size());
    for (const auto & pt : pts) {
        normalized_pts.push_back(transform * pt);
    }

    return transform;
}

Eigen::Matrix3d Ransac::estimateFundamentalMatrix(const std::vector<Eigen::Vector3d> & pts_left, const std::vector<Eigen::Vector3d> & pts_right) {
    size_t constraint_num = pts_left.size();  // each correspondence produces one constraint on F (9 unknowns))
    Eigen::MatrixXd A(constraint_num, 9);  // epipolar constrainst matrix

    // Hartley normalization: rescale points before the SVD, undo it on F at the end
    std::vector<Eigen::Vector3d> pts_left_norm, pts_right_norm;
    Eigen::Matrix3d T_left  = doHartleyNormalization(pts_left, pts_left_norm);
    Eigen::Matrix3d T_right = doHartleyNormalization(pts_right, pts_right_norm);

    // Create the linear system of equations for the 8-point algorithm
        // x'^T * F * x = 0
        // Af = 0; 
            // A = rows of Kroneecker products of point correspondences
                // a = (x'x, x'y, x', y'x, y'y, y', x, y, 1)
            // f = vectorized form of F (9x1): (f11, f12, f13, f21, f22, f23, f31, f32, f33)
        
    for (size_t i = 0; i < pts_left_norm.size(); ++i) {
        Eigen::Vector3d pt_left = pts_left_norm[i];
        Eigen::Vector3d pt_right = pts_right_norm[i];

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

    // Undo the normalization: F_rank2 is only valid in normalized coordinates
    Eigen::Matrix3d F = T_right.transpose() * F_rank2 * T_left;

    return F;
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

    // TODO: consider LO-RANSAC (local optimization: refit F from inliers within the loop, not just once at the end)
    for (size_t i = 0; i < m_num_iters; ++i) {
        // Randomly sample 8 points
        std::shuffle(indices.begin(), indices.end(), rng);  // Randomly Shuffle indices
        std::vector<Eigen::Vector3d> sample_left, sample_right;
        for (size_t j = 0; j < m_min_sample_size; ++j) {
            sample_left.push_back(pts_left[indices[j]]);
            sample_right.push_back(pts_right[indices[j]]);
        }

        // Skip degenerate samples (all points collinear in either image).
        if (isNearCollinear(sample_left, kMinPerpDistPx) || isNearCollinear(sample_right, kMinPerpDistPx))
            continue;

        // Estimate F_rank2 from the 8 samples
        Eigen::Matrix3d F = estimateFundamentalMatrix(sample_left, sample_right);

        // Compute inliers based on the Sampson distance
        int inlier_count = 0;
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