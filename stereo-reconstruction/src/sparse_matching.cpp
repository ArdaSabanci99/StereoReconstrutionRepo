#include "sparse_matching.h"
#include "Eigen.h"
#include <opencv2/core/eigen.hpp>   // cv::cv2eigen / cv::eigen2cv
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// ── Sparse feature matching + pose recovery (Member 1) ────────────────────

SparseMatchResult computeSparseMatches(const cv::Mat& left, const cv::Mat& right,
                                        const CalibData& calib) {
    SparseMatchResult result;

    cv::Mat gray_l, gray_r;
    cv::cvtColor(left,  gray_l, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, gray_r, cv::COLOR_BGR2GRAY);

    // TODO-1: Keypoint detection — SIFT (or ORB for speed)
    //   cv::Ptr<cv::SIFT> detector = cv::SIFT::create(2000);
    //   std::vector<cv::KeyPoint> kp_l, kp_r;
    //   cv::Mat desc_l, desc_r;
    //   detector->detectAndCompute(gray_l, cv::noArray(), kp_l, desc_l);
    //   detector->detectAndCompute(gray_r, cv::noArray(), kp_r, desc_r);

    // TODO-2: Feature matching — BFMatcher + Lowe ratio test (threshold 0.75)
    //   cv::BFMatcher matcher(cv::NORM_L2);
    //   std::vector<std::vector<cv::DMatch>> knn_matches;
    //   matcher.knnMatch(desc_l, desc_r, knn_matches, 2);
    //   for (auto& m : knn_matches)
    //       if (m[0].distance < 0.75f * m[1].distance) {
    //           result.pts_left .push_back(kp_l[m[0].queryIdx].pt);
    //           result.pts_right.push_back(kp_r[m[0].trainIdx].pt);
    //       }

    // TODO-3: 8-point algorithm — estimate F via RANSAC, enforce rank-2
    //   result.F = cv::findFundamentalMat(result.pts_left, result.pts_right,
    //                                     cv::FM_RANSAC, 1.0, 0.99,
    //                                     result.inlier_mask);
    //
    //   Enforce rank-2 using Eigen SVD (cleaner than cv::SVD):
    //   Eigen::Matrix3d F_eigen;
    //   cv::cv2eigen(result.F, F_eigen);
    //   Eigen::JacobiSVD<Eigen::Matrix3d> svd(F_eigen,
    //       Eigen::ComputeFullU | Eigen::ComputeFullV);
    //   Eigen::Vector3d s = svd.singularValues();
    //   s(2) = 0;
    //   F_eigen = svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();
    //   cv::eigen2cv(F_eigen, result.F);

    // TODO-4: Essential matrix  E = K1^T * F * K0
    //   result.E = calib.K1.t() * result.F * calib.K0;

    // TODO-5: Recover R, t from E (pick the solution with positive depth)
    //   Filter to inlier matches first:
    //   std::vector<cv::Point2f> inlier_l, inlier_r;
    //   for (size_t i = 0; i < result.inlier_mask.size(); ++i)
    //       if (result.inlier_mask[i]) {
    //           inlier_l.push_back(result.pts_left[i]);
    //           inlier_r.push_back(result.pts_right[i]);
    //       }
    //   cv::recoverPose(result.E, inlier_l, inlier_r, calib.K0,
    //                   result.R, result.t);

    // TODO-6 (evaluation hook): print inlier count and mean reprojection error
    //   int n_inliers = cv::countNonZero(result.inlier_mask);
    //   std::cout << "[sparse] matches=" << result.pts_left.size()
    //             << "  inliers=" << n_inliers << "\n";

    std::cout << "[sparse_matching] Not yet implemented.\n";
    return result;
}

#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: sparse_matching <scene_dir>\n";
        return 1;
    }
    fs::path scene(argv[1]);
    CalibData calib = loadCalib(scene);
    cv::Mat left  = cv::imread((scene / "im0.png").string());
    cv::Mat right = cv::imread((scene / "im1.png").string());
    if (left.empty() || right.empty()) {
        std::cerr << "Could not load images from " << scene << "\n"; return 1;
    }

    SparseMatchResult r = computeSparseMatches(left, right, calib);

    if (!r.F.empty()) std::cout << "F =\n" << r.F << "\n";
    if (!r.R.empty()) std::cout << "R =\n" << r.R << "\n";
    if (!r.t.empty()) std::cout << "t =\n" << r.t << "\n";

    return 0;
}
#endif
