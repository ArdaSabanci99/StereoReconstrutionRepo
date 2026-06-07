#include "rectification.h"
#include "sparse_matching.h"
#include <iostream>

// ── OpenCV baseline (DONE) ─────────────────────────────────────────────────
RectifyResult rectifyOpenCV(const cv::Mat& left, const cv::Mat& right,
                             const CalibData& calib) {
    cv::Size img_size(left.cols, left.rows);

    cv::Mat D0 = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat D1 = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat T  = (cv::Mat_<double>(3,1) << -calib.baseline, 0, 0);
    cv::Mat R  = cv::Mat::eye(3, 3, CV_64F);

    cv::Mat R0, R1, P1, P2, Q;
    cv::stereoRectify(calib.K0, D0, calib.K1, D1,
                      img_size, R, T,
                      R0, R1, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0);

    cv::Mat map0x, map0y, map1x, map1y;
    cv::initUndistortRectifyMap(calib.K0, D0, R0, P1, img_size, CV_32FC1, map0x, map0y);
    cv::initUndistortRectifyMap(calib.K1, D1, R1, P2, img_size, CV_32FC1, map1x, map1y);

    RectifyResult result;
    cv::remap(left,  result.left_rect,  map0x, map0y, cv::INTER_LINEAR);
    cv::remap(right, result.right_rect, map1x, map1y, cv::INTER_LINEAR);
    result.Q  = Q;
    result.P1 = P1;
    result.P2 = P2;

    std::cout << "[rectification] OpenCV done.\n";
    return result;
}

// ── Manual rectification — Loop & Zhang 1999 (Member 2) ───────────────────
RectifyResult rectifyManual(const cv::Mat& left, const cv::Mat& right,
                             const CalibData& calib) {
    // Step 1: Get relative pose from sparse matching (Member 1's output)
    SparseMatchResult sparse = computeSparseMatches(left, right, calib);

    if (sparse.R.empty()) {
        std::cout << "[rectification] Pose recovery failed, falling back to OpenCV.\n";
        return rectifyOpenCV(left, right, calib);
    }

    // TODO-6: Compute rectifying homographies from recovered R and t
    //
    //   Option A — use OpenCV stereoRectify with our R, t:
    //   cv::Size img_size(left.cols, left.rows);
    //   cv::Mat D0 = cv::Mat::zeros(5, 1, CV_64F);
    //   cv::Mat D1 = cv::Mat::zeros(5, 1, CV_64F);
    //   cv::Mat R0, R1, P1, P2, Q;
    //   cv::stereoRectify(calib.K0, D0, calib.K1, D1,
    //                     img_size, sparse.R, sparse.t,
    //                     R0, R1, P1, P2, Q,
    //                     cv::CALIB_ZERO_DISPARITY, 0);
    //
    //   Option B — Loop & Zhang direct homography (see paper):
    //   Compute epipole e0 in the left image from F:
    //     SVD(F^T) → e0 is left null-vector (last column of V)
    //   Build H0 that maps e0 to infinity (projective transform)
    //   Find H1 minimising vertical disparity via affine correction:
    //     Ha = argmin Σ ||Ha * H0 * m_l - H1 * m_r||^2

    // TODO-7: Warp both images with computed rectification maps
    //   cv::Mat map0x, map0y, map1x, map1y;
    //   cv::initUndistortRectifyMap(calib.K0, D0, R0, P1, img_size, CV_32FC1, map0x, map0y);
    //   cv::initUndistortRectifyMap(calib.K1, D1, R1, P2, img_size, CV_32FC1, map1x, map1y);
    //   cv::remap(left,  result.left_rect,  map0x, map0y, cv::INTER_LINEAR);
    //   cv::remap(right, result.right_rect, map1x, map1y, cv::INTER_LINEAR);
    //   result.Q = Q; result.P1 = P1; result.P2 = P2;

    std::cout << "[rectification] Manual (TODO-6/7) not yet implemented, falling back to OpenCV.\n";
    return rectifyOpenCV(left, right, calib);
}

#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: rectification <scene_dir> [--manual]\n";
        return 1;
    }
    fs::path scene(argv[1]);
    bool manual = (argc >= 3 && std::string(argv[2]) == "--manual");

    CalibData calib = loadCalib(scene);
    cv::Mat left  = cv::imread((scene / "im0.png").string());
    cv::Mat right = cv::imread((scene / "im1.png").string());

    if (left.empty() || right.empty()) {
        std::cerr << "Could not load images from " << scene << "\n";
        return 1;
    }

    auto result = manual ? rectifyManual(left, right, calib)
                         : rectifyOpenCV(left, right, calib);

    fs::create_directories("results");
    cv::imwrite("results/left_rect.png",  result.left_rect);
    cv::imwrite("results/right_rect.png", result.right_rect);
    std::cout << "Saved rectified images to results/\n";
    return 0;
}
#endif
