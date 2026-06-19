#include "sparse_matching.h"
#include "DataLoader.h"
#include "Eigen.h"
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <filesystem>
#include <random>
#include <numeric>
#include <cmath>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build normalisation transform T so that:
//   - centroid of pts → origin
//   - mean distance from origin → sqrt(2)
// Returns T (3×3, double) and writes normalised points to pts_n.
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat normalisePoints(const std::vector<cv::Point2f>& pts,
                                std::vector<cv::Point2f>& pts_n) {
    // Centroid
    double cx = 0, cy = 0;
    for (const auto& p : pts) { cx += p.x; cy += p.y; }
    cx /= pts.size(); cy /= pts.size();

    // Mean distance from centroid
    double dist = 0;
    for (const auto& p : pts)
        dist += std::sqrt((p.x - cx)*(p.x - cx) + (p.y - cy)*(p.y - cy));
    dist /= pts.size();

    double scale = (dist < 1e-9) ? 1.0 : std::sqrt(2.0) / dist;

    pts_n.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
        pts_n[i] = { (float)((pts[i].x - cx) * scale),
                     (float)((pts[i].y - cy) * scale) };

    cv::Mat T = (cv::Mat_<double>(3,3)
        << scale,   0,  -scale * cx,
              0, scale, -scale * cy,
              0,     0,           1);
    return T;
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalised 8-point algorithm (Hartley 1997)
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat estimateFundamentalManual(const std::vector<cv::Point2f>& pts1,
                                   const std::vector<cv::Point2f>& pts2) {
    assert(pts1.size() == pts2.size() && pts1.size() >= 8);
    int N = (int)pts1.size();

    // 1. Normalise
    std::vector<cv::Point2f> p1n, p2n;
    cv::Mat T1 = normalisePoints(pts1, p1n);
    cv::Mat T2 = normalisePoints(pts2, p2n);

    // 2. Build A (N × 9)
    //    Each row: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    cv::Mat A(N, 9, CV_64F);
    for (int i = 0; i < N; ++i) {
        double x1 = p1n[i].x, y1 = p1n[i].y;
        double x2 = p2n[i].x, y2 = p2n[i].y;
        double* row = A.ptr<double>(i);
        row[0]=x2*x1; row[1]=x2*y1; row[2]=x2;
        row[3]=y2*x1; row[4]=y2*y1; row[5]=y2;
        row[6]=x1;    row[7]=y1;    row[8]=1.0;
    }

    // 3. SVD(A) → f is last column of V
    cv::Mat U, S, Vt;
    cv::SVD::compute(A, S, U, Vt, cv::SVD::FULL_UV);
    cv::Mat f = Vt.row(8).reshape(0, 3);         // 3×3

    // 4. Enforce rank-2: SVD(F), zero out smallest singular value
    cv::SVD::compute(f, S, U, Vt, cv::SVD::FULL_UV);
    S.at<double>(2) = 0.0;
    cv::Mat F_tilde = U * cv::Mat::diag(S) * Vt;

    // 5. Denormalise: F = T2^T * F_tilde * T1
    cv::Mat F = T2.t() * F_tilde * T1;

    // 6. Normalise so ||F||_F = 1
    double nrm = cv::norm(F);
    if (nrm > 1e-12) F /= nrm;

    return F;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampson distance (first-order geometric error)
//   d = (x2^T F x1)^2 / ( (Fx1)[0]^2 + (Fx1)[1]^2 + (F^T x2)[0]^2 + (F^T x2)[1]^2 )
// ─────────────────────────────────────────────────────────────────────────────
static double sampsonDistance(const cv::Mat& F,
                               const cv::Point2f& p1,
                               const cv::Point2f& p2) {
    cv::Mat x1 = (cv::Mat_<double>(3,1) << p1.x, p1.y, 1.0);
    cv::Mat x2 = (cv::Mat_<double>(3,1) << p2.x, p2.y, 1.0);

    cv::Mat Fx1  = F  * x1;
    cv::Mat Ftx2 = F.t() * x2;
    double  num  = cv::Mat(x2.t() * Fx1).at<double>(0,0);
    double  den  = Fx1.at<double>(0)*Fx1.at<double>(0)
                 + Fx1.at<double>(1)*Fx1.at<double>(1)
                 + Ftx2.at<double>(0)*Ftx2.at<double>(0)
                 + Ftx2.at<double>(1)*Ftx2.at<double>(1);
    return (den < 1e-12) ? 1e12 : (num * num) / den;
}

// ─────────────────────────────────────────────────────────────────────────────
// RANSAC wrapper around the manual 8-point estimator
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat ransacFundamental(const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2,
                                  std::vector<uchar>& inlier_mask,
                                  double threshold = 1.5,
                                  int max_iter     = 2000,
                                  double conf      = 0.99) {
    int N = (int)pts1.size();
    inlier_mask.assign(N, 0);

    std::mt19937 rng(42);
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    cv::Mat best_F;
    int best_inliers = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Sample 8 random indices
        std::shuffle(idx.begin(), idx.end(), rng);
        std::vector<cv::Point2f> s1(8), s2(8);
        for (int i = 0; i < 8; ++i) {
            s1[i] = pts1[idx[i]];
            s2[i] = pts2[idx[i]];
        }

        cv::Mat F = estimateFundamentalManual(s1, s2);
        if (F.empty()) continue;

        // Count inliers
        std::vector<uchar> mask(N, 0);
        int n_in = 0;
        for (int i = 0; i < N; ++i) {
            if (sampsonDistance(F, pts1[i], pts2[i]) < threshold) {
                mask[i] = 1; ++n_in;
            }
        }

        if (n_in > best_inliers) {
            best_inliers = n_in;
            best_F       = F.clone();
            inlier_mask  = mask;

            // Adaptive iteration count
            double w = (double)n_in / N;
            if (w > 1.0 - 1e-9) break;
            double new_max = std::log(1.0 - conf) /
                             std::log(1.0 - std::pow(w, 8));
            max_iter = std::min(max_iter, (int)std::ceil(new_max));
        }
    }

    // Re-estimate F on all inliers
    std::vector<cv::Point2f> in1, in2;
    for (int i = 0; i < N; ++i)
        if (inlier_mask[i]) { in1.push_back(pts1[i]); in2.push_back(pts2[i]); }

    if ((int)in1.size() >= 8)
        best_F = estimateFundamentalManual(in1, in2);

    return best_F;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cheirality test: returns the (R,t) pair from the 4 SVD candidates where the
// most points triangulate in front of both cameras.
// ─────────────────────────────────────────────────────────────────────────────
static void recoverRtFromE(const cv::Mat& E,
                            const std::vector<cv::Point2f>& pts1,
                            const std::vector<cv::Point2f>& pts2,
                            const cv::Mat& K,
                            cv::Mat& R_out, cv::Mat& t_out) {
    // SVD of E
    cv::Mat U, S, Vt;
    cv::SVD::compute(E, S, U, Vt);

    // Enforce singular values = [1,1,0]
    cv::Mat S2 = (cv::Mat_<double>(3,1) << 1, 1, 0);
    cv::Mat E2 = U * cv::Mat::diag(S2) * Vt;
    cv::SVD::compute(E2, S, U, Vt);

    // Ensure det(U)*det(Vt) = +1 (proper rotation)
    if (cv::determinant(U) < 0) U  *= -1.0;
    if (cv::determinant(Vt)< 0) Vt *= -1.0;

    cv::Mat W = (cv::Mat_<double>(3,3) << 0,-1,0, 1,0,0, 0,0,1);

    cv::Mat R1 = U *  W   * Vt;
    cv::Mat R2 = U *  W.t()* Vt;
    cv::Mat t_pos =  U.col(2);
    cv::Mat t_neg = -U.col(2);

    // 4 candidate solutions
    cv::Mat Rs[4] = { R1, R1, R2, R2 };
    cv::Mat ts[4] = { t_pos, t_neg, t_pos, t_neg };

    // Count positive-depth points for each candidate (simplified cheirality)
    auto countPositiveDepth = [&](const cv::Mat& R, const cv::Mat& t) -> int {
        // P0 = K [I | 0]   P1 = K [R | t]
        cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
        cv::Mat Rt(3, 4, CV_64F);
        R.copyTo(Rt.colRange(0,3));
        t.copyTo(Rt.col(3));
        cv::Mat P1 = K * Rt;

        int count = 0;
        for (size_t i = 0; i < std::min(pts1.size(), (size_t)50); ++i) {
            // Linear triangulation
            cv::Mat A(4, 4, CV_64F);
            A.row(0) = pts1[i].x * P0.row(2) - P0.row(0);
            A.row(1) = pts1[i].y * P0.row(2) - P0.row(1);
            A.row(2) = pts2[i].x * P1.row(2) - P1.row(0);
            A.row(3) = pts2[i].y * P1.row(2) - P1.row(1);
            cv::Mat Ua, Sa, Va;
            cv::SVD::compute(A, Sa, Ua, Va, cv::SVD::FULL_UV);
            cv::Mat X = Va.row(3).t();  // homogeneous
            double w  = X.at<double>(3);
            double Z0 = X.at<double>(2) / w;                         // depth cam0
            cv::Mat Xc1 = R * (X.rowRange(0,3) / w) + t;
            double Z1  = Xc1.at<double>(2);
            if (Z0 > 0 && Z1 > 0) ++count;
        }
        return count;
    };

    int best = -1, best_count = -1;
    for (int i = 0; i < 4; ++i) {
        int c = countPositiveDepth(Rs[i], ts[i]);
        if (c > best_count) { best_count = c; best = i; }
    }
    R_out = Rs[best].clone();
    t_out = ts[best].clone();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main sparse matching pipeline
// ─────────────────────────────────────────────────────────────────────────────
SparseMatchResult computeSparseMatches(const cv::Mat& left,
                                        const cv::Mat& right,
                                        const CalibData& calib) {
    SparseMatchResult result;

    cv::Mat gray_l, gray_r;
    if (left.channels()  == 3) cv::cvtColor(left,  gray_l, cv::COLOR_BGR2GRAY);
    else gray_l = left;
    if (right.channels() == 3) cv::cvtColor(right, gray_r, cv::COLOR_BGR2GRAY);
    else gray_r = right;

    // ── 1. SIFT detection ─────────────────────────────────────────────────
    auto sift = cv::SIFT::create(3000);
    std::vector<cv::KeyPoint> kp_l, kp_r;
    cv::Mat desc_l, desc_r;
    sift->detectAndCompute(gray_l, cv::noArray(), kp_l, desc_l);
    sift->detectAndCompute(gray_r, cv::noArray(), kp_r, desc_r);

    if (kp_l.empty() || kp_r.empty()) {
        std::cerr << "[sparse] No keypoints found.\n";
        return result;
    }

    // ── 2. BFMatcher + Lowe ratio test ────────────────────────────────────
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc_l, desc_r, knn, 2);

    for (auto& m : knn) {
        if (m.size() < 2) continue;
        if (m[0].distance < 0.75f * m[1].distance) {
            result.pts_left .push_back(kp_l[m[0].queryIdx].pt);
            result.pts_right.push_back(kp_r[m[0].trainIdx].pt);
        }
    }
    result.n_matches = (int)result.pts_left.size();
    std::cout << "[sparse] Keypoints L=" << kp_l.size()
              << " R=" << kp_r.size()
              << "  Matches after Lowe=" << result.n_matches << "\n";

    if (result.n_matches < 8) {
        std::cerr << "[sparse] Too few matches.\n";
        return result;
    }

    // ── 3. RANSAC + manual 8-point ────────────────────────────────────────
    result.F = ransacFundamental(result.pts_left, result.pts_right,
                                  result.inlier_mask,
                                  /*threshold=*/1.5,
                                  /*max_iter=*/ 3000);

    result.n_inliers = 0;
    for (uchar v : result.inlier_mask) if (v) ++result.n_inliers;

    std::cout << "[sparse] RANSAC inliers=" << result.n_inliers
              << " / " << result.n_matches << "\n";

    if (result.F.empty()) {
        std::cerr << "[sparse] F estimation failed.\n";
        return result;
    }

    // ── 4. Mean epipolar error on inliers ─────────────────────────────────
    double err_sum = 0; int n = 0;
    for (int i = 0; i < result.n_matches; ++i) {
        if (!result.inlier_mask[i]) continue;
        err_sum += sampsonDistance(result.F, result.pts_left[i], result.pts_right[i]);
        ++n;
    }
    result.mean_epipolar_error = (n > 0) ? err_sum / n : 0.0;
    std::cout << "[sparse] Mean Sampson error = " << result.mean_epipolar_error << "\n";

    // ── 5. Essential matrix ────────────────────────────────────────────────
    result.E = calib.K1.t() * result.F * calib.K0;

    // ── 6. Recover R, t (cheirality test) ─────────────────────────────────
    std::vector<cv::Point2f> in_l, in_r;
    for (int i = 0; i < result.n_matches; ++i)
        if (result.inlier_mask[i]) {
            in_l.push_back(result.pts_left[i]);
            in_r.push_back(result.pts_right[i]);
        }

    recoverRtFromE(result.E, in_l, in_r, calib.K0, result.R, result.t);

    std::cout << "[sparse] R =\n" << result.R << "\n"
              << "[sparse] t =  " << result.t.t() << "\n";

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone executable
// ─────────────────────────────────────────────────────────────────────────────
#if !defined(PIPELINE_BUILD) && !defined(BUILDING_RECTIFICATION)
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: sparse_matching <data_path> <scene_id> <left_id> <right_id> [--light N]\n";
        return 1;
    }
    fs::path dataPath(argv[1]);
    const std::string sceneId(argv[2]);
    const std::string viewL = padViewId(std::stoi(argv[3]));
    const std::string viewR = padViewId(std::stoi(argv[4]));
    std::string lightId = "0";
    for (int i = 5; i < argc; ++i)
        if (std::string(argv[i]) == "--light" && i+1 < argc) lightId = argv[++i];

    DTUDataLoader loader(dataPath.string());
    CalibData calib = loader.loadCalib(viewL, viewR);
    cv::Mat imgL = loader.loadImage(sceneId, viewL, lightId);
    cv::Mat imgR = loader.loadImage(sceneId, viewR, lightId);
    if (imgL.empty() || imgR.empty()) { std::cerr << "Could not load images.\n"; return 1; }

    SparseMatchResult r = computeSparseMatches(imgL, imgR, calib);

    // Save match visualisation
    std::vector<cv::KeyPoint> kpL, kpR;
    std::vector<cv::DMatch>   good;
    // Re-detect for visualisation
    auto sift = cv::SIFT::create(3000);
    cv::Mat dL, dR;
    sift->detectAndCompute(imgL, cv::noArray(), kpL, dL);
    sift->detectAndCompute(imgR, cv::noArray(), kpR, dR);
    // Draw first 50 inlier matches approximately
    cv::Mat vis;
    if (!kpL.empty() && !kpR.empty()) {
        std::vector<cv::DMatch> dm;
        cv::BFMatcher bfm(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> knn;
        bfm.knnMatch(dL, dR, knn, 2);
        for (auto& m : knn)
            if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
                dm.push_back(m[0]);
        std::vector<cv::DMatch> subset(dm.begin(), dm.begin() + std::min((int)dm.size(), 50));
        cv::drawMatches(imgL, kpL, imgR, kpR, subset, vis);
        std::string savePath = "results/scene" + sceneId + "/sparse";
        fs::create_directories(savePath);
        cv::imwrite(savePath + "/matches_" + viewL + "_" + viewR + ".png", vis);
        std::cout << "Match visualisation saved.\n";
    }
    return 0;
}
#endif
