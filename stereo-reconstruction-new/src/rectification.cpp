#include "rectification.h"
#include "sparse_matching.h"
#include "DataLoader.h"
#include "Eigen.h"
#include <iostream>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// OpenCV baseline
// ─────────────────────────────────────────────────────────────────────────────
RectifyResult rectifyOpenCV(const cv::Mat& left, const cv::Mat& right,
                              const CalibData& calib) {
    cv::Size sz(left.cols, left.rows);
    cv::Mat D0 = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat D1 = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat R0, R1, P1, P2, Q;
    cv::stereoRectify(calib.K0, D0, calib.K1, D1,
                      sz, calib.R_rel, calib.t_rel,
                      R0, R1, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1);

    cv::Mat m0x, m0y, m1x, m1y;
    cv::initUndistortRectifyMap(calib.K0, D0, R0, P1, sz, CV_32FC1, m0x, m0y);
    cv::initUndistortRectifyMap(calib.K1, D1, R1, P2, sz, CV_32FC1, m1x, m1y);

    RectifyResult res;
    cv::remap(left,  res.left_rect,  m0x, m0y, cv::INTER_LINEAR);
    cv::remap(right, res.right_rect, m1x, m1y, cv::INTER_LINEAR);
    res.Q       = Q;
    res.P1      = P1;
    res.P2      = P2;
    res.R0_rect = R0;
    std::cout << "[rectification] OpenCV done.\n";
    return res;
}


// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Build a 3×3 rotation matrix that rotates 3-vector v to [norm(v), 0, 0]
static cv::Mat rotationToXAxis(cv::Mat v) {
    v /= cv::norm(v);
    double x = v.at<double>(0), y = v.at<double>(1);
    double r = std::sqrt(x*x + y*y);
    if (r < 1e-12) return cv::Mat::eye(3, 3, CV_64F);

    // Rotation around z-axis by -atan2(y, x)
    double c = x/r, s = -y/r;
    cv::Mat Rz = (cv::Mat_<double>(3,3)
        <<  c, -s, 0,
            s,  c, 0,
            0,  0, 1);
    return Rz;
}

// Compute epipole e: F^T e = 0  (null vector of F^T)
static cv::Mat computeEpipole(const cv::Mat& Ft) {
    cv::Mat U, S, Vt;
    cv::SVD::compute(Ft, S, U, Vt, cv::SVD::FULL_UV);
    cv::Mat e = Vt.row(2).t();  // last row of Vt = last column of V
    e /= e.at<double>(2);       // normalise so e[2] = 1
    return e;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build rectification homography H2 that maps the right epipole to infinity
// on the x-axis.  (Loop & Zhang §3)
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat buildH2(const cv::Mat& e2, int W, int H) {
    // Step 1: Translate image centre to origin
    cv::Mat T = (cv::Mat_<double>(3,3)
        << 1, 0, -W/2.0,
           0, 1, -H/2.0,
           0, 0,  1);

    cv::Mat e2t = T * e2;  // translate epipole

    // Step 2: Rotate so e2t lies on positive x-axis
    cv::Mat R = rotationToXAxis(e2t);

    // Step 3: Send to infinity: G = [1,0,0; 0,1,0; -1/f, 0, 1]
    cv::Mat Re = R * e2t;
    double f = Re.at<double>(0);       // distance from origin after rotation
    if (std::abs(f) < 1e-9) f = 1e-9;
    cv::Mat G = (cv::Mat_<double>(3,3)
        << 1, 0,     0,
           0, 1,     0,
           -1.0/f, 0, 1);

    // H2 = T^-1 G R T  (translate back after warp so the image stays centred)
    cv::Mat T_inv = (cv::Mat_<double>(3,3)
        << 1, 0,  W/2.0,
           0, 1,  H/2.0,
           0, 0,  1);

    return T_inv * G * R * T;
}

// ─────────────────────────────────────────────────────────────────────────────
// Find H1 by minimising vertical disparity (affine correction)
// We want  Ha * H2 * m_l  ≈  H1 * m_r  for corresponding points.
// Set H0 = H2 * inv(K0) * K1  as a starting guess transfer,
// then fit Ha (affine row-1) via least squares.
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat buildH1(const cv::Mat& H2,
                        const std::vector<cv::Point2f>& pts_l,
                        const std::vector<cv::Point2f>& pts_r,
                        const cv::Mat& F,
                        int W, int H) {
    // M = H2 * F  — the transfer from right to left in rectified space
    // We want H1 = Ha * M minimising sum_i ||Ha * H2 * p_l_i - H2 * p_r_i||_y^2
    // Simpler: minimise vertical displacement after applying H2 to both sides
    // → solve for a,b,c in Ha = [a b c; 0 1 0; 0 0 1] on warped points.

    // Warp the matched points through H2 (right) and a candidate H1=H2 (left)
    int N = (int)pts_l.size();
    if (N < 4) return H2;  // fallback

    // System: for each i:
    //   [H2*p_l_i]_y = a * [H2*p_r_i]_x + b * [H2*p_r_i]_y + c
    cv::Mat A_ls(N, 3, CV_64F);
    cv::Mat b_ls(N, 1, CV_64F);

    auto warpPt = [](const cv::Mat& H, cv::Point2f p) -> cv::Point2f {
        cv::Mat x = (cv::Mat_<double>(3,1) << p.x, p.y, 1.0);
        cv::Mat y = H * x;
        double w = y.at<double>(2);
        return { (float)(y.at<double>(0)/w), (float)(y.at<double>(1)/w) };
    };

    for (int i = 0; i < N; ++i) {
        cv::Point2f wr = warpPt(H2, pts_r[i]);   // right point in H2-space
        cv::Point2f wl = warpPt(H2, pts_l[i]);   // left  point through H2
        double* ar = A_ls.ptr<double>(i);
        ar[0] = wr.x; ar[1] = wr.y; ar[2] = 1.0;
        b_ls.at<double>(i) = wl.y;    // we want the y of the left to match
    }

    cv::Mat abc;
    cv::solve(A_ls, b_ls, abc, cv::DECOMP_SVD);

    double a = abc.at<double>(0), b2 = abc.at<double>(1), c = abc.at<double>(2);
    cv::Mat Ha = (cv::Mat_<double>(3,3)
        << a,  b2, c,
           0,   1, 0,
           0,   0, 1);

    return Ha * H2;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build Q matrix from rectified camera parameters
// Q = [ 1  0  0  -cx;  0  1  0  -cy;  0  0  0   f;  0  0  -1/Tx  0 ]
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat buildQ(const cv::Mat& K, double baseline) {
    double f  = K.at<double>(0,0);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);
    double Tx = baseline;   // caller passes baseline already negated (-B); right camera is to the right
    cv::Mat Q = (cv::Mat_<double>(4,4)
        <<  1,  0,  0,        -cx,
            0,  1,  0,        -cy,
            0,  0,  0,          f,
            0,  0, -1.0/Tx,     0);
    return Q;
}

// ─────────────────────────────────────────────────────────────────────────────
// Manual Loop-Zhang rectification
// ─────────────────────────────────────────────────────────────────────────────
RectifyResult rectifyManual(const cv::Mat& left, const cv::Mat& right,
                              const CalibData& calib,
                              const std::vector<cv::Point2f>& in_l,
                              const std::vector<cv::Point2f>& in_r) {
    RectifyResult res;
    int W = left.cols, H = left.rows;

    // ── 1. Compute epipoles ──────────────────────────────────────────────
    //   e1: null(F^T) → F e1 = 0   (left  epipole, seen from left  image)
    //   e2: null(F)   → F^T e2 = 0 (right epipole, seen from right image)
    cv::Mat e1 = computeEpipole(calib.F.t());   // F  * e1 = 0
    cv::Mat e2 = computeEpipole(calib.F);       // F^T* e2 = 0

    std::cout << "[rectify] e1 = " << e1.t() << "\n"
              << "[rectify] e2 = " << e2.t() << "\n";

    // ── 2. H2: send right epipole to infinity ────────────────────────────
    res.H2 = buildH2(e2, W, H);

    // ── 4. H1: affine correction to minimise vertical disparity ──────────
    res.H1 = buildH1(res.H2, in_l, in_r, calib.F, W, H);

    // ── 5. Warp images ───────────────────────────────────────────────────
    cv::Size sz(W, H);
    cv::warpPerspective(left,  res.left_rect,  res.H1, sz, cv::INTER_LINEAR);
    cv::warpPerspective(right, res.right_rect, res.H2, sz, cv::INTER_LINEAR);

    // ── 6. Build Q matrix for depth computation ───────────────────────────
    // Estimate baseline from calibration T_rel
    double B = cv::norm(calib.t_rel);
    res.Q = buildQ(calib.K0, -B);

    // Store R0_rect as identity (we used homographies, not rotation decomp)
    res.R0_rect = cv::Mat::eye(3, 3, CV_64F);

    std::cout << "[rectify] Loop-Zhang done. Baseline=" << B << " m\n";

    // ── 7. Check vertical alignment quality ───────────────────────────────
    {
        auto warpPt = [](const cv::Mat& Hh, cv::Point2f p) -> cv::Point2f {
            cv::Mat x = (cv::Mat_<double>(3,1) << p.x, p.y, 1.0);
            cv::Mat y = Hh * x;
            double w = y.at<double>(2);
            return { (float)(y.at<double>(0)/w), (float)(y.at<double>(1)/w) };
        };
        double yerr = 0; int cnt = 0;
        for (size_t i = 0; i < in_l.size() && i < 200; ++i) {
            cv::Point2f wl = warpPt(res.H1, in_l[i]);
            cv::Point2f wr = warpPt(res.H2, in_r[i]);
            yerr += std::abs(wl.y - wr.y);
            ++cnt;
        }
        std::cout << "[rectify] Mean vertical disparity after rectification: "
                  << (cnt > 0 ? yerr/cnt : 0.0) << " px\n";
    }

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone executable
// ─────────────────────────────────────────────────────────────────────────────
#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: rectification <data_path> <scene_id> <left_id> <right_id> [--manual]\n";
        return 1;
    }
    fs::path dataPath(argv[1]);
    const std::string sceneId(argv[2]);
    const std::string viewL = padViewId(std::stoi(argv[3]));
    const std::string viewR = padViewId(std::stoi(argv[4]));
    bool manual = false;
    for (int i = 5; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--manual") manual = true;
    }

    DTUDataLoader loader(dataPath.string());
    std::string calib_path = "results/scene" + sceneId + "/sparse_matching/calib_" + viewL + "_" + viewR + ".yaml";
    CalibData calib = loadCalibData(calib_path);
    if (!calib.hasIntrinsics() || !calib.hasRelativePose()) {
        std::cout << "[rectification]: Missing intrinsics or relative pose — run sparse_matching first.\n";
        throw std::runtime_error("Incomplete calibration: " + calib_path);
    }

    std::string effLeftId = calib.swapped ? viewR : viewL;
    std::string effRightId = calib.swapped ? viewL : viewR;

    cv::Mat imgL = loader.loadImage(sceneId, effLeftId);
    cv::Mat imgR = loader.loadImage(sceneId, effRightId);
    if (imgL.empty() || imgR.empty()) { std::cerr << "Could not load images.\n"; return 1; }
    

    std::vector<cv::Point2f> in_l, in_r;
    if (manual) {
        std::string inliers_path = "results/scene" + sceneId +
            "/sparse_matching/inliers_" + viewL + "_" + viewR + ".yaml";
        loadInlierPoints(inliers_path, in_l, in_r);
    }
    RectifyResult res = manual ? rectifyManual(imgL, imgR, calib, in_l, in_r)
                               : rectifyOpenCV(imgL, imgR, calib);

    std::string savePath = "results/scene" + sceneId + "/rectification";
    fs::create_directories(savePath);
    cv::imwrite(savePath + "/view_" + viewL + ".png",  res.left_rect);
    cv::imwrite(savePath + "/view_" + viewR + ".png",  res.right_rect);

    // Side-by-side comparison with horizontal lines
    cv::Mat side;
    cv::hconcat(res.left_rect, res.right_rect, side);
    for (int y = 0; y < side.rows; y += 40)
        cv::line(side, {0, y}, {side.cols, y}, {0, 255, 0}, 1);
    cv::imwrite(savePath + "/side_by_side_" + viewL + "_" + viewR + ".png", side);

    std::cout << "Saved rectified images to " << savePath << "\n";
    return 0;
}
#endif
