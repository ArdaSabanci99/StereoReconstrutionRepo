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
                      sz, calib.R_rel, calib.T_rel,
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
// 1. Data Structures & Conversion Helpers
// ─────────────────────────────────────────────────────────────────────────────

struct Point2D { float x; float y; };


// Helper to convert Eigen::Matrix3d to cv::Mat
static cv::Mat eigenToCv(const Eigen::Matrix3d& src) {
    cv::Mat dst(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            dst.at<double>(i, j) = src(i, j);
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Core Math Engine
// ─────────────────────────────────────────────────────────────────────────────

// Compute the epipole (null space of the given matrix)
static Eigen::Vector3d computeEpipole(const Eigen::Matrix3d& F) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullV);
    Eigen::Vector3d e = svd.matrixV().col(2);
    return e / e.z(); // Normalize to homogeneous coordinates
}

// Build the projective warp that sends an epipole to infinity
static Eigen::Matrix3d buildEpipoleWarp(const Eigen::Vector3d& epipole, int W, int H) {
    double cx = W / 2.0;
    double cy = H / 2.0;

    // Step 1: Translate image center to origin
    Eigen::Matrix3d T;
    T << 1, 0, -cx,
        0, 1, -cy,
        0, 0, 1;

    Eigen::Vector3d e_translated = T * epipole;

    // Step 2: Rotate the epipole so it lies on the X-axis
    double x = e_translated.x();
    double y = e_translated.y();
    double theta = std::atan2(y, x);

    // Prevent 180-degree upside-down flips by snapping to the closest X-axis side
    if (theta > M_PI / 2.0) theta -= M_PI;
    if (theta < -M_PI / 2.0) theta += M_PI;

    double c = std::cos(theta);
    double s = std::sin(theta);

    Eigen::Matrix3d R;
    R << c, s, 0,
        -s, c, 0,
        0, 0, 1;

    Eigen::Vector3d e_rotated = R * e_translated;

    // Step 3: Project the epipole to infinity along the X-axis
    double f = e_rotated.x();
    if (std::abs(f) < 1e-9) f = 1e-9; // Prevent division by zero

    Eigen::Matrix3d G;
    G << 1, 0, 0,
        0, 1, 0,
        -1.0 / f, 0, 1;

    // Step 4: Translate back to the original image center
    Eigen::Matrix3d T_inv;
    T_inv << 1, 0, cx,
        0, 1, cy,
        0, 0, 1;

    return T_inv * G * R * T;
}

// Build the Affine correction matrix to align the left image to the right image
static Eigen::Matrix3d buildAffineAlignment(const Eigen::Matrix3d& H1_prime,
    const Eigen::Matrix3d& H2,
    const std::vector<Point2D>& pts_l,
    const std::vector<Point2D>& pts_r) {
    int N = static_cast<int>(pts_l.size());
    if (N < 4) return H1_prime; // Fallback if insufficient points

    Eigen::MatrixXd Ax(N, 3), Ay(N, 2);
    Eigen::VectorXd bx(N), by(N);

    // Lambda to manually warp a point via a 3x3 Homography
    auto warpPoint = [](const Eigen::Matrix3d& H, const Point2D& p) -> Point2D {
        Eigen::Vector3d x(p.x, p.y, 1.0);
        Eigen::Vector3d y = H * x;
        return { static_cast<float>(y.x() / y.z()), static_cast<float>(y.y() / y.z()) };
        };

    for (int i = 0; i < N; ++i) {
        Point2D wl = warpPoint(H1_prime, pts_l[i]);
        Point2D wr = warpPoint(H2, pts_r[i]);

        // System 1: Align X (fixes horizontal scale, shift, and skew)
        Ax(i, 0) = wl.x;
        Ax(i, 1) = wl.y;
        Ax(i, 2) = 1.0;
        bx(i) = wr.x;

        // System 2: Align Y (fixes vertical scale and shift. NO X-DEPENDENCE to preserve horizontal lines)
        Ay(i, 0) = wl.y;
        Ay(i, 1) = 1.0;
        by(i) = wr.y;
    }

    // Solve using SVD
    Eigen::Vector3d abc = Ax.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bx);
    Eigen::Vector2d ef = Ay.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(by);

    Eigen::Matrix3d Ha;
    Ha << abc(0), abc(1), abc(2),
        0, ef(0), ef(1),
        0, 0, 1;

    // Apply affine correction to the initial left warp
    return Ha * H1_prime;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. The Loop-Zhang Method
// ─────────────────────────────────────────────────────────────────────────────

RectifyResult rectifyLoopZhang(const cv::Mat& left, const cv::Mat& right,
    const cv::Mat& F_cv,
    const std::vector<cv::Point2f>& cv_pts_l,
    const std::vector<cv::Point2f>& cv_pts_r) {
    RectifyResult res;
    int W = left.cols;
    int H = left.rows;

    // Convert OpenCV Fundamental Matrix to Eigen
    Eigen::Matrix3d F;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            F(i, j) = F_cv.at<double>(i, j);

    // 1. Find the Epipoles
    // F * e1 = 0 (Left Epipole), F^T * e2 = 0 (Right Epipole)
    Eigen::Vector3d e1 = computeEpipole(F);
    Eigen::Vector3d e2 = computeEpipole(F.transpose());

    // 2. Build the initial perspective homographies to send epipoles to infinity
    Eigen::Matrix3d H1_prime = buildEpipoleWarp(e1, W, H);
    Eigen::Matrix3d H2 = buildEpipoleWarp(e2, W, H);

    // 3. Extract the inlier matched points
    std::vector<Point2D> in_l, in_r;
    for (size_t i = 0; i < cv_pts_l.size(); ++i) {
        in_l.push_back({ cv_pts_l[i].x, cv_pts_l[i].y });
        in_r.push_back({ cv_pts_r[i].x, cv_pts_r[i].y });
    }

    // 4. Calculate the Affine Homography to align the images
    Eigen::Matrix3d H1 = buildAffineAlignment(H1_prime, H2, in_l, in_r);

    // 5. Convert back to OpenCV and warp the images
    res.H1 = eigenToCv(H1);
    res.H2 = eigenToCv(H2);

    cv::Size sz(W, H);
    cv::warpPerspective(left, res.left_rect, res.H1, sz, cv::INTER_LINEAR);
    cv::warpPerspective(right, res.right_rect, res.H2, sz, cv::INTER_LINEAR);

    std::cout << "[rectify] Custom Loop-Zhang fully executed.\n";
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone executable
// ─────────────────────────────────────────────────────────────────────────────
#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: rectification <data_path> <scene_id> <left_id> <right_id> [--manual] [--light N]\n";
        return 1;
    }
    fs::path dataPath(argv[1]);
    const std::string sceneId(argv[2]);
    const std::string viewL = padViewId(std::stoi(argv[3]));
    const std::string viewR = padViewId(std::stoi(argv[4]));
    bool manual = false;
    std::string lightId = "0";
    for (int i = 5; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--manual") manual = true;
        else if (a == "--light" && i+1 < argc) lightId = argv[++i];
    }

    DTUDataLoader loader(dataPath.string());
    CalibData calib = loader.loadCalib(viewL, viewR);
    cv::Mat imgL = loader.loadImage(sceneId, viewL, lightId);
    cv::Mat imgR = loader.loadImage(sceneId, viewR, lightId);
    if (imgL.empty() || imgR.empty()) { std::cerr << "Could not load images.\n"; return 1; }

    RectifyResult res;
    if (manual) {
        SparseMatchResult sp = computeSparseMatches(imgL, imgR, calib);
        if (sp.F.empty()) {
            std::cerr << "Sparse failed.\n";
            return 1;
        }

        // Extract only the verified inlier points for the Loop-Zhang affine alignment step
        std::vector<cv::Point2f> inliersL, inliersR;
        inliersL.reserve(sp.n_inliers);
        inliersR.reserve(sp.n_inliers);

        for (int i = 0; i < sp.n_matches; ++i) {
            if (sp.inlier_mask[i]) {
                inliersL.push_back(sp.pts_left[i]);
                inliersR.push_back(sp.pts_right[i]);
            }
        }

        // Call the specific Loop-Zhang orchestration function with the filtered points
        res = rectifyLoopZhang(imgL, imgR, sp.F, inliersL, inliersR);
    }
    else {
        res = rectifyOpenCV(imgL, imgR, calib);
    }

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
