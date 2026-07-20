#include "rectification.h"
#include "sparse_matching.h"
#include "DataLoader.h"
#include "utils.h"
#include "Eigen.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <array>


// -----------------------------------------------------------------------------
// OpenCV calibrated rectification
// -----------------------------------------------------------------------------

/**
 * @brief OpenCV calibrated rectification implementation.
 *
 * The remapping masks are produced from an all-white source image with the
 * same maps as the photographs. A zero mask value therefore means that the
 * output pixel came from outside the original image.
 */
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
    cv::remap(left,  res.left_rect,  m0x, m0y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(right, res.right_rect, m1x, m1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // Build masks with the exact same sampling maps as the images.
    cv::Mat sourceMask(sz, CV_8U, cv::Scalar(255));
    cv::remap(sourceMask, res.mask1, m0x, m0y, cv::INTER_NEAREST,
              cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::remap(sourceMask, res.mask2, m1x, m1y, cv::INTER_NEAREST,
              cv::BORDER_CONSTANT, cv::Scalar(0));

    res.Q = Q;
    res.P1 = P1;
    res.P2 = P2;
    res.R0_rect = R0;

    // cv::Mat K1_rect = P1(cv::Rect(0, 0, 3, 3));
    // cv::Mat K2_rect = P2(cv::Rect(0, 0, 3, 3));

    // res.H1 = K1_rect * R0 * calib.K0.inv();
    // res.H2 = K2_rect * R1 * calib.K1.inv();
    
    std::cout << "[rectification] OpenCV done.\n";
    
    return res;
}

/**
 * @brief Find the epipole as the null vector of the fundamental matrix.
 *
 * @param[in] Fundamental matrix F.
 * @return Epipole in homogeneous coordinates (3-vector).
 */
static Eigen::Vector3d computeEpipole(const Eigen::Matrix3d& F) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullV);
    Eigen::Vector3d epipole = svd.matrixV().col(2);

    if (std::abs(epipole.z()) < 1e-12 || !epipole.allFinite())
        throw std::runtime_error("Cannot normalize the Loop-Zhang epipole");
    
    return epipole / epipole.z();
}

/** 
 * @brief Return the matrix that computes a cross product with v. 
 * 
 * @param[in] v 3D vector
 * @return 3x3 skew-symmetric matrix S such that S * w = v x w for any 3D vector w.
 */
static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S <<      0, -v.z(),  v.y(),
          v.z(),      0, -v.x(),
         -v.y(),  v.x(),      0;
    
    return S;
}

/** @brief Transform one image point and divide by its homogeneous scale. */
static Eigen::Vector2d warpPointH(const Eigen::Matrix3d& H, double x, double y) {
    Eigen::Vector3d p(x, y, 1.0);
    Eigen::Vector3d q = H * p;
    return Eigen::Vector2d(q.x() / q.z(), q.y() / q.z());
}

/** @brief Axis-aligned bounds of a warped image. */
struct BBox { double xmin, xmax, ymin, ymax; };

/** @brief Compute the bounds of the four transformed image corners. */
static BBox warpedBBox(const Eigen::Matrix3d& H, int W, int Hh) {
    BBox b{ 1e18, -1e18, 1e18, -1e18 };
    double xs[4] = { 0.0, (double)W, 0.0,       (double)W };
    double ys[4] = { 0.0, 0.0,       (double)Hh, (double)Hh };
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector2d p = warpPointH(H, xs[i], ys[i]);
        b.xmin = std::min(b.xmin, p.x()); b.xmax = std::max(b.xmax, p.x());
        b.ymin = std::min(b.ymin, p.y()); b.ymax = std::max(b.ymax, p.y());
    }
    return b;
}

/**
 * @brief Transform the four image corners in polygon order.
 *
 * The order is top-left, top-right, bottom-right, bottom-left.  Keeping the
 * corners in this order is useful when a later step needs to inspect the
 * actual warped quadrilateral rather than only its axis-aligned bounds.
 */
static std::array<Eigen::Vector2d, 4> warpedQuad(
    const Eigen::Matrix3d& H, int W, int Hh) {
    const double xs[4] = {0.0, static_cast<double>(W),
                          static_cast<double>(W), 0.0};
    const double ys[4] = {0.0, 0.0,
                          static_cast<double>(Hh), static_cast<double>(Hh)};

    std::array<Eigen::Vector2d, 4> quad;
    for (int i = 0; i < 4; ++i)
        quad[i] = warpPointH(H, xs[i], ys[i]);
    return quad;
}

// -----------------------------------------------------------------------------
// Shared Loop-Zhang output stage
//
// 
// -----------------------------------------------------------------------------
/**
 * @brief Finalize the rectification process by applying a shared canvas transform.
 *        The core algorithm produces two raw homographies. 
 *        This stage applies one shared canvas transform, warps the images and masks, 
 *        and updates the camera projection matrices used by triangulation.
 * 
 * @param H1 
 * @param H2 
 * @param left 
 * @param right 
 * @param calib 
 * @param W 
 * @param H 
 * @return RectifyResult 
 */
static RectifyResult finalizeRectification(
    const Eigen::Matrix3d& H1, const Eigen::Matrix3d& H2,
    const cv::Mat& left, const cv::Mat& right, const CalibData& calib,
    int W, int H) {
    RectifyResult res;

    BBox b1 = warpedBBox(H1, W, H);
    BBox b2 = warpedBBox(H2, W, H);

    const double xmin = std::min(b1.xmin, b2.xmin);
    const double xmax = std::max(b1.xmax, b2.xmax);
    const double ymin = std::min(b1.ymin, b2.ymin);
    const double ymax = std::max(b1.ymax, b2.ymax);
    const double spanX = xmax - xmin;
    const double spanY = ymax - ymin;
    if (!(spanX > 1e-9) || !(spanY > 1e-9) ||
        !std::isfinite(spanX) || !std::isfinite(spanY))
        throw std::runtime_error("Invalid Loop-Zhang warped bounds");

    const double sxFit = static_cast<double>(W) / spanX;
    const double syFit = static_cast<double>(H) / spanY;
    const double sharedScale = std::min(sxFit, syFit);
    const double sx = sharedScale;
    const double sy = sharedScale;

    const double tx = -xmin * sx + 0.5 * (W - spanX * sx);
    const double ty = -ymin * sy + 0.5 * (H - spanY * sy);

    Eigen::Matrix3d C;
    C << sx, 0, tx,
         0, sy, ty,
         0,  0,  1;
    Eigen::Matrix3d H1_final = C * H1;
    Eigen::Matrix3d H2_final = C * H2;

    std::cout << "[rectify] shared canvas: sx=" << sx << " sy=" << sy
              << " tx=" << tx << " ty=" << ty << "\n";

    res.H1 = eigenToCv(H1_final);
    res.H2 = eigenToCv(H2_final);

    cv::Size sz(W, H);
    cv::warpPerspective(left, res.left_rect, res.H1, sz, cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar());
    cv::warpPerspective(right, res.right_rect, res.H2, sz, cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar());

    cv::Mat sourceMask(H, W, CV_8U, cv::Scalar(255));
    cv::warpPerspective(sourceMask, res.mask1, res.H1, sz, cv::INTER_NEAREST,
                        cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::warpPerspective(sourceMask, res.mask2, res.H2, sz, cv::INTER_NEAREST,
                        cv::BORDER_CONSTANT, cv::Scalar(0));

    Eigen::Matrix3d K0 = cvToEigen3x3(calib.K0);
    Eigen::Matrix3d K1 = cvToEigen3x3(calib.K1);
    Eigen::Matrix3d R_rel = cvToEigen3x3(calib.R_rel);
    Eigen::Vector3d t_rel = cvToEigenVec3(calib.t_rel);

    Eigen::Matrix<double, 3, 4> P1_orig;
    P1_orig << K0, Eigen::Vector3d::Zero();

    Eigen::Matrix<double, 3, 4> Rt;
    Rt << R_rel, t_rel;
    Eigen::Matrix<double, 3, 4> P2_orig = K1 * Rt;

    res.P1 = eigenToCv34(H1_final * P1_orig);
    res.P2 = eigenToCv34(H2_final * P2_orig);

    res.R0_rect = cv::Mat::eye(3, 3, CV_64F);

    return res;
}

/**
 * @brief Construct a homeography whose projective row makes the left epipole map to a point at infinity.
 * 
 * @param[in] e Left epipole
 * @param[in] z Chosen direction 
 * @return Eigen::Matrix3d Left projective homeography
 */
static Eigen::Matrix3d buildLeftProjectiveHomeography(const Eigen::Vector3d& e, const Eigen::Vector3d& z) {
    Eigen::Vector3d w = skewSymmetric(e) * z;
    
    double wz = w.z();
    if (std::abs(wz) < 1e-9 || !std::isfinite(wz))
        throw std::runtime_error("Degenerate Loop-Zhang left projective transform");
    w /= wz;

    Eigen::Matrix3d projective_homeography;
    projective_homeography << 1, 0, 0,
                              0, 1, 0,
                              w.x(), w.y(), 1;
    return projective_homeography;
}

/**
 * @brief Build the right projective homography that sends the right epipole to infinity along a given direction.
 *        Fz is a right epipolar line, it is orthogonal to the right epipole.
 *        So, we can use Fz as the projective row to send the epipole to infinity.
 * @param[in] F Fundamental matrix
 * @param[in] z Chosen direction
 * @return Eigen::Matrix3d Right projective homeography
 */
static Eigen::Matrix3d buildRightProjectiveHomeography(const Eigen::Matrix3d& F, const Eigen::Vector3d& z) {
    Eigen::Vector3d wp = F * z;
    double wz = wp.z();
    if (std::abs(wz) < 1e-9 || !std::isfinite(wz))
        throw std::runtime_error("Degenerate Loop-Zhang right projective transform");
    wp /= wz;

    Eigen::Matrix3d projective_homeography;
    projective_homeography << 1, 0, 0,
                              0, 1, 0,
                              wp.x(), wp.y(), 1;
    return projective_homeography;
}



struct DistortionMats { Eigen::Matrix3d A, B, Ap, Bp; };

/**
 * @brief Precomputs four matrices that let us quickly measure how much distortion a candidate direction would cause in both images
 *        A, B describe the left image using its epipole and image geometry
 *        A', B' describe the right image using the fundamental matrix
 * @param left_epipole 
 * @param F 
 * @param width 
 * @param height 
 * @return DistortionMats 
 */
static DistortionMats computeDistortionMatrices(const Eigen::Vector3d& left_epipole, const Eigen::Matrix3d& F,
                                int width, int height) {
    
    // Covariance of the pixel coordinates in the image
    const double w = static_cast<double>(width);
    const double h = static_cast<double>(height);

    
    Eigen::Matrix3d pixel_spread = Eigen::Matrix3d::Zero();
    // Pixel spread concerns only the x- and y- coordinates
        // How much the projective scaling varies across the whole image
    pixel_spread(0, 0) = (w * h / 12.0) * (w * w - 1.0);
    pixel_spread(1, 1) = (w * h / 12.0) * (h * h - 1.0);
    
    

    // Create image center
    const Eigen::Vector3d image_center(0.5 * w, 0.5 * h, 1.0);
    const Eigen::Matrix3d center_moment = image_center * image_center.transpose();  // Measures the projective scale at the center of the image

    
    // Computes the projective row of the homeography given the epipole and the direction
        // Measures how much the projective scaling varies across the whole image
    const Eigen::Matrix3d left_proj_row = skewSymmetric(left_epipole);

    DistortionMats matrices;

    matrices.A = left_proj_row.transpose() * pixel_spread *left_proj_row;
    matrices.B = left_proj_row.transpose() * center_moment * left_proj_row;

    matrices.Ap = F.transpose() * pixel_spread * F;
    matrices.Bp = F.transpose() * center_moment * F;
    
    return matrices;
}

/**
 * @brief Computes the distortion objective for a given angle.
 *        Compute distoriton in both images and sum them. 
 * @param phi Candidate angle
 * @param dist_matrices Precomputed distortion matrices for the left and right images.
 * @return double 
 */
static double distortionObjective(double phi, const DistortionMats& dist_matrices) {
    // Convert candidate angle into rectificaiton direction
    Eigen::Vector3d z(std::cos(phi), std::sin(phi), 0.0);
    
    double den1 = z.dot(dist_matrices.B  * z);
    double den2 = z.dot(dist_matrices.Bp * z);
    
    if (std::abs(den1) < 1e-9 || std::abs(den2) < 1e-9)
        return std::numeric_limits<double>::infinity();
    
    double num1 = z.dot(dist_matrices.A  * z);
    double num2 = z.dot(dist_matrices.Ap * z);
    
    return num1 / den1 + num2 / den2;
}

/**
 * @brief Find the direction that sends the epipoles to infinity 
 *        while introducing as little distortion as possible in both images. 
 * 
 * @param[in] dist_matrices Precomputed distortion matrices for the left and right images.
 * @return double Direction angle that minimizes distortion in both images.
 */
static double findOptimalPhi(const DistortionMats& dist_matrices) {
    auto objective = [&](double phi) {
        return distortionObjective(phi, dist_matrices);
    };

    // Grid search over directions
    constexpr int k_grid_samples = 3600; // 0.05 deg resolution
    double best_phi = 0.0, best_cost = std::numeric_limits<double>::infinity();
    for (int i = 0; i < k_grid_samples; ++i) {
        double phi = M_PI * (double)i / (double)k_grid_samples;
        double cost = objective(phi);
        if (cost < best_cost) { best_cost = cost; best_phi = phi; }
    }

    // Refine the best direction with golden-section search
    const double golden_section_ratio =(std::sqrt(5.0) - 1.0) / 2.0;
    double step = M_PI / k_grid_samples;
    
    double lo = best_phi - step, hi = best_phi + step;
    double x1 = hi - golden_section_ratio * (hi - lo);
    double x2 = lo + golden_section_ratio * (hi - lo);
    
    double f1 = objective(x1);
    double f2 = objective(x2);
    
    for (int it = 0; it < 60 && (hi - lo) > 1e-10; ++it) {
        if (f1 < f2) {
            hi = x2; x2 = x1; f2 = f1;
            x1 = hi - golden_section_ratio * (hi - lo);
            f1 = objective(x1);
        } else {
            lo = x1; x1 = x2; f1 = f2;
            x2 = lo + golden_section_ratio * (hi - lo);
            f2 = objective(x2);
        }
    }
    return 0.5 * (lo + hi);
}


static std::pair<Eigen::Matrix3d, Eigen::Matrix3d> buildSimilarityProjection(
    const Eigen::Matrix3d& F, const Eigen::Vector3d& w, const Eigen::Vector3d& wp,
    double vcPrime) {
    Eigen::Matrix3d Hsim = Eigen::Matrix3d::Zero();
    Hsim(0, 0) = F(2, 1) - w.y() * F(2, 2);
    Hsim(0, 1) = w.x() * F(2, 2) - F(2, 0);
    Hsim(1, 0) = F(2, 0) - w.x() * F(2, 2);
    Hsim(1, 1) = Hsim(0, 0);
    Hsim(1, 2) = F(2, 2) + vcPrime;
    Hsim(2, 2) = 1.0;

    Eigen::Matrix3d HsimP = Eigen::Matrix3d::Zero();
    HsimP(0, 0) = wp.y() * F(2, 2) - F(1, 2);
    HsimP(0, 1) = F(0, 2) - wp.x() * F(2, 2);
    HsimP(1, 0) = wp.x() * F(2, 2) - F(0, 2);
    HsimP(1, 1) = HsimP(0, 0);
    HsimP(1, 2) = vcPrime;
    HsimP(2, 2) = 1.0;

    return { Hsim, HsimP };
}


/**
 * @brief Computes the vertical translation needed to align the two warped images.
 *      The translation is computed by finding the minimum y-coordinate of the warped corners of both images
 * 
 * @param left_proj_homeography 
 * @param right_proj_homeography 
 * @param F 
 * @param w 
 * @param wp 
 * @param width 
 * @param height 
 * @return double 
 */
static double findCommonVerticalTranslation(const Eigen::Matrix3d& left_proj_homeography, const Eigen::Matrix3d& right_proj_homeography,
                            const Eigen::Matrix3d& F,
                            const Eigen::Vector3d& w, const Eigen::Vector3d& wp,
                            int width, int height) {
    auto [Hsim0, HsimP0] = buildSimilarityProjection(F, w, wp, 0.0);
    Eigen::Matrix3d preL = Hsim0  * left_proj_homeography;
    Eigen::Matrix3d preR = HsimP0 * right_proj_homeography;

    auto quadL = warpedQuad(preL, width, height);
    auto quadR = warpedQuad(preR, width, height);

    double minV = std::numeric_limits<double>::infinity();
    for (const auto& p : quadL) minV = std::min(minV, p.y());
    for (const auto& p : quadR) minV = std::min(minV, p.y());

    return -minV;
}


static Eigen::Matrix3d buildShear(const Eigen::Matrix3d& H_partial, int W, int Hh) {
    Eigen::Vector3d a(W / 2.0 - 1, 0,            1); // top edge midpoint
    Eigen::Vector3d b(0,           Hh / 2.0 - 1, 1); // left edge midpoint
    Eigen::Vector3d c(W / 2.0 - 1, Hh - 1,       1); // bottom edge midpoint
    Eigen::Vector3d d(W - 1,       Hh / 2.0 - 1, 1); // right edge midpoint

    auto warp = [&](const Eigen::Vector3d& p) {
        Eigen::Vector3d q = H_partial * p;
        return Eigen::Vector2d(q.x() / q.z(), q.y() / q.z());
    };
    Eigen::Vector2d ah = warp(a), bh = warp(b), ch = warp(c), dh = warp(d);
    Eigen::Vector2d xhat = bh - dh; // x_hat = (xu, xv)
    Eigen::Vector2d yhat = ah - ch; // y_hat = (yu, yv)

    double xu = xhat.x(), xv = xhat.y();
    double yu = yhat.x(), yv = yhat.y();
    double M = W, N = Hh;

    double denom = M * N * (xv * yu - xu * yv);
    double sa = (N * N * xv * xv + M * M * yv * yv) / denom;
    double sb = (N * N * xu * xv + M * M * yu * yv) / -denom;

    if (sa < 0.0) {
        sa = -sa;
        sb = -sb;
    }

    Eigen::Matrix3d Hsh = Eigen::Matrix3d::Identity();
    Hsh(0, 0) = sa; Hsh(0, 1) = sb;
    return Hsh;
}


RectifyResult rectifyLoopZhang(const cv::Mat& left, const cv::Mat& right,
    const CalibData& calib) {
    int W = left.cols;
    int H = left.rows;

    Eigen::Matrix3d F = cvToEigen3x3(calib.F);
    
    // 1. Compute the left epipole
    Eigen::Vector3d e1 = computeEpipole(F);

    // 2. Find a direction that minimizes distortion in both images
    DistortionMats dist_matrices = computeDistortionMatrices(e1, F, W, H);
    double phi = findOptimalPhi(dist_matrices);
    Eigen::Vector3d z(std::cos(phi), std::sin(phi), 0.0);

    // 3: Find warps that send the epipoles to infinity along the chosen direction z -> parallel lines in infinity
    Eigen::Matrix3d proj_homeography_left  = buildLeftProjectiveHomeography(e1, z);
    Eigen::Matrix3d proj_homeography_right = buildRightProjectiveHomeography(F, z);

    Eigen::Vector3d w  = proj_homeography_left.row(2).transpose();
    Eigen::Vector3d wp = proj_homeography_right.row(2).transpose();

    // 4. Build the similarity transforms to rotate, scale and vertically shift the two images
        // So the corresponding epipolar lines lie on the same horizontal rows
        // Projective homeographies make teh epipolar lines parallel (not horizontal and aligned between images)
    
    double vertical_trans = findCommonVerticalTranslation(proj_homeography_left, proj_homeography_right, F, w, wp, W, H);
    auto [sim_left, sim_right] = buildSimilarityProjection(F, w, wp, vertical_trans);

    // Combine the projective and row-alignment stages.
    Eigen::Matrix3d left_aligned = sim_left  * proj_homeography_left;
    Eigen::Matrix3d right_aligned = sim_right * proj_homeography_right;

    // Reduce shape distortion without changing the rectified row coordinate
    Eigen::Matrix3d left_shear  = buildShear(left_aligned, W, H);
    Eigen::Matrix3d right_shear = buildShear(right_aligned, W, H);

    // Compose the raw homographies before fitting them to the output canvas
    Eigen::Matrix3d H1 = left_shear  * left_aligned;
    Eigen::Matrix3d H2 = right_shear * right_aligned;

    // Place both warped views on one shared canvas. 
    // The same scale and translation are applied to both images, so row alignment is preserved
    return finalizeRectification(H1, H2, left, right, calib, W, H);
}


#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: rectification <data_path> <scene_id> <left_id> <right_id> "
                     "[--manual]\n"
                     "  --manual    use Loop-Zhang rectification instead of OpenCV\n";
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
    if (!calib.hasIntrinsics() || !calib.hasRelativePose() || !calib.hasFundamentalMatrix()) {
        std::cout << "[rectification]: Missing intrinsics, relative pose, or fundamental matrix — run sparse_matching first.\n";
        throw std::runtime_error("Incomplete calibration: " + calib_path);
    }

    std::string effLeftId = calib.swapped ? viewR : viewL;
    std::string effRightId = calib.swapped ? viewL : viewR;

    cv::Mat imgL = loader.loadImage(sceneId, effLeftId);
    cv::Mat imgR = loader.loadImage(sceneId, effRightId);
    if (imgL.empty() || imgR.empty()) { std::cerr << "Could not load images.\n"; return 1; }

    std::string tag = !manual ? "opencv" : "loopzhang";
    RectifyResult res = !manual ? rectifyOpenCV(imgL, imgR, calib)
                                 : rectifyLoopZhang(imgL, imgR, calib);

    std::string savePath = "results/scene" + sceneId + "/rectification/" + tag;
    fs::create_directories(savePath);
    cv::imwrite(savePath + "/view_" + viewL + ".png",  res.left_rect);
    cv::imwrite(savePath + "/view_" + viewR + ".png",  res.right_rect);
    cv::imwrite(savePath + "/mask_" + viewL + ".png",  res.mask1);
    cv::imwrite(savePath + "/mask_" + viewR + ".png",  res.mask2);

    // Persist P1/P2/Q (and H1/H2, for reference) so depth.cpp can pick the
    // correct triangulation path without recomputing them from calib alone --
    // Q is only valid for the OpenCV path and is left empty for Loop-Zhang.
    {
        cv::FileStorage fsOut(savePath + "/proj_" + viewL + "_" + viewR + ".yaml",
                               cv::FileStorage::WRITE);
        fsOut << "P1" << res.P1;
        fsOut << "P2" << res.P2;
        fsOut << "Q"  << res.Q;   // empty Mat for Loop-Zhang -- depth.cpp checks .empty()
        fsOut << "H1" << res.H1;
        fsOut << "H2" << res.H2;
        fsOut.release();
    }

    cv::Mat side;
    cv::hconcat(res.left_rect, res.right_rect, side);

    for (int y = 0; y < side.rows; y += 40) {
        const bool validInLeft =
            cv::countNonZero(res.mask1.row(y)) > 0;

        const bool validInRight =
            cv::countNonZero(res.mask2.row(y)) > 0;

        if (validInLeft && validInRight) {
            cv::line(side,cv::Point(0, y), cv::Point(side.cols - 1, y), cv::Scalar(0, 255, 0), 1);
        }
    }
    cv::imwrite(savePath + "/epipolar_lines_" + viewL + "_" + viewR + ".png", side);
    
    std::cout << "Saved rectified images to " << savePath << "\n";
    return 0;
}
#endif