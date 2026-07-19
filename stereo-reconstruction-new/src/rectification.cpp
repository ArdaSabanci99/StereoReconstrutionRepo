#include "rectification.h"
#include "sparse_matching.h"
#include "DataLoader.h"
#include "Eigen.h"
#include <iostream>
#include <cmath>
#include <array>
#include <limits>

// ─────────────────────────────────────────────────────────────────────────────
// NOTE: RectifyResult (rectification.h) must have these two fields added --
// they are new in this version and are not yet declared in the header:
//
//     cv::Mat mask1, mask2;  // CV_8U, 255 = valid content, 0 = black warp padding
//
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// OpenCV baseline
// ─────────────────────────────────────────────────────────────────────────────

// Tilt/valid-fraction fit diagnostic for this (OpenCV) rectification path --
// kept independent (no Eigen) so this path stays self-contained.
static cv::Point2d warpPointCv(const cv::Mat& H, double x, double y) {
    cv::Mat p = (cv::Mat_<double>(3, 1) << x, y, 1.0);
    cv::Mat q = H * p;
    double w = q.at<double>(2, 0);
    return { q.at<double>(0, 0) / w, q.at<double>(1, 0) / w };
}

static void logRectificationFit(const cv::Mat& H_left, const cv::Mat& H_right, int W, int Hh) {
    auto fit = [&](const cv::Mat& H) {
        double xs[4] = { 0.0, (double)W, (double)W, 0.0 };
        double ys[4] = { 0.0, 0.0,       (double)Hh, (double)Hh };
        cv::Point2d q[4];
        for (int i = 0; i < 4; ++i) q[i] = warpPointCv(H, xs[i], ys[i]);
        double area2 = 0.0;
        for (int i = 0; i < 4; ++i) {
            const auto& p = q[i]; const auto& n = q[(i + 1) % 4];
            area2 += p.x * n.y - n.x * p.y;
        }
        double frac = std::abs(area2) / 2.0 / (double(W) * double(Hh));
        double tilt = std::atan2(q[1].y - q[0].y, q[1].x - q[0].x) * 180.0 / CV_PI;
        return std::make_pair(tilt, frac);
    };
    auto [tilt1, frac1] = fit(H_left);
    auto [tilt2, frac2] = fit(H_right);
    std::cout << "[rectify] canvas fit — left: tilt=" << tilt1
              << "deg valid=" << (frac1 * 100.0) << "% | right: tilt=" << tilt2
              << "deg valid=" << (frac2 * 100.0) << "%\n";
}

// Rasterize a homography's warped W x H quad into a binary validity mask:
// 255 where a pixel in the (W,H) output canvas is covered by actual warped
// source content, 0 where it's black warp padding. Used by matching.cpp to
// invalidate disparity computed over padding (see the padding/SGM-noise
// discussion this was added for).
static cv::Mat rasterizeQuadMaskCv(const cv::Mat& H, int W, int Hh) {
    double xs[4] = { 0.0, (double)W, (double)W, 0.0 };
    double ys[4] = { 0.0, 0.0,       (double)Hh, (double)Hh };
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (int i = 0; i < 4; ++i) {
        cv::Point2d p = warpPointCv(H, xs[i], ys[i]);
        poly.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    cv::Mat mask = cv::Mat::zeros(Hh, W, CV_8U);
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    return mask;
}

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

    // Forward (original-pixel -> rectified-pixel) homography implied by each
    // camera's rectifying rotation, H = newK * R * K^-1 -- same quantity
    // warpPerspective uses in the Loop-Zhang path, just derived from
    // stereoRectify's R/newK instead of an explicit projective homography.
    cv::Mat newK0 = P1(cv::Rect(0, 0, 3, 3));
    cv::Mat newK1 = P2(cv::Rect(0, 0, 3, 3));
    cv::Mat H_left  = newK0 * R0 * calib.K0.inv();
    cv::Mat H_right = newK1 * R1 * calib.K1.inv();
    logRectificationFit(H_left, H_right, sz.width, sz.height);

    RectifyResult res;
    cv::remap(left,  res.left_rect,  m0x, m0y, cv::INTER_LINEAR);
    cv::remap(right, res.right_rect, m1x, m1y, cv::INTER_LINEAR);
    res.Q       = Q;
    res.P1      = P1;
    res.P2      = P2;
    res.R0_rect = R0;

    // OpenCV's alpha=-1 already crops/zooms to minimize black padding, but
    // there can still be a residual sliver at the extreme edges; derive the
    // mask the same way as the Loop-Zhang path so matching.cpp has a single
    // consistent mask contract regardless of which rectifier ran.
    res.mask1 = rasterizeQuadMaskCv(H_left,  sz.width, sz.height);
    res.mask2 = rasterizeQuadMaskCv(H_right, sz.width, sz.height);

    std::cout << "[rectification] OpenCV done.\n";
    return res;
}


// ─────────────────────────────────────────────────────────────────────────────
// 1. Conversion Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Helper to convert Eigen::Matrix3d to cv::Mat
static cv::Mat eigenToCv(const Eigen::Matrix3d& src) {
    cv::Mat dst(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            dst.at<double>(i, j) = src(i, j);
    return dst;
}

// Helper to convert a 3x4 Eigen matrix (projection matrix) to cv::Mat
static cv::Mat eigenToCv34(const Eigen::Matrix<double, 3, 4>& src) {
    cv::Mat dst(3, 4, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            dst.at<double>(i, j) = src(i, j);
    return dst;
}

// Helpers to convert cv::Mat (CV_64F) intrinsics/extrinsics to Eigen
static Eigen::Matrix3d cvToEigen3x3(const cv::Mat& src) {
    Eigen::Matrix3d dst;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            dst(i, j) = src.at<double>(i, j);
    return dst;
}

static Eigen::Vector3d cvToEigenVec3(const cv::Mat& src) {
    return Eigen::Vector3d(src.at<double>(0), src.at<double>(1), src.at<double>(2));
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

// Skew-symmetric cross-product matrix: skewSymmetric(v) * x == v.cross(x)
static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S <<      0, -v.z(),  v.y(),
          v.z(),      0, -v.x(),
         -v.y(),  v.x(),      0;
    return S;
}

// Warp a single 2D point through a 3x3 homography
static Eigen::Vector2d warpPointH(const Eigen::Matrix3d& H, double x, double y) {
    Eigen::Vector3d p(x, y, 1.0);
    Eigen::Vector3d q = H * p;
    return Eigen::Vector2d(q.x() / q.z(), q.y() / q.z());
}

struct BBox { double xmin, xmax, ymin, ymax; };

// Bounding box of an image's 4 corners after warping through H
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

// Warp the image's 4 corners through H, in polygon (not bbox-min/max) order,
// so callers can measure the actual occupied quadrilateral rather than its bbox.
static std::array<Eigen::Vector2d, 4> warpedQuad(const Eigen::Matrix3d& H, int W, int Hh) {
    double xs[4] = { 0.0, (double)W, (double)W, 0.0 };
    double ys[4] = { 0.0, 0.0,       (double)Hh, (double)Hh };
    std::array<Eigen::Vector2d, 4> q;
    for (int i = 0; i < 4; ++i) q[i] = warpPointH(H, xs[i], ys[i]);
    return q;
}

// Determinant of a homography's upper-left 2x2 (linear) block. A rectifying
// homography that represents a physically valid camera view must be
// orientation-preserving (positive determinant); a negative determinant
// means the transform contains a reflection, which no rigid re-viewing of
// the same scene can produce, and makes H*[K|0] an invalid camera matrix.
static double linearDet2x2(const Eigen::Matrix3d& H) {
    return H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0);
}

// Rasterize a homography's warped W x H quad (Eigen version) into a binary
// validity mask -- same contract as rasterizeQuadMaskCv above, used by the
// Loop-Zhang path's finalizeRectification.
static cv::Mat rasterizeQuadMask(const Eigen::Matrix3d& H, int W, int Hh) {
    auto quad = warpedQuad(H, W, Hh);
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (const auto& p : quad) poly.emplace_back(cvRound(p.x()), cvRound(p.y()));
    cv::Mat mask = cv::Mat::zeros(Hh, W, CV_8U);
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    return mask;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Shared tail: canvas fit, warp, projection matrices.
//    Used by every Loop-Zhang variant below once H1/H2 (pre-canvas-fit) are
//    known — independent of how H1/H2 were constructed.
// ─────────────────────────────────────────────────────────────────────────────

static RectifyResult finalizeRectification(
    const Eigen::Matrix3d& H1, const Eigen::Matrix3d& H2,
    const cv::Mat& left, const cv::Mat& right, const CalibData& calib,
    int W, int H,
    bool preserveAspectRatio = false) {
    RectifyResult res;

    // 1. Fit the warped content back into the (W,H) canvas. The vertical
    //    scale/offset MUST be shared by both images to preserve the row
    //    alignment just established; the horizontal scale is also shared
    //    (keeps the disparity range sane for the matching stage), with an
    //    independent horizontal offset per image (a harmless disparity shift).
    //
    //    NOTE: this is a zero-crop, "fit everything, pad the rest with
    //    black" fit -- for wide-aspect-ratio images with even modest tilt,
    //    forcing a same-aspect-ratio zoom-to-fill (no black padding) turns
    //    out to require discarding the majority of the frame (see prior
    //    analysis), so that approach was deliberately NOT kept. Padding is
    //    handled downstream instead via res.mask1/res.mask2 (see below),
    //    which matching.cpp uses to keep the matcher from treating the
    //    black padding as real, flat texture.
    BBox b1 = warpedBBox(H1, W, H);
    BBox b2 = warpedBBox(H2, W, H);
    double ymin = std::min(b1.ymin, b2.ymin), ymax = std::max(b1.ymax, b2.ymax);
    double sy = H / (ymax - ymin);
    double sx = W / std::max(b1.xmax - b1.xmin, b2.xmax - b2.xmin);
    if (preserveAspectRatio) {
        // Share a single scale between x and y so the canvas fit cannot
        // distort geometry (stretch content differently per axis); only the
        // translations remain independent per image/axis.
        double s = std::min(sx, sy);
        sx = s;
        sy = s;
    }
    double ty = -ymin * sy;
    double tx1 = -b1.xmin * sx;
    double tx2 = -b2.xmin * sx;

    Eigen::Matrix3d S1; S1 << sx, 0, tx1, 0, sy, ty, 0, 0, 1;
    Eigen::Matrix3d S2; S2 << sx, 0, tx2, 0, sy, ty, 0, 0, 1;
    Eigen::Matrix3d H1_final = S1 * H1;
    Eigen::Matrix3d H2_final = S2 * H2;

    // 2. Convert back to OpenCV and warp the images
    res.H1 = eigenToCv(H1_final);
    res.H2 = eigenToCv(H2_final);

    cv::Size sz(W, H);
    cv::warpPerspective(left, res.left_rect, res.H1, sz, cv::INTER_LINEAR);
    cv::warpPerspective(right, res.right_rect, res.H2, sz, cv::INTER_LINEAR);

    // 2b. Validity masks: warped-quad content region rasterized to a
    // per-pixel mask so matching.cpp can invalidate disparity computed over
    // black padding instead of trying to eliminate the padding via cropping.
    res.mask1 = rasterizeQuadMask(H1_final, W, H);
    res.mask2 = rasterizeQuadMask(H2_final, W, H);

    // 3. Rectified projection matrices: for a planar image-space homography
    //    H applied to an image with original projection matrix P_orig, the
    //    projection matrix valid for the warped image is simply H * P_orig.
    //    This is exact regardless of how distorted H1/H2 are, unlike a
    //    calibrated-style Q matrix (which assumes a shared, pure-rotation
    //    rectified camera model that a general projective warp doesn't have).
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

    // Points triangulated via P1 = H1*[K0|0] land directly in the original,
    // unrotated left-camera frame (H1 only warps the 2D image — it implies
    // no actual 3D camera rotation), so no extra un-rotation is needed here.
    res.R0_rect = cv::Mat::eye(3, 3, CV_64F);

    // res.Q intentionally left empty: there is no valid single-focal-length
    // Q matrix for a general projective warp. Downstream code must triangulate
    // via res.P1/res.P2 (see disparityToCloudViaP in depth.cpp) instead of Q.

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Loop-Zhang closed-form construction (rectifyLoopZhang helpers)
//
// Faithful to Loop & Zhang, "Computing Rectifying Homographies for Stereo
// Vision" (CVPR 1999): a distortion-minimizing projective step that couples
// the left/right epipole warps through F, a closed-form similarity step
// derived directly from F's entries, and a closed-form shear step applied
// symmetrically to both images.
// ─────────────────────────────────────────────────────────────────────────────

// Bottom row of the left projective transform H_p: w = [e]_x * z. H_p sends
// the left epipole e to infinity along the shared direction z. Normalizing
// so the bottom row ends in 1 is required, not optional: Hp's top two rows
// are fixed to [1,0,0]/[0,1,0] (not free to rescale), and buildHsimPair's
// closed-form formulas below use w.x()/w.y() assuming exactly this
// normalization. The 1e-9 floor guards the case where z happens to be
// (near-)parallel to the direction that sends wz to zero.
static Eigen::Matrix3d buildHp(const Eigen::Vector3d& e, const Eigen::Vector3d& z) {
    Eigen::Vector3d w = skewSymmetric(e) * z;
    double wz = w.z();
    if (std::abs(wz) < 1e-9) wz = 1e-9; // last-resort guard, see comment above
    w /= wz;

    Eigen::Matrix3d Hp;
    Hp << 1, 0, 0,
          0, 1, 0,
          w.x(), w.y(), 1;
    return Hp;
}

// Bottom row of the right projective transform H'_p: w' = F * z. Coupled to
// buildHp through the same z and F, so H_p/H'_p stay in epipolar
// correspondence. Same normalization requirement and last-resort guard as
// buildHp above.
static Eigen::Matrix3d buildHpPrime(const Eigen::Matrix3d& F, const Eigen::Vector3d& z) {
    Eigen::Vector3d wp = F * z;
    double wz = wp.z();
    if (std::abs(wz) < 1e-9) wz = 1e-9;
    wp /= wz;

    Eigen::Matrix3d Hpp;
    Hpp << 1, 0, 0,
           0, 1, 0,
           wp.x(), wp.y(), 1;
    return Hpp;
}

// Closed-form pixel-spread second-moment block for a W x H image: the
// covariance of pixel coordinates about the image center, i.e. sum over all
// pixels of (x-x_bar)(y-y_bar). Legitimately zero off-diagonal (x,y indices
// are independent/separable) and legitimately zero in row/col 2 when
// embedded (variance of a constant is 0, covariance of a coordinate with a
// constant is 0) -- unlike pcpcT3x3 below, this one really is a 2x2 fact.
static Eigen::Matrix2d ppT2x2(double W, double H) {
    Eigen::Matrix2d PPt;
    PPt << (W * H / 12.0) * (W * W - 1), 0,
           0,                             (W * H / 12.0) * (H * H - 1);
    return PPt;
}

// Full pc*pc^T for the homogeneous image-center point pc=(W/2,H/2,1). Unlike
// ppT2x2 (a covariance, genuinely zero in row/col 2), this is a literal
// point's outer product: w^T*pc = w1*(W/2) + w2*(H/2) + w3, so the row/col-2
// cross terms and the trailing 1 are real, non-negligible contributions to
// the quadratic form and must not be dropped.
static Eigen::Matrix3d pcpcT3x3(double W, double H) {
    Eigen::Vector3d pc(0.5 * W, 0.5 * H, 1.0);
    return pc * pc.transpose();
}

// Embed a 2x2 block into the top-left of a 3x3 matrix (zero row/col for the
// homogeneous coordinate) -- only valid for covariance-like blocks (see
// ppT2x2) where that row/col is genuinely zero, not for point outer products
// (see pcpcT3x3, which builds its own full 3x3 instead of using this).
static Eigen::Matrix3d embed2x2(const Eigen::Matrix2d& block) {
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    M.block<2, 2>(0, 0) = block;
    return M;
}

struct DistortionMats { Eigen::Matrix3d A, B, Ap, Bp; };

// Build the four distortion matrices used by the DT(z) objective. A,B come
// from the left epipole/image size; A',B' come from F conjugating the same
// closed-form blocks, since w' = F*z couples the right transform to z
// through F directly (see buildHpPrime) rather than through the right
// epipole's cross matrix. Independent of z, so this is computed once per
// rectifyLoopZhang call and reused by every sample of the search.
static DistortionMats obtainAB(const Eigen::Vector3d& e1, const Eigen::Matrix3d& F,
                                int W, int H) {
    Eigen::Matrix3d PPt   = embed2x2(ppT2x2((double)W, (double)H));
    Eigen::Matrix3d PcPct = pcpcT3x3((double)W, (double)H);
    Eigen::Matrix3d ex    = skewSymmetric(e1);

    DistortionMats m;
    m.A  = ex.transpose() * PPt   * ex;
    m.B  = ex.transpose() * PcPct * ex;
    m.Ap = F.transpose()  * PPt   * F;
    m.Bp = F.transpose()  * PcPct * F;
    return m;
}

// DT(phi) = (z^T A z)/(z^T B z) + (z^T A' z)/(z^T B' z), z = (cos phi, sin phi, 0).
// z is a direction at infinity (scale-invariant), so a single angle fully
// parameterizes the search space -- this also sidesteps inverting B/B',
// which are rank-1 outer products and thus ill-posed for a generalized
// eigenvalue formulation.
static double distortionObjective(double phi, const DistortionMats& m) {
    Eigen::Vector3d z(std::cos(phi), std::sin(phi), 0.0);
    double den1 = z.dot(m.B  * z);
    double den2 = z.dot(m.Bp * z);
    if (std::abs(den1) < 1e-9 || std::abs(den2) < 1e-9)
        return std::numeric_limits<double>::infinity();
    double num1 = z.dot(m.A  * z);
    double num2 = z.dot(m.Ap * z);
    return num1 / den1 + num2 / den2;
}

// Global minimum of Loop & Zhang's DT(phi) over phi in [0, pi): a coarse
// grid search (the objective can have multiple local minima) followed by a
// golden-section refine around the best sample.
static double findOptimalPhi(const DistortionMats& m) {
    auto objective = [&](double phi) {
        return distortionObjective(phi, m);
    };

    constexpr int kGridSamples = 3600; // 0.05 deg resolution
    double bestPhi = 0.0, bestCost = std::numeric_limits<double>::infinity();
    for (int i = 0; i < kGridSamples; ++i) {
        double phi = M_PI * (double)i / (double)kGridSamples;
        double cost = objective(phi);
        if (cost < bestCost) { bestCost = cost; bestPhi = phi; }
    }

    constexpr double kGoldenRatio = 0.6180339887498949;
    double step = M_PI / kGridSamples;
    double lo = bestPhi - step, hi = bestPhi + step;
    double x1 = hi - kGoldenRatio * (hi - lo);
    double x2 = lo + kGoldenRatio * (hi - lo);
    double f1 = objective(x1);
    double f2 = objective(x2);
    for (int it = 0; it < 60 && (hi - lo) > 1e-10; ++it) {
        if (f1 < f2) {
            hi = x2; x2 = x1; f2 = f1;
            x1 = hi - kGoldenRatio * (hi - lo);
            f1 = objective(x1);
        } else {
            lo = x1; x1 = x2; f1 = f2;
            x2 = lo + kGoldenRatio * (hi - lo);
            f2 = objective(x2);
        }
    }
    return 0.5 * (lo + hi);
}

// Closed-form similarity components H_sim, H_sim' from F's entries and the
// normalized bottom rows w, w' of H_p, H_p' (Loop & Zhang, paper Eqs.
// 99-108). This codebase's F is 0-indexed (row,col) (calib.F.at<double>(row,col)),
// matching the paper's own indexing -- formulas transcribe with F(row,col)
// directly, no transpose.
// Diagonal terms use w.y()/wp.y(), off-diagonal terms use w.x()/wp.x().
//
// Derivation note: solving the rectifying epipolar constraint
// HpreR^T [i]_x HpreL = k*F (with HpreL = Hsim*Hp, HpreR = Hsim'*Hp', and
// Hsim/Hsim' constrained to the [[a,b,0],[-b,a,c],[0,0,1]] similarity form)
// for k=1 gives a,b uniquely from F,w and ap,bp uniquely from F,wp -- the
// right-image (primed) terms are NOT the same sign pattern as the left, so
// they must be derived independently rather than assumed symmetric with
// w<->wp substituted in.
//
// The first row of H_sim/H_sim' is only constrained (by the paper's
// derivation) up to a +/-90-degree rotation -- orthogonality to the second
// row has two solutions, (vb,-va) and (-vb,va). The formula below fixes one
// branch for each image independently, which is not guaranteed to pick the
// same handedness for both; the caller (rectifyLoopZhang) checks
// linearDet2x2 on the composed H_pre for both images and flips the right
// image's first row to the other branch if they disagree in orientation.
// A row-alignment residual check alone cannot catch this: |Δy| is blind to
// a left-right mirror in x.
static std::pair<Eigen::Matrix3d, Eigen::Matrix3d> buildHsimPair(
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

// Closed-form solve for vc' (the one free scalar left in H_sim/H_sim'): with
// vc'=0, H_sim/H_sim' are pure affine similarities (bottom row [0,0,1]), so
// composed with the already-projective H_p/H_p' the warped v-coordinate of
// every corner is exactly affine (additive) in vc' -- no search needed, just
// a min over the 8 corner v-values (4 per image, vc'=0 baseline already
// includes the per-image F(2,2) asymmetry from buildHsimPair) so the
// tightest common vertical bound starts at 0.
static double solveVcPrime(const Eigen::Matrix3d& Hp, const Eigen::Matrix3d& HpP,
                            const Eigen::Matrix3d& F,
                            const Eigen::Vector3d& w, const Eigen::Vector3d& wp,
                            int W, int H) {
    auto [Hsim0, HsimP0] = buildHsimPair(F, w, wp, 0.0);
    Eigen::Matrix3d preL = Hsim0  * Hp;
    Eigen::Matrix3d preR = HsimP0 * HpP;

    auto quadL = warpedQuad(preL, W, H);
    auto quadR = warpedQuad(preR, W, H);

    double minV = std::numeric_limits<double>::infinity();
    for (const auto& p : quadL) minV = std::min(minV, p.y());
    for (const auto& p : quadR) minV = std::min(minV, p.y());

    return -minV;
}

// Closed-form shear correction from edge-midpoint transforms (Loop & Zhang
// §2.3). Called once per image, each with that image's own H_sim*H_p (or
// H_sim'*H_p') composite, applied symmetrically to both.
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
    // denom = M*N*(xv*yu - xu*yv); sa/sb share this denom with opposite sign,
    // per Loop & Zhang Eqs. 124-125. Note: sa pairs N^2 (height^2) with
    // xv^2 and M^2 (width^2) with yv^2 -- NOT M^2 with xv^2 -- this was a
    // transcription bug in an earlier version of this function; verified
    // against an independent reference implementation of the same formula.
    double denom = M * N * (xv * yu - xu * yv);
    double sa = (N * N * xv * xv + M * M * yv * yv) / denom;
    double sb = (N * N * xu * xv + M * M * yu * yv) / -denom;

    // The closed-form solve above has two algebraic roots -- (sa,sb) and
    // (-sa,-sb) -- differing by a reflection. Only the positive-sa root is
    // a pure shear+scale; a negative sa means this branch would flip the
    // whole image (visible downstream as a canvas tilt near 180 deg and a
    // negative-determinant warning). Force the shear-preserving branch,
    // matching the convention used by other Loop-Zhang implementations of
    // this same closed form.
    if (sa < 0.0) {
        sa = -sa;
        sb = -sb;
    }

    Eigen::Matrix3d Hsh = Eigen::Matrix3d::Identity();
    Hsh(0, 0) = sa; Hsh(0, 1) = sb;
    return Hsh;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. The Loop-Zhang Method (faithful closed-form construction)
// ─────────────────────────────────────────────────────────────────────────────

RectifyResult rectifyLoopZhang(const cv::Mat& left, const cv::Mat& right,
    const CalibData& calib) {
    int W = left.cols;
    int H = left.rows;

    // Convert OpenCV Fundamental Matrix to Eigen
    Eigen::Matrix3d F;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            F(i, j) = calib.F.at<double>(i, j);

    // 1. Left epipole. (The right epipole is never needed explicitly here --
    //    A'/B' and H'_p are built from F directly, see obtainAB/buildHpPrime.)
    Eigen::Vector3d e1 = computeEpipole(F);

    // 2. Distortion-minimizing shared direction z, found via grid +
    //    golden-section search over the angle phi (z=(cos phi, sin phi, 0)).
    DistortionMats dm = obtainAB(e1, F, W, H);
    double phi = findOptimalPhi(dm);
    Eigen::Vector3d z(std::cos(phi), std::sin(phi), 0.0);

    // 3. Projective transforms, coupled through F.
    Eigen::Matrix3d Hp  = buildHp(e1, z);
    Eigen::Matrix3d HpP = buildHpPrime(F, z);
    Eigen::Vector3d w  = Hp.row(2).transpose();
    Eigen::Vector3d wp = HpP.row(2).transpose();

    // 4. Similarity components: solve the one free scalar vc' in closed
    //    form, then build the finalized H_sim/H_sim'.
    double vcPrime = solveVcPrime(Hp, HpP, F, w, wp, W, H);
    auto [Hsim, HsimP] = buildHsimPair(F, w, wp, vcPrime);

    // 5. Orientation check: H_sim's first row is only fixed up to a
    //    +/-90-degree rotation (see buildHsimPair), and nothing guarantees
    //    the same branch is correct for both images. If the two composed
    //    partials disagree in handedness (opposite-sign determinant), flip
    //    H_sim' to the other branch -- a row-alignment residual check alone
    //    can't catch this, since |Δy| is blind to a left-right mirror in x.
    Eigen::Matrix3d HpreL = Hsim  * Hp;
    Eigen::Matrix3d HpreR = HsimP * HpP;
    if (linearDet2x2(HpreL) * linearDet2x2(HpreR) < 0.0) {
        HsimP(0, 0) *= -1.0;
        HsimP(0, 1) *= -1.0;
        HpreR = HsimP * HpP;
    }

    // 6. Shear, built independently for both images.
    Eigen::Matrix3d Hsh  = buildShear(HpreL, W, H);
    Eigen::Matrix3d HshP = buildShear(HpreR, W, H);

    // 7. Compose final pre-canvas-fit homographies.
    Eigen::Matrix3d H1 = Hsh  * HpreL;
    Eigen::Matrix3d H2 = HshP * HpreR;

    return finalizeRectification(H1, H2, left, right, calib, W, H);
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone executable
// ─────────────────────────────────────────────────────────────────────────────
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
    if (!calib.hasIntrinsics() || !calib.hasRelativePose()) {
        std::cout << "[rectification]: Missing intrinsics or relative pose — run sparse_matching first.\n";
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