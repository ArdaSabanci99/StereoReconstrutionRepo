#include "depth.h"
#include "DataLoader.h"
#include "Eigen.h"
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Disparity → depth map  Z = f * B / (d)
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat disparityToDepth(const cv::Mat& disparity, const CalibData& calib) {
    double f = calib.K0.at<double>(0, 0);
    cv::Mat depth(disparity.size(), CV_32F, 0.0f);
    for (int y = 0; y < disparity.rows; ++y)
        for (int x = 0; x < disparity.cols; ++x) {
            float d = disparity.at<float>(y, x);
            if (d <= 0) continue;
            depth.at<float>(y, x) = (float)(f * calib.baseline / d);
        }
    return depth;
}

// ─────────────────────────────────────────────────────────────────────────────
// Disparity → Point Cloud via Q matrix (standard after rectification)
// Q = [1 0 0 -cx; 0 1 0 -cy; 0 0 0 f; 0 0 -1/Tx 0]
//
// NOTE: Q is only a valid camera model for a calibrated, pure-rotation
// rectification (shared single focal length, e.g. OpenCV's stereoRectify
// output). For a general projective rectification (Loop-Zhang) there is no
// valid single Q -- use disparityToCloudViaP below instead. Callers should
// check res.Q.empty() (or the persisted Q in proj_*.yaml) to pick the right
// function; see main() below.
// ─────────────────────────────────────────────────────────────────────────────
PointCloud disparityToCloud(const cv::Mat& disp, const cv::Mat& Q,
                             const cv::Mat& color_img,
                             float min_disp, float max_depth) {
    // Manual implementation (avoids cv::reprojectImageTo3D dependency)
    
    // Extract the specific coefficients from the Q matrix.
    // These constants represent the camera focal length (f), principal center (cx, cy),
    // and the stereo baseline (Tx), which define the mapping from 2D to 3D.
    double q03 = Q.at<double>(0,3);   // -cx
    double q13 = Q.at<double>(1,3);   // -cy
    double q23 = Q.at<double>(2,3);   // f
    double q32 = Q.at<double>(3,2);   // -1/Tx
    double q33 = Q.at<double>(3,3);   // 0

    bool has_color = !color_img.empty();
    PointCloud cloud;

    // Iterate through every pixel in the disparity map.
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x) {
            float d = disp.at<float>(y, x);

            // Reject invalid points (where disparity is too low or invalid).
            if (d < min_disp) continue;

            
            // Calculate the homogeneous coordinate denominator (W).
            double W = q32 * d + q33;
            if (std::abs(W) < 1e-9) continue;

            // Triangulate: Convert (x, y, disparity) to (X, Y, Z) in 3D camera space.
            float X = (float)((x + q03) / W);
            float Y = (float)((y + q13) / W);
            float Z = (float)(q23 / W);

            // Validity check based on physical depth constraints.
            if (Z <= 0 || Z > max_depth) continue;

            // Store the computed 3D point in the cloud.
            cloud.points.emplace_back(X, Y, Z);

            // Assign color data if available, otherwise default to neutral gray.
            if (has_color) {
                cv::Vec3b c = color_img.at<cv::Vec3b>(y, x);
                cloud.colors.emplace_back(c[2], c[1], c[0], 255);  // BGR→RGBA
            } else {
                cloud.colors.emplace_back(200, 200, 200, 255);
            }
        }

    std::cout << "[depth] disparityToCloud → " << cloud.points.size() << " points\n";
    return cloud;
}

// ─────────────────────────────────────────────────────────────────────────────
// DLT triangulation for a single point pair
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat triangulatePointDLT(const cv::Point2f& pt1, const cv::Point2f& pt2,
                              const cv::Mat& P1, const cv::Mat& P2) {
    // Build 4×4: each point contributes 2 rows
    cv::Mat A(4, 4, CV_64F);
    for (int j = 0; j < 4; ++j) {
        A.at<double>(0,j) = pt1.x * P1.at<double>(2,j) - P1.at<double>(0,j);
        A.at<double>(1,j) = pt1.y * P1.at<double>(2,j) - P1.at<double>(1,j);
        A.at<double>(2,j) = pt2.x * P2.at<double>(2,j) - P2.at<double>(0,j);
        A.at<double>(3,j) = pt2.y * P2.at<double>(2,j) - P2.at<double>(1,j);
    }
    cv::Mat U, S, Vt;
    cv::SVD::compute(A, S, U, Vt, cv::SVD::FULL_UV);
    cv::Mat X = Vt.row(3).t();   // homogeneous
    return X.rowRange(0,3) / X.at<double>(3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Disparity → Point Cloud via P1/P2 (general projective rectification path)
//
// Used instead of disparityToCloud/Q whenever the rectifier was a general
// projective warp (Loop-Zhang) rather than a calibrated pure-rotation
// rectification -- see the note on disparityToCloud above and the comment
// in rectification.cpp's finalizeRectification explaining why Q is left
// empty in that case. Reuses triangulatePointDLT: for a disparity pixel at
// (x,y) with value d, the corresponding right-image point is (x-d, y)
// (rectification guarantees the same row, d is the horizontal offset).
// This is O(1) DLT solves per pixel rather than Q's O(1) divide, so it's
// slower, but it's the only correct option for a non-calibrated rectifying
// homography pair.
// ─────────────────────────────────────────────────────────────────────────────
PointCloud disparityToCloudViaP(const cv::Mat& disp, const cv::Mat& P1, const cv::Mat& P2,
                                 const cv::Mat& color_img,
                                 float min_disp, float max_depth,
                                 const cv::Mat& mask1, const cv::Mat& mask2) {
    bool has_color = !color_img.empty();
    bool has_mask1 = !mask1.empty();
    bool has_mask2 = !mask2.empty();
    PointCloud cloud;

    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x) {
            float d = disp.at<float>(y, x);
            if (d < min_disp) continue;

            // Defensive re-check against the validity masks: matching.cpp
            // should already have invalidated disparity computed over black
            // warp padding (see the padding/SGM-noise discussion), but this
            // is cheap insurance against stale disparity maps computed
            // before that fix, or any other source that ignored the masks.
            if (has_mask1 && mask1.at<uchar>(y, x) == 0) continue;

            int xr = cvRound(x - d);
            if (xr < 0 || xr >= disp.cols) continue;
            if (has_mask2 && mask2.at<uchar>(y, xr) == 0) continue;

            cv::Point2f pl((float)x, (float)y);
            cv::Point2f pr((float)(x - d), (float)y);
            cv::Mat X = triangulatePointDLT(pl, pr, P1, P2);

            float Z = (float)X.at<double>(2);
            if (Z <= 0 || Z > max_depth) continue;

            cloud.points.emplace_back((float)X.at<double>(0),
                                       (float)X.at<double>(1), Z);

            if (has_color) {
                cv::Vec3b c = color_img.at<cv::Vec3b>(y, x);
                cloud.colors.emplace_back(c[2], c[1], c[0], 255);  // BGR→RGBA
            } else {
                cloud.colors.emplace_back(200, 200, 200, 255);
            }
        }

    std::cout << "[depth] disparityToCloudViaP → " << cloud.points.size() << " points\n";
    return cloud;
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangulate a batch of correspondences
// ─────────────────────────────────────────────────────────────────────────────
PointCloud triangulateCloud(const std::vector<cv::Point2f>& pts1,
                              const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& P1, const cv::Mat& P2,
                              const std::vector<cv::Vec3b>& colors) {
    PointCloud cloud;
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat X = triangulatePointDLT(pts1[i], pts2[i], P1, P2);
        float Z = (float)X.at<double>(2);
        if (Z <= 0 || Z > 1e4f) continue;
        cloud.points.emplace_back((float)X.at<double>(0),
                                   (float)X.at<double>(1), Z);
        if (!colors.empty() && i < colors.size())
            cloud.colors.emplace_back(colors[i][2], colors[i][1], colors[i][0], 255);
        else
            cloud.colors.emplace_back(200, 200, 200, 255);
    }
    std::cout << "[depth] DLT triangulated " << cloud.points.size() << " / "
              << pts1.size() << " points\n";
    return cloud;
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal estimation via PCA on k-nearest neighbours
// (simple O(N²) for moderate clouds; replace with FLANN for large clouds)
// ─────────────────────────────────────────────────────────────────────────────
void estimateNormals(PointCloud& cloud, int k) {
    int N = (int)cloud.points.size();
    cloud.normals.resize(N, Eigen::Vector3f(0, 0, 1));
    if (N < k) return;

    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3f& pi = cloud.points[i];

        // Find k nearest neighbours (brute-force)
        std::vector<std::pair<float,int>> dists(N);
        for (int j = 0; j < N; ++j)
            dists[j] = { (cloud.points[j] - pi).squaredNorm(), j };
        std::partial_sort(dists.begin(), dists.begin()+k, dists.end());

        // Covariance matrix of neighbourhood
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (int ki = 0; ki < k; ++ki) mean += cloud.points[dists[ki].second];
        mean /= k;

        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (int ki = 0; ki < k; ++ki) {
            Eigen::Vector3f d = cloud.points[dists[ki].second] - mean;
            cov += d * d.transpose();
        }

        // Smallest eigenvector = normal
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        Eigen::Vector3f n = solver.eigenvectors().col(0);

        // Orient towards origin (camera)
        if (n.dot(-pi) < 0) n = -n;
        cloud.normals[i] = n;
    }
    std::cout << "[depth] Estimated " << N << " normals.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// PLY export (coloured)
// ─────────────────────────────────────────────────────────────────────────────
void savePointCloud(const PointCloud& cloud, const std::string& path) {
    std::ofstream f(path);
    bool has_color  = !cloud.colors.empty();
    bool has_normal = !cloud.normals.empty();

    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << cloud.points.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n";
    if (has_normal)
        f << "property float nx\nproperty float ny\nproperty float nz\n";
    if (has_color)
        f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "end_header\n";

    for (size_t i = 0; i < cloud.points.size(); ++i) {
        const auto& p = cloud.points[i];
        f << p.x() << " " << p.y() << " " << p.z();
        if (has_normal) {
            const auto& n = cloud.normals[i];
            f << " " << n.x() << " " << n.y() << " " << n.z();
        }
        if (has_color) {
            const auto& c = cloud.colors[i];
            f << " " << (int)c[0] << " " << (int)c[1] << " " << (int)c[2];
        }
        f << "\n";
    }
    std::cout << "[depth] Saved PLY → " << path
              << "  (" << cloud.points.size() << " pts)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// COFF export (MeshLab-compatible)
// ─────────────────────────────────────────────────────────────────────────────
void savePointCloudOFF(const PointCloud& cloud, const std::string& path) {
    std::ofstream f(path);
    bool has_color = !cloud.colors.empty();
    f << (has_color ? "COFF" : "OFF") << "\n"
      << cloud.points.size() << " 0 0\n";
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        const auto& p = cloud.points[i];
        f << p.x() << " " << p.y() << " " << p.z();
        if (has_color) {
            const auto& c = cloud.colors[i];
            f << "  " << (int)c[0] << " " << (int)c[1] << " " << (int)c[2] << " 255";
        }
        f << "\n";
    }
    std::cout << "[depth] Saved COFF → " << path << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone
// ─────────────────────────────────────────────────────────────────────────────
#if !defined(PIPELINE_BUILD) && !defined(BUILDING_ICP) && !defined(BUILDING_MESH)
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: depth <data_path> <scene_id> <left_id> <right_id> [method] [rect_tag]\n"
                     "  rect_tag: opencv | loopzhang | loopzhang_correct (default: opencv)\n";
        return 1;
    }
    fs::path dataPath(argv[1]);
    const std::string sceneId(argv[2]);
    const std::string viewL = padViewId(std::stoi(argv[3]));
    const std::string viewR = padViewId(std::stoi(argv[4]));
    std::string method = (argc >= 6) ? argv[5] : "sgm";
    std::string rectTag = (argc >= 7) ? argv[6] : "opencv";

    std::string match_path = "results/scene" + sceneId + "/matching";
    cv::Mat disp = loadDisparity(match_path + "/view_" + viewL + "_" + viewR + "_" + method + "_raw.png");
    cv::Mat color = cv::imread(match_path + "/view_" + viewL + "_" + viewR + "_" + method + ".png");

    std::string calib_path = "results/scene" + sceneId + "/sparse_matching/calib_" + viewL + "_" + viewR + ".yaml";
    CalibData calib = loadCalibData(calib_path);
    if (!calib.hasIntrinsics())
        throw std::runtime_error("[depth] Calib incomplete — run sparse_matching first: " + calib_path);

    // ── Load the rectifier's actual P1/P2/Q (and masks) instead of
    // hand-building a Q from the ORIGINAL, unrectified calib.K0 intrinsics.
    // That reconstruction is only valid for a calibrated pure-rotation
    // rectification and silently produced wrong 3D geometry for the
    // Loop-Zhang (projective) path, since it ignored the rectifying
    // homographies entirely. rectification.cpp's main() now persists this
    // file for exactly this reason.
    std::string rectPath = "results/scene" + sceneId + "/rectification/" + rectTag;
    std::string projFile = rectPath + "/proj_" + viewL + "_" + viewR + ".yaml";

    cv::Mat Q, P1, P2;
    {
        cv::FileStorage fsIn(projFile, cv::FileStorage::READ);
        if (!fsIn.isOpened())
            throw std::runtime_error("[depth] Missing rectification output — run rectification "
                                      "first (tag=" + rectTag + "): " + projFile);
        fsIn["Q"]  >> Q;
        fsIn["P1"] >> P1;
        fsIn["P2"] >> P2;
        fsIn.release();
    }

    cv::Mat mask1 = cv::imread(rectPath + "/mask_" + viewL + ".png", cv::IMREAD_GRAYSCALE);
    cv::Mat mask2 = cv::imread(rectPath + "/mask_" + viewR + ".png", cv::IMREAD_GRAYSCALE);

    PointCloud cloud;
    if (!Q.empty()) {
        // Calibrated pure-rotation rectification (OpenCV path): Q gives an
        // exact, cheap closed-form triangulation.
        std::cout << "[depth] using Q-based triangulation (tag=" << rectTag << ")\n";
        cloud = disparityToCloud(disp, Q, color);
    } else {
        // General projective rectification (Loop-Zhang): no valid single Q,
        // triangulate per-pixel via P1/P2 instead.
        if (P1.empty() || P2.empty())
            throw std::runtime_error("[depth] Neither Q nor P1/P2 available in " + projFile);
        std::cout << "[depth] Q empty, using P1/P2 DLT triangulation (tag=" << rectTag << ")\n";
        cloud = disparityToCloudViaP(disp, P1, P2, color, 1.0f, 1e4f, mask1, mask2);
    }

    std::string save = "results/scene" + sceneId + "/pointcloud";
    fs::create_directories(save);
    savePointCloud(cloud, save + "/views_" + viewL + "_" + viewR + ".ply");
    savePointCloudOFF(cloud, save + "/views_" + viewL + "_" + viewR + ".off");
    return 0;
}
#endif