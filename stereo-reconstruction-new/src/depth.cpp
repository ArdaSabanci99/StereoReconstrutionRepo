#include "depth.h"
#include "DataLoader.h"
#include "Eigen.h"
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Disparity → depth map  Z = f * B / (d + doffs)
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat disparityToDepth(const cv::Mat& disparity, const CalibData& calib) {
    double f = calib.K0.at<double>(0, 0);
    cv::Mat depth(disparity.size(), CV_32F, 0.0f);
    for (int y = 0; y < disparity.rows; ++y)
        for (int x = 0; x < disparity.cols; ++x) {
            float d = disparity.at<float>(y, x);
            if (d <= 0) continue;
            depth.at<float>(y, x) = (float)(f * calib.baseline / (d + calib.doffs));
        }
    return depth;
}

// ─────────────────────────────────────────────────────────────────────────────
// Disparity → Point Cloud via Q matrix (standard after rectification)
// Q = [1 0 0 -cx; 0 1 0 -cy; 0 0 0 f; 0 0 -1/Tx 0]
// ─────────────────────────────────────────────────────────────────────────────
PointCloud disparityToCloud(const cv::Mat& disp, const cv::Mat& Q,
                             const cv::Mat& color_img,
                             float min_disp, float max_depth) {
    // Manual implementation (avoids cv::reprojectImageTo3D dependency)
    double q03 = Q.at<double>(0,3);   // -cx
    double q13 = Q.at<double>(1,3);   // -cy
    double q23 = Q.at<double>(2,3);   // f
    double q32 = Q.at<double>(3,2);   // -1/Tx
    double q33 = Q.at<double>(3,3);   // 0

    bool has_color = !color_img.empty();
    PointCloud cloud;

    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x) {
            float d = disp.at<float>(y, x);
            if (d < min_disp) continue;

            double W  = q32 * d + q33;
            if (std::abs(W) < 1e-9) continue;

            float X = (float)((x + q03) / W);
            float Y = (float)((y + q13) / W);
            float Z = (float)(q23 / W);

            if (Z <= 0 || Z > max_depth) continue;

            cloud.points.emplace_back(X, Y, Z);

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
        std::cerr << "Usage: depth <data_path> <scene_id> <left_id> <right_id> [method]\n";
        return 1;
    }
    fs::path dataPath(argv[1]);
    const std::string sceneId(argv[2]);
    const std::string viewL = padViewId(std::stoi(argv[3]));
    const std::string viewR = padViewId(std::stoi(argv[4]));
    std::string method = (argc >= 6) ? argv[5] : "sgm";

    std::string match_path = "results/scene" + sceneId + "/matching";
    cv::Mat disp = loadDisparity(match_path + "/view_" + viewL + "_" + viewR + "_" + method + "_raw.png");
    cv::Mat color = cv::imread(match_path + "/view_" + viewL + "_" + viewR + "_" + method + ".png");

   
    std::string calib_path = "results/scene" + sceneId + "/sparse_matching/calib_" + viewL + "_" + viewR + ".yaml";
    CalibData calib = loadCalibData(calib_path);
    if (!calib.hasIntrinsics())
        throw std::runtime_error("[depth] Calib incomplete — run sparse_matching first: " + calib_path);

    // Load left correct extrinsics
    std::string effLeftId = calib.swapped ? viewR : viewL;
    std::string effRightId = calib.swapped ? viewL : viewR;

    DTUDataLoader loader(dataPath.string());
    CalibData orig_calib;
    loader.loadLeftExtrinsics(orig_calib, effLeftId, effRightId);
    calib.R0 = orig_calib.R0;
    calib.t0 = orig_calib.t0;

     // Load Q from saved rectification (simplified: try to get from calibration)
    
    double f  = calib.K0.at<double>(0,0);
    double cx = calib.K0.at<double>(0,2), cy = calib.K0.at<double>(1,2);
    double B  = cv::norm(calib.T_rel);
    // Q: Tx = -B (right camera is to the right, so Tx < 0)
    // q32 = -1/Tx = 1/B → Z = f/W = f*B/d > 0 for d > 0.
    cv::Mat Q = (cv::Mat_<double>(4,4)
        << 1, 0, 0, -cx,
           0, 1, 0, -cy,
           0, 0, 0,   f,
           0, 0, 1/B,  0);

    PointCloud cloud = disparityToCloud(disp, Q, color);

    std::string save = "results/scene" + sceneId + "/pointcloud";
    fs::create_directories(save);
    savePointCloud(cloud, save + "/views_" + viewL + "_" + viewR + ".ply");
    savePointCloudOFF(cloud, save + "/views_" + viewL + "_" + viewR + ".off");
    return 0;
}
#endif
