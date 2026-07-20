#include "depth.h"
#include "DataLoader.h"
#include "Eigen.h"
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;


// Disparity → depth map  Z = f * B / (d)
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

PointCloud disparityToCloud(const cv::Mat& disp, const cv::Mat& Q,
                             const cv::Mat& color_img,
                             float min_disp, float max_depth) {
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

PointCloud disparityToCloudViaP(const cv::Mat& disp, const cv::Mat& P1, const cv::Mat& P2,
                                 const cv::Mat& color_img,
                                 float min_disp, float max_depth,
                                 const cv::Mat& mask1, const cv::Mat& mask2) {
    bool has_color = !color_img.empty();
    bool has_mask1 = !mask1.empty();
    bool has_mask2 = !mask2.empty();
    PointCloud cloud;

    Eigen::Matrix<double, 3, 4> P_left, P_right;
    cv::cv2eigen(P1, P_left);
    cv::cv2eigen(P2, P_right);

    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x) {
            float d = disp.at<float>(y, x);
            if (d < min_disp) continue;

            if (has_mask1 && mask1.at<uchar>(y, x) == 0) continue;

            int xr = cvRound(x - d);
            if (xr < 0 || xr >= disp.cols) continue;
            if (has_mask2 && mask2.at<uchar>(y, xr) == 0) continue;

            Eigen::Vector3d pt_left(x, y, 1.0);
            Eigen::Vector3d pt_right(x - d, y, 1.0);
            Eigen::Vector3d X = triangulatePoint(pt_left, pt_right, P_left, P_right);

            float Z = (float)X.z();
            if (Z <= 0 || Z > max_depth) continue;

            cloud.points.emplace_back((float)X.x(), (float)X.y(), Z);

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
        std::cout << "[depth] using Q-based triangulation (tag=" << rectTag << ")\n";
        cloud = disparityToCloud(disp, Q, color);
    } else {
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