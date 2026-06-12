#include "depth.h"
#include "DataLoader.h"
#include <fstream>
#include <iostream>

// ── Disparity → Depth (DONE) ──────────────────────────────────────────────
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

// ── Depth → Point Cloud (DONE) ────────────────────────────────────────────
PointCloud depthToPointCloud(const cv::Mat& depth, const CalibData& calib,
                              const cv::Mat& color_img) {
    double fx = calib.K0.at<double>(0, 0);
    double fy = calib.K0.at<double>(1, 1);
    double cx = calib.K0.at<double>(0, 2);
    double cy = calib.K0.at<double>(1, 2);

    bool has_color = !color_img.empty();
    PointCloud cloud;

    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            float Z = depth.at<float>(y, x);
            if (Z <= 0 || Z > 1e4f) continue;

            float X = (float)((x - cx) * Z / fx);
            float Y = (float)((y - cy) * Z / fy);
            cloud.points.emplace_back(X, Y, Z);

            if (has_color) {
                cv::Vec3b c = color_img.at<cv::Vec3b>(y, x);
                cloud.colors.emplace_back(c[2], c[1], c[0], 255);  // BGR → RGBA
            } else {
                cloud.colors.emplace_back(200, 200, 200, 255);
            }
        }
    }
    std::cout << "[depth] Point cloud: " << cloud.points.size() << " points\n";
    return cloud;
}

// ── Normal estimation via central differences (for ICP point-to-plane) ────
void estimateNormals(PointCloud& cloud, const cv::Mat& depth, const CalibData& calib) {
    double fx = calib.K0.at<double>(0, 0);
    double fy = calib.K0.at<double>(1, 1);
    double cx = calib.K0.at<double>(0, 2);
    double cy = calib.K0.at<double>(1, 2);

    cloud.normals.clear();

    auto getPoint = [&](int px, int py) -> Eigen::Vector3f {
        float z = depth.at<float>(py, px);
        return { (float)((px - cx) * z / fx),
                 (float)((py - cy) * z / fy), z };
    };

    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            float Z = depth.at<float>(y, x);
            if (Z <= 0 || Z > 1e4f) continue;

            if (x == 0 || x == depth.cols-1 || y == 0 || y == depth.rows-1) {
                cloud.normals.emplace_back(0.f, 0.f, 1.f);
                continue;
            }
            Eigen::Vector3f dx = getPoint(x+1, y) - getPoint(x-1, y);
            Eigen::Vector3f dy = getPoint(x, y+1) - getPoint(x, y-1);
            Eigen::Vector3f n  = dx.cross(dy);
            float norm = n.norm();
            cloud.normals.push_back(norm > 1e-6f ? n / norm
                                                 : Eigen::Vector3f(0.f, 0.f, 1.f));
        }
    }
    std::cout << "[depth] Normals estimated: " << cloud.normals.size() << "\n";
}

// ── DLT triangulation (for non-rectified / fused pairs) ──────────────────

Eigen::Matrix<float,3,4> buildProjectionMatrix(const cv::Mat& K,
                                                const Eigen::Matrix4f& pose) {
    Eigen::Matrix3f K_e;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            K_e(i,j) = (float)K.at<double>(i,j);
    return K_e * pose.topRows<3>();   // K * [R | t]
}

Eigen::Vector3f triangulateDLT(const Eigen::Vector2f& p0,
                                const Eigen::Vector2f& p1,
                                const Eigen::Matrix<float,3,4>& P0,
                                const Eigen::Matrix<float,3,4>& P1) {
    // Build 4×4 system  A * X = 0  from epipolar constraints
    Eigen::Matrix4f A;
    A.row(0) = p0[0] * P0.row(2) - P0.row(0);
    A.row(1) = p0[1] * P0.row(2) - P0.row(1);
    A.row(2) = p1[0] * P1.row(2) - P1.row(0);
    A.row(3) = p1[1] * P1.row(2) - P1.row(1);

    // Solution: last column of V from SVD(A)
    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f X = svd.matrixV().col(3);

    if (std::abs(X[3]) < 1e-8f) return Eigen::Vector3f::Zero();
    return X.head<3>() / X[3];
}

PointCloud triangulatePoints(const std::vector<Eigen::Vector2f>& pts_left,
                              const std::vector<Eigen::Vector2f>& pts_right,
                              const Eigen::Matrix<float,3,4>& P0,
                              const Eigen::Matrix<float,3,4>& P1) {
    PointCloud cloud;
    cloud.points.reserve(pts_left.size());
    for (size_t i = 0; i < pts_left.size(); ++i) {
        Eigen::Vector3f p = triangulateDLT(pts_left[i], pts_right[i], P0, P1);
        if (p.z() > 0 && p.z() < 1e4f)
            cloud.points.push_back(p);
    }
    std::cout << "[depth] DLT triangulated " << cloud.points.size() << " points\n";
    return cloud;
}

// ── Save PLY (DONE) ───────────────────────────────────────────────────────
void savePointCloud(const PointCloud& cloud, const std::string& path) {
    std::ofstream f(path);
    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << cloud.points.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";

    const bool has_color = !cloud.colors.empty();
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        const auto& p = cloud.points[i];
        f << p.x() << " " << p.y() << " " << p.z() << " ";
        if (has_color) {
            const auto& c = cloud.colors[i];
            f << (int)c[0] << " " << (int)c[1] << " " << (int)c[2] << "\n";
        } else {
            f << "200 200 200\n";
        }
    }
    std::cout << "[depth] Saved point cloud to " << path << "\n";
}

// ── Save COFF — same format as Exercise 5, MeshLab-compatible (DONE) ──────
void savePointCloudOFF(const PointCloud& cloud, const std::string& path) {
    std::ofstream f(path);
    const bool has_color = !cloud.colors.empty();

    f << (has_color ? "COFF" : "OFF") << "\n"
      << cloud.points.size() << " 0 0\n";   // N vertices, 0 faces, 0 edges

    for (size_t i = 0; i < cloud.points.size(); ++i) {
        const auto& p = cloud.points[i];
        f << p.x() << " " << p.y() << " " << p.z();
        if (has_color) {
            const auto& c = cloud.colors[i];
            f << "  " << (int)c[0] << " " << (int)c[1]
              << " "  << (int)c[2] << " 255";
        }
        f << "\n";
    }
    std::cout << "[depth] Saved point cloud to " << path << " (COFF)\n";
}

#if !defined(PIPELINE_BUILD) && !defined(BUILDING_ICP)
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: depth <data_path> <scene_id> <left_view_id> <right_view_id>\n";
        return 1;
    }
    fs::path dataPath(argv[1]);
    const std::string sceneId(argv[2]);
    const std::string viewLeftId = padViewId(std::stoi(argv[3]));
    const std::string viewRightId = padViewId(std::stoi(argv[4]));

    DTUDataLoader loader(dataPath.string());
    CalibData calib = loader.loadCalib(viewLeftId, viewRightId);

    std::string load_path = "results/scene_" + sceneId + "/matching";

    cv::Mat disp = loadDisparity(load_path + "/view_" + viewLeftId + "_" + viewRightId + "_disparity_raw.png");
    cv::Mat colorImg = cv::imread(load_path + "/view_" + viewLeftId + "_" + viewRightId + "_disparity.png");

    auto depth = disparityToDepth(disp, calib);
    auto cloud = depthToPointCloud(depth, calib, colorImg);
    estimateNormals(cloud, depth, calib);

    std::string save_path = "results/scene" + sceneId + "/pointcloud";
    fs::create_directories(save_path);
    savePointCloud(cloud, save_path + "/views_" + viewLeftId + "_" + viewRightId + ".ply");
    savePointCloudOFF(cloud, save_path + "/views_" + viewLeftId + "_" + viewRightId + ".off");
    
    return 0;
}
#endif
