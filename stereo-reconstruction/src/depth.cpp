#include "depth.h"
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
                cloud.colors.emplace_back(c[2], c[1], c[0], 255);  // BGR -> RGBA
            } else {
                cloud.colors.emplace_back(200, 200, 200, 255);
            }
        }
    }
    std::cout << "[depth] Point cloud: " << cloud.points.size() << " points\n";
    return cloud;
}

// ── Normal estimation from depth map (for ICP point-to-plane) ─────────────
void estimateNormals(PointCloud& cloud, const cv::Mat& depth, const CalibData& calib) {
    // Normals are estimated per-pixel using central differences, then stored
    // in the same order as the points were added in depthToPointCloud.

    double fx = calib.K0.at<double>(0, 0);
    double fy = calib.K0.at<double>(1, 1);
    double cx = calib.K0.at<double>(0, 2);
    double cy = calib.K0.at<double>(1, 2);

    cloud.normals.clear();

    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            float Z = depth.at<float>(y, x);
            if (Z <= 0 || Z > 1e4f) continue;

            // Central differences on 3D positions
            auto getPoint = [&](int px, int py) -> Eigen::Vector3f {
                float z = depth.at<float>(py, px);
                return { (float)((px - cx) * z / fx),
                         (float)((py - cy) * z / fy), z };
            };

            if (x == 0 || x == depth.cols-1 || y == 0 || y == depth.rows-1) {
                cloud.normals.emplace_back(0.f, 0.f, 1.f);
                continue;
            }

            Eigen::Vector3f dx = getPoint(x+1, y) - getPoint(x-1, y);
            Eigen::Vector3f dy = getPoint(x, y+1) - getPoint(x, y-1);

            Eigen::Vector3f n = dx.cross(dy);
            float norm = n.norm();
            cloud.normals.push_back(norm > 1e-6f ? n / norm
                                                 : Eigen::Vector3f(0.f, 0.f, 1.f));
        }
    }
    std::cout << "[depth] Normals estimated: " << cloud.normals.size() << "\n";
}

// ── Save Point Cloud as ASCII PLY (DONE) ──────────────────────────────────
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

#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: depth <scene_dir>\n"; return 1;
    }
    fs::path scene(argv[1]);
    CalibData calib = loadCalib(scene);

    cv::Mat disp  = loadDisparity("results/disparity_raw.png");
    cv::Mat left  = cv::imread("results/left_rect.png");

    auto depth = disparityToDepth(disp, calib);
    auto cloud = depthToPointCloud(depth, calib, left);
    estimateNormals(cloud, depth, calib);

    fs::create_directories("results");
    savePointCloud(cloud, "results/pointcloud.ply");
    return 0;
}
#endif
