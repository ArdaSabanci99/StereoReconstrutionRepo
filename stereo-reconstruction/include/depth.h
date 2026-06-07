#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "utils.h"
#include "PointCloud.h"

// ── Rectified pipeline (disparity → depth → point cloud) ─────────────────

// Convert disparity map to metric depth:  Z = f * B / (d + doffs)
cv::Mat disparityToDepth(const cv::Mat& disparity, const CalibData& calib);

// Back-project depth map to 3D point cloud.
// color_img: optional BGR image for vertex colors.
PointCloud depthToPointCloud(const cv::Mat& depth, const CalibData& calib,
                              const cv::Mat& color_img = cv::Mat());

// Estimate per-vertex normals from depth map using central differences.
// Populates cloud.normals in the same order as depthToPointCloud.
void estimateNormals(PointCloud& cloud, const cv::Mat& depth, const CalibData& calib);

// ── Non-rectified / fused pipeline (DLT triangulation) ────────────────────

// Build a 3×4 projection matrix  P = K * [R | t]  from intrinsics and pose.
// pose: 4×4 world-to-camera transform (Eigen, same convention as ICPOptimizer).
Eigen::Matrix<float,3,4> buildProjectionMatrix(const cv::Mat& K,
                                                const Eigen::Matrix4f& pose);

// Triangulate a single 3D point from two pixel observations (homogeneous DLT).
// Returns the 3D point in the world frame.
Eigen::Vector3f triangulateDLT(const Eigen::Vector2f& p0,
                                const Eigen::Vector2f& p1,
                                const Eigen::Matrix<float,3,4>& P0,
                                const Eigen::Matrix<float,3,4>& P1);

// Triangulate a list of corresponding 2D points (e.g. from sparse matching).
PointCloud triangulatePoints(const std::vector<Eigen::Vector2f>& pts_left,
                              const std::vector<Eigen::Vector2f>& pts_right,
                              const Eigen::Matrix<float,3,4>& P0,
                              const Eigen::Matrix<float,3,4>& P1);

// ── Output ────────────────────────────────────────────────────────────────

// Save as ASCII PLY (binary viewers, CloudCompare)
void savePointCloud(const PointCloud& cloud, const std::string& path);

// Save as colored OFF (COFF) — same format as Exercise 5, MeshLab compatible
void savePointCloudOFF(const PointCloud& cloud, const std::string& path);
