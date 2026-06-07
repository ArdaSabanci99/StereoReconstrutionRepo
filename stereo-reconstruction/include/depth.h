#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "utils.h"
#include "PointCloud.h"   // Eigen-based PointCloud

// Convert disparity map to metric depth  Z = f * B / (d + doffs)
cv::Mat disparityToDepth(const cv::Mat& disparity, const CalibData& calib);

// Back-project depth map to 3D point cloud.
// color_img: optional BGR image for vertex colors.
PointCloud depthToPointCloud(const cv::Mat& depth, const CalibData& calib,
                              const cv::Mat& color_img = cv::Mat());

// Estimate per-point normals from depth map using central differences.
// Call after depthToPointCloud to populate cloud.normals.
void estimateNormals(PointCloud& cloud, const cv::Mat& depth, const CalibData& calib);

// Save point cloud as ASCII PLY
void savePointCloud(const PointCloud& cloud, const std::string& path);
