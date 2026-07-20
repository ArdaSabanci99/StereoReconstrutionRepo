#pragma once
#include "utils.h"
#include "PointCloud.h"
#include <opencv2/opencv.hpp>

// Convert disparity map to per-pixel depth (Z = f*B/d), invalid pixels → 0.
cv::Mat disparityToDepth(const cv::Mat& disparity, const CalibData& calib);

PointCloud disparityToCloudViaP(const cv::Mat& disp, const cv::Mat& P1, const cv::Mat& P2,
                                 const cv::Mat& color_img,
                                 float min_disp = 1.0f, float max_depth = 1e4f,
                                 const cv::Mat& mask1 = cv::Mat(), const cv::Mat& mask2 = cv::Mat());

// Convert disparity map to coloured point cloud.
// Uses the Q matrix from rectification and the colour image for RGB.
PointCloud disparityToCloud(const cv::Mat& disp,
                             const cv::Mat& Q,
                             const cv::Mat& color_img,
                             float min_disp = 1.0f,
                             float max_depth= 5000.0f);

// Normal estimation via PCA on local neighbourhood
void estimateNormals(PointCloud& cloud, int k_neighbours = 10);

// Save / Load PLY
void savePointCloud(const PointCloud& cloud, const std::string& path);
void savePointCloudOFF(const PointCloud& cloud, const std::string& path);
