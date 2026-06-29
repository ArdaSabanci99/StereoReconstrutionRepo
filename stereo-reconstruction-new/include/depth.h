#pragma once
#include "utils.h"
#include "PointCloud.h"
#include <opencv2/opencv.hpp>

// Convert disparity map to per-pixel depth (Z = f*B/d), invalid pixels → 0.
cv::Mat disparityToDepth(const cv::Mat& disparity, const CalibData& calib);

// Convert disparity map to coloured point cloud.
// Uses the Q matrix from rectification and the colour image for RGB.
PointCloud disparityToCloud(const cv::Mat& disp,
                             const cv::Mat& Q,
                             const cv::Mat& color_img,
                             float min_disp = 1.0f,
                             float max_depth= 5000.0f);

// DLT triangulation for a single point pair.
// pts1, pts2: corresponding points (pixel coords)
// P1, P2: 3×4 projection matrices
cv::Mat triangulatePointDLT(const cv::Point2f& pt1, const cv::Point2f& pt2,
                              const cv::Mat& P1, const cv::Mat& P2);

// Triangulate all valid disparity pixels using DLT (for non-rectified pairs)
PointCloud triangulateCloud(const std::vector<cv::Point2f>& pts1,
                              const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& P1, const cv::Mat& P2,
                              const std::vector<cv::Vec3b>& colors);

// Normal estimation via PCA on local neighbourhood
void estimateNormals(PointCloud& cloud, int k_neighbours = 10);

// Save / Load PLY
void savePointCloud(const PointCloud& cloud, const std::string& path);
void savePointCloudOFF(const PointCloud& cloud, const std::string& path);
