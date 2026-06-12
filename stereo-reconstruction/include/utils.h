#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>


// TODO: change with DTU

namespace fs = std::filesystem;

struct CalibData {
    cv::Mat K0;        // camera matrix left
    cv::Mat K1;        // camera matrix right
    double baseline;   // in mm
    int vmin  = 0;     // min valid disparity (for evaluation)
    int vmax  = 255;   // max valid disparity
    int doffs = 0;  // not used in DTU

};

// Save/load disparity map as .pfm or .png
void saveDisparity(const cv::Mat& disp, const fs::path& path);
cv::Mat loadDisparity(const fs::path& path);

// Format DTU view IDs as three-digit zero-padded strings.
std::string padViewId(int viewId);

// Print a cv::Mat summary (for debugging)
void printMatInfo(const std::string& name, const cv::Mat& m);
