#include "utils.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <stdexcept>

void saveDisparity(const cv::Mat& disp, const std::string& path) {
    // Invalid pixels (negative) are stored as 0; valid disparities scaled ×16.
    cv::Mat clipped;
    cv::max(disp, 0.0f, clipped);
    cv::Mat disp16;
    clipped.convertTo(disp16, CV_16U, 16.0);
    cv::imwrite(path, disp16);
}

cv::Mat loadDisparity(const std::string& path) {
    cv::Mat disp16 = cv::imread(path, cv::IMREAD_UNCHANGED);
    cv::Mat disp32;
    disp16.convertTo(disp32, CV_32F, 1.0 / 16.0);
    return disp32;
}

std::string padViewId(int id) {
    std::ostringstream ss;
    ss << std::setw(3) << std::setfill('0') << id;
    return ss.str();
}

void printMatInfo(const std::string& name, const cv::Mat& m) {
    std::cout << name << ": " << m.cols << "×" << m.rows
              << " channels=" << m.channels() << "\n";
}

void saveCalibData(const CalibData& calib, const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        throw std::runtime_error("[calib] Cannot open for writing: " + path);
    fs << "baseline" << calib.baseline
       << "swapped"  << calib.swapped
       << "K0"       << calib.K0
       << "K1"       << calib.K1;
    if (!calib.F.empty())     fs << "F"     << calib.F;
    if (!calib.R0.empty())    fs << "R0"    << calib.R0;
    if (!calib.t0.empty())    fs << "t0"    << calib.t0;
    if (!calib.R_rel.empty()) fs << "R_rel" << calib.R_rel;
    if (!calib.t_rel.empty()) fs << "t_rel" << calib.t_rel;
}

CalibData loadCalibData(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("[calib] Cannot open: " + path);
    CalibData calib;
    fs["baseline"] >> calib.baseline;
    fs["swapped"]  >> calib.swapped;
    fs["K0"]       >> calib.K0;
    fs["K1"]       >> calib.K1;
    fs["F"]        >> calib.F;
    fs["R0"]       >> calib.R0;
    fs["t0"]       >> calib.t0;
    fs["R_rel"]    >> calib.R_rel;
    fs["t_rel"]    >> calib.t_rel;
    return calib;
}

void saveInlierPoints(const std::vector<cv::Point2f>& left,
                      const std::vector<cv::Point2f>& right,
                      const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        throw std::runtime_error("[inliers] Cannot open for writing: " + path);
    fs << "inlier_pts_left" << left << "inlier_pts_right" << right;
}

void loadInlierPoints(const std::string& path,
                      std::vector<cv::Point2f>& left,
                      std::vector<cv::Point2f>& right) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("[inliers] Cannot open: " + path);
    fs["inlier_pts_left"]  >> left;
    fs["inlier_pts_right"] >> right;
}

void CalibData::verifyLeftRightCameraOrder() {
    if (!hasRelativePose()) {
        throw std::runtime_error("[verify L/R] Cannot verify left/right camera order: missing relative pose (R_rel, t_rel).");
    }
    if (t_rel.at<double>(0) > 0) {  // In OpenCV recoverPose convention (x2 = R*x1 + t), t[0] < 0 means camera 2 is to the RIGHT (correct order). Swap only when t[0] > 0 (camera 2 is to the LEFT).
        std::cout << "[verify L/R]: Right camera is geometrically to the left of the left camera. Swapping L/R.\n";
        
        if (!hasIntrinsics()) {
            throw std::runtime_error("[verify L/R] Cannot swap left/right cameras: missing intrinsics (K0, K1, baseline).");
        }

        std::swap(K0, K1);
        
        R_rel = R_rel.t();
        t_rel = -R_rel * t_rel;

        if (hasFundamentalMatrix())
            F = F.t();
    
        swapped = !swapped;
    }
}