#include "utils.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <stdexcept>

void CalibData::verifyLeftRightCameraOrder() {
    if (!hasRelativePose()) {
        throw std::runtime_error("[verify L/R] Cannot verify left/right camera order: missing relative pose (R_rel, t_rel).");
    }
    if (t_rel.at<double>(0) > 0) {
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

void printMatInfo(const std::string& name, const cv::Mat& m) {
    std::cout << name << ": " << m.cols << "×" << m.rows
              << " channels=" << m.channels() << "\n";
}

std::string padViewId(int id) {
    std::ostringstream ss;
    ss << std::setw(3) << std::setfill('0') << id;
    return ss.str();
}

Eigen::Vector3d triangulatePoint(const Eigen::Vector3d & pt_left, const Eigen::Vector3d & pt_right,
                                  const Eigen::Matrix<double, 3, 4> & P_left, const Eigen::Matrix<double, 3, 4> & P_right) {
    Eigen::Matrix4d A;

    // Constraints from left view
        // u = p1.X / p3.X -> u . (p3.X) - (p1.X) = 0 -> (u.p3 - p1)X = 0 -> row in linear system AX = 0
        // v = p2.X / p3.X
    A.row(0) = pt_left.x() * P_left.row(2) - P_left.row(0);
    A.row(1) = pt_left.y() * P_left.row(2) - P_left.row(1);

    // Constraints from right view
    A.row(2) = pt_right.x() * P_right.row(2) - P_right.row(0);
    A.row(3) = pt_right.y() * P_right.row(2) - P_right.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

    // Smallest singular vector minimizes AX=0
    Eigen::Vector4d X_hom = svd.matrixV().col(3);

    // Normalizing by scale factor
    return X_hom.head<3>() / X_hom(3);
}

cv::Mat eigenToCv(const Eigen::Matrix3d& src) {
    cv::Mat dst(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            dst.at<double>(i, j) = src(i, j);
    return dst;
}

cv::Mat eigenToCv34(const Eigen::Matrix<double, 3, 4>& src) {
    cv::Mat dst(3, 4, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            dst.at<double>(i, j) = src(i, j);
    return dst;
}

Eigen::Matrix3d cvToEigen3x3(const cv::Mat& src) {
    Eigen::Matrix3d dst;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            dst(i, j) = src.at<double>(i, j);
    return dst;
}

Eigen::Vector3d cvToEigenVec3(const cv::Mat& src) {
    return Eigen::Vector3d(src.at<double>(0), src.at<double>(1), src.at<double>(2));
}
