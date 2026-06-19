#include "utils.h"
#include <iomanip>
#include <sstream>
#include <iostream>

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
