#include "utils.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

void saveDisparity(const cv::Mat& disp, const fs::path& path) {
    cv::Mat disp_16u;
    disp.convertTo(disp_16u, CV_16U, 256.0);
    cv::imwrite(path.string(), disp_16u);
}

cv::Mat loadDisparity(const fs::path& path) {
    cv::Mat disp_16u = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    cv::Mat disp;
    disp_16u.convertTo(disp, CV_32F, 1.0 / 256.0);
    return disp;
}

std::string padViewId(int viewId) {
    std::ostringstream out;
    out << std::setw(3) << std::setfill('0') << viewId;
    return out.str();
}

void printMatInfo(const std::string& name, const cv::Mat& m) {
    std::cout << name << ": " << m.rows << "x" << m.cols
              << " type=" << m.type()
              << " channels=" << m.channels() << "\n";
}
