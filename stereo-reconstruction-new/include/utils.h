#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// ── Camera calibration for a stereo pair ────────────────────────────────────
struct CalibData {
    cv::Mat K0, K1;          // 3×3 intrinsic matrices
    cv::Mat R0, R1;          // 3×3 rotation  (world → camera)
    cv::Mat t0, t1;          // 3×1 translation (world → camera)
    cv::Mat P0, P1;          // 3×4 projection matrices (K[R|t])

    // Relative pose: right camera relative to left
    cv::Mat R_rel;           // R1 * R0^T
    cv::Mat T_rel;           // t1 - R_rel * t0

    double baseline = 0.0;   // |camera centres| in mm
    double doffs    = 0.0;   // disparity offset (Middlebury; 0 for DTU)
    int    vmin     = 0;     // min valid disparity (for evaluation)
    int    vmax     = 255;   // max valid disparity
};

// ── Disparity save / load (16-bit PNG, scale ×16) ───────────────────────────
void saveDisparity(const cv::Mat& disp, const std::string& path);
cv::Mat loadDisparity(const std::string& path);

// ── View ID padding (1 → "001") ─────────────────────────────────────────────
std::string padViewId(int id);

// ── Debug print ──────────────────────────────────────────────────────────────
void printMatInfo(const std::string& name, const cv::Mat& m);
