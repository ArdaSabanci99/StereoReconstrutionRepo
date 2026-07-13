#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// ── Camera calibration for a stereo pair ────────────────────────────────────
struct CalibData {
    cv::Mat K0, K1;          // 3×3 intrinsic matrices
    cv::Mat R0, t0;          // left camera absolute pose (world → camera), used for ICP world-frame alignment only
    // Relative pose: right camera relative to left
    cv::Mat R_rel;           // R1 * R0^T
    cv::Mat t_rel;           // metric translation (sparse.t * baseline), not a unit vector

    cv::Mat F;               // 3×3 fundamental matrix (set by sparse matching, used by rectifyManual)

    double baseline = 0.0;   // |camera centres| in mm
    bool   swapped  = false; // true if L/R were swapped during sparse matching to enforce t[0]>0

    // Validation helpers — three levels of completeness:
    //   hasIntrinsics:   K0, K1, baseline — minimum for sparse matching
    //   hasRelativePose: R_rel, T_rel     — relative pose between the two cameras
    //   hasLeftCamExtrinsics: R0, t0       — left camera world-frame pose, needed for ICP alignment
    bool hasIntrinsics()        const { return !K0.empty() && !K1.empty() && baseline > 0.0; }
    bool hasRelativePose()      const { return !R_rel.empty() && !t_rel.empty(); }
    bool hasLeftCamExtrinsics() const { return !R0.empty() && !t0.empty(); }
    bool hasFundamentalMatrix()  const { return !F.empty(); }
    bool isFullyValid()         const { return hasIntrinsics() && hasRelativePose() && hasLeftCamExtrinsics(); }

    // check if left camera is geometrically to the left
    // if not, fix
    void verifyLeftRightCameraOrder();
};

// ── CalibData serialisation (YAML via cv::FileStorage) ──────────────────────
// Saves/loads the complete CalibData struct. Complements DTUDataLoader::loadCalibIntrinsics,
// which only populates K0, K1, baseline. Use these to persist/reload the full
// (potentially-swapped) calib between standalone pipeline steps.
void saveCalibData(const CalibData& calib, const std::string& path);
CalibData loadCalibData(const std::string& path);

// ── Inlier correspondences serialisation ────────────────────────────────────
// Saves/loads the RANSAC inlier point pairs produced by sparse matching, required by rectifyManual
void saveInlierPoints(const std::vector<cv::Point2f>& left,
                      const std::vector<cv::Point2f>& right,
                      const std::string& path);
void loadInlierPoints(const std::string& path,
                      std::vector<cv::Point2f>& left,
                      std::vector<cv::Point2f>& right);

// ── Disparity save / load (16-bit PNG, scale ×16) ───────────────────────────
void saveDisparity(const cv::Mat& disp, const std::string& path);
cv::Mat loadDisparity(const std::string& path);

// ── View ID padding (1 → "001") ─────────────────────────────────────────────
std::string padViewId(int id);

// ── Debug print ──────────────────────────────────────────────────────────────
void printMatInfo(const std::string& name, const cv::Mat& m);
