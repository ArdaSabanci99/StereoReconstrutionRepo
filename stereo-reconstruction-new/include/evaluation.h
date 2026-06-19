#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

struct EvalResult {
    double bad1     = 0;   // % pixels with |err| > 1 px
    double bad2     = 0;   // % pixels with |err| > 2 px
    double bad4     = 0;   // % pixels with |err| > 4 px
    double rmse     = 0;   // sqrt(mean squared error)
    double avgerr   = 0;   // mean absolute error
    double coverage = 0;   // % of GT pixels with valid estimate
    int    valid    = 0;   // # pixels with valid estimate over GT region
    int    total_gt = 0;   // # valid GT pixels
};

// Load a Middlebury PFM ground-truth disparity map
cv::Mat loadPFM(const fs::path& path);

// Evaluate estimated disparity against ground truth.
// Pixels with gt <= 0 are ignored (occluded / invalid).
// Pixels with est outside [vmin, vmax] count as invalid → reduce coverage.
EvalResult evaluateDisparity(const cv::Mat& estimated,
                              const cv::Mat& ground_truth,
                              int vmin = 0, int vmax = 1000);

void printEvalResult(const EvalResult& r, const std::string& label = "");
