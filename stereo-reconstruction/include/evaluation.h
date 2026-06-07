#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Evaluation metrics for a single disparity map (Member 4)
struct EvalResult {
    double bad1     = 0;   // % pixels with |err| > 1 px
    double bad2     = 0;   // % pixels with |err| > 2 px
    double bad4     = 0;   // % pixels with |err| > 4 px
    double rmse     = 0;   // root mean squared error (px)
    double avgerr   = 0;   // mean absolute error (px)
    double coverage = 0;   // % of valid-GT pixels that have a valid estimate
    int    valid    = 0;   // pixels evaluated (valid GT + valid estimate)
    int    total_gt = 0;   // total pixels with valid ground truth
};

// Load Middlebury .pfm ground-truth disparity map
cv::Mat loadPFM(const fs::path& path);

// Compute bad-pixel ratios, RMSE, and coverage against ground truth.
// Pixels where ground_truth <= 0 are excluded (invalid GT).
EvalResult evaluateDisparity(const cv::Mat& estimated,
                              const cv::Mat& ground_truth,
                              int vmin = 0, int vmax = 255);

void printEvalResult(const EvalResult& r, const std::string& label = "");
