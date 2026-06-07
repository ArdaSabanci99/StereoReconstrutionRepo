#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Evaluation metrics for a single disparity map (Member 4)
struct EvalResult {
    double bad1   = 0;   // % pixels with |err| > 1 px
    double bad2   = 0;   // % pixels with |err| > 2 px
    double bad4   = 0;   // % pixels with |err| > 4 px
    double rmse   = 0;   // root mean squared error (px)
    double avgerr = 0;   // mean absolute error (px)
    int    valid  = 0;   // number of evaluated pixels
};

// Load Middlebury .pfm ground-truth disparity map
cv::Mat loadPFM(const fs::path& path);

// Compute bad-pixel ratios and RMSE against ground truth.
// Pixels where ground_truth <= 0 are excluded (invalid GT).
EvalResult evaluateDisparity(const cv::Mat& estimated,
                              const cv::Mat& ground_truth,
                              int vmin = 0, int vmax = 255);

void printEvalResult(const EvalResult& r, const std::string& label = "");
