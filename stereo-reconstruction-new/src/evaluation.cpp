#include "evaluation.h"
#include "utils.h"
#include "matching.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// PFM loader
// Format: "Pf\n<width> <height>\n<scale>\n<binary float data>"
// Rows stored bottom-to-top; scale < 0 means little-endian.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat loadPFM(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "[eval] Cannot open " << path << "\n"; return {}; }

    std::string magic;
    f >> magic;
    if (magic != "Pf" && magic != "PF") {
        std::cerr << "[eval] Not a PFM file (magic=" << magic << ")\n"; return {};
    }

    int W, H; float scale;
    f >> W >> H >> scale;
    f.ignore(1);   // skip newline after scale

    bool little_endian = (scale < 0.0f);

    cv::Mat img(H, W, CV_32F);

    // PFM is stored bottom-row first
    for (int y = H - 1; y >= 0; --y) {
        float* row = img.ptr<float>(y);
        f.read(reinterpret_cast<char*>(row), W * sizeof(float));

        // Byte-swap if host endianness differs from file
        if (little_endian) {
            // On a little-endian host (x86/ARM), no swap needed when file is LE
            // If host is BE, swap. We'll auto-detect via union trick:
            union { uint32_t u; uint8_t b[4]; } probe; probe.u = 1;
            bool host_le = (probe.b[0] == 1);
            if (!host_le) {
                for (int x = 0; x < W; ++x) {
                    uint32_t* p = reinterpret_cast<uint32_t*>(&row[x]);
                    *p = ((*p & 0xFF000000u) >> 24) | ((*p & 0x00FF0000u) >>  8) |
                         ((*p & 0x0000FF00u) <<  8) | ((*p & 0x000000FFu) << 24);
                }
            }
        } else {
            // File is big-endian; swap on LE host
            union { uint32_t u; uint8_t b[4]; } probe; probe.u = 1;
            bool host_le = (probe.b[0] == 1);
            if (host_le) {
                for (int x = 0; x < W; ++x) {
                    uint32_t* p = reinterpret_cast<uint32_t*>(&row[x]);
                    *p = ((*p & 0xFF000000u) >> 24) | ((*p & 0x00FF0000u) >>  8) |
                         ((*p & 0x0000FF00u) <<  8) | ((*p & 0x000000FFu) << 24);
                }
            }
        }
    }

    std::cout << "[eval] Loaded PFM " << path << "  " << W << "×" << H
              << "  endian=" << (little_endian ? "LE" : "BE") << "\n";
    return img;
}

// ─────────────────────────────────────────────────────────────────────────────
// Disparity evaluation
// ─────────────────────────────────────────────────────────────────────────────
EvalResult evaluateDisparity(const cv::Mat& estimated,
                              const cv::Mat& ground_truth,
                              int vmin, int vmax) {
    CV_Assert(estimated.type()   == CV_32F);
    CV_Assert(ground_truth.type()== CV_32F);
    CV_Assert(estimated.size()   == ground_truth.size());

    EvalResult r;
    double sum_sq = 0, sum_abs = 0;
    int bad1 = 0, bad2 = 0, bad4 = 0;

    for (int y = 0; y < ground_truth.rows; ++y)
        for (int x = 0; x < ground_truth.cols; ++x) {
            float gt = ground_truth.at<float>(y, x);
            if (gt <= 0.0f) continue;   // invalid GT
            ++r.total_gt;

            float est = estimated.at<float>(y, x);
            if (est < (float)vmin || est > (float)vmax) continue;  // no valid est
            ++r.valid;

            float err = std::abs(est - gt);
            if (err > 1.0f) ++bad1;
            if (err > 2.0f) ++bad2;
            if (err > 4.0f) ++bad4;
            sum_sq  += (double)(err * err);
            sum_abs += (double)err;
        }

    if (r.valid > 0) {
        r.bad1   = 100.0 * bad1 / r.valid;
        r.bad2   = 100.0 * bad2 / r.valid;
        r.bad4   = 100.0 * bad4 / r.valid;
        r.rmse   = std::sqrt(sum_sq  / r.valid);
        r.avgerr = sum_abs / r.valid;
    }
    if (r.total_gt > 0)
        r.coverage = 100.0 * r.valid / r.total_gt;

    return r;
}

void printEvalResult(const EvalResult& r, const std::string& label) {
    if (!label.empty()) std::cout << "=== " << label << " ===\n";
    std::cout
        << "  bad1     = " << r.bad1     << " %\n"
        << "  bad2     = " << r.bad2     << " %\n"
        << "  bad4     = " << r.bad4     << " %\n"
        << "  RMSE     = " << r.rmse     << " px\n"
        << "  avg err  = " << r.avgerr   << " px\n"
        << "  coverage = " << r.coverage << " %\n"
        << "  valid    = " << r.valid    << " / " << r.total_gt << " px\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone: run all methods on a Middlebury scene and print comparison table
// ─────────────────────────────────────────────────────────────────────────────
#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: evaluation <left.png> <right.png> <gt.pfm> [vmin vmax]\n";
        return 1;
    }
    cv::Mat imgL = cv::imread(argv[1]);
    cv::Mat imgR = cv::imread(argv[2]);
    cv::Mat gt   = loadPFM(argv[3]);

    if (imgL.empty() || imgR.empty() || gt.empty()) {
        std::cerr << "Could not load inputs.\n"; return 1;
    }

    int vmin = (argc >= 5) ? std::stoi(argv[4]) : 0;
    int vmax = (argc >= 6) ? std::stoi(argv[5]) : 1000;

    struct { const char* name; MatchMethod m; } methods[] = {
        {"SAD",    MatchMethod::MANUAL_SAD},
        {"SSD",    MatchMethod::MANUAL_SSD},
        {"NCC",    MatchMethod::MANUAL_NCC},
        {"Census", MatchMethod::MANUAL_CENSUS},
        {"SGM",    MatchMethod::MANUAL_SGM},
        {"BM",     MatchMethod::OPENCV_BM},
        {"SGBM",   MatchMethod::OPENCV_SGBM},
    };

    std::cout << "\n" << std::left
              << std::setw(8) << "Method"
              << std::setw(10) << "bad1(%)"
              << std::setw(10) << "bad2(%)"
              << std::setw(10) << "RMSE"
              << std::setw(10) << "cover(%)"
              << std::setw(10) << "time(ms)" << "\n";
    std::cout << std::string(58, '-') << "\n";

    for (auto& mi : methods) {
        MatchParams p;
        p.method = mi.m;
        p.num_disparities = 128;
        p.window_size = 5;

        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat disp = computeDisparity(imgL, imgR, p);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count();

        // Resize disp to GT size if needed
        if (disp.size() != gt.size())
            cv::resize(disp, disp, gt.size(), 0, 0, cv::INTER_NEAREST);

        EvalResult r = evaluateDisparity(disp, gt, vmin, vmax);
        std::cout << std::left
                  << std::setw(8)  << mi.name
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.bad1
                  << std::setw(10) << r.bad2
                  << std::setw(10) << r.rmse
                  << std::setw(10) << r.coverage
                  << std::setw(10) << (int)ms << "\n";
    }
    return 0;
}
#endif
