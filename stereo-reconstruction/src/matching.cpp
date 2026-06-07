#include "matching.h"
#include "utils.h"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <bitset>

namespace fs = std::filesystem;

// ── Manual SAD (DONE) ──────────────────────────────────────────
static cv::Mat computeSAD(const cv::Mat& left, const cv::Mat& right,
                           const MatchParams& p) {
    int half = p.window_size / 2;
    cv::Mat disp(left.size(), CV_32F, -1.0f);

    for (int y = half; y < left.rows - half; ++y) {
        for (int x = half + p.min_disparity + p.num_disparities;
             x < left.cols - half; ++x) {
            float best_cost = std::numeric_limits<float>::max();
            int   best_d    = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d;
                if (xr - half < 0) break;
                float cost = 0;
                for (int dy = -half; dy <= half; ++dy)
                    for (int dx = -half; dx <= half; ++dx)
                        cost += std::abs((float)left .at<uchar>(y+dy, x+dx) -
                                         (float)right.at<uchar>(y+dy, xr+dx));
                if (cost < best_cost) { best_cost = cost; best_d = d; }
            }
            disp.at<float>(y, x) = (float)best_d;
        }
    }
    return disp;
}

// ── Manual SSD (DONE) ──────────────────────────────────────────
static cv::Mat computeSSD(const cv::Mat& left, const cv::Mat& right,
                           const MatchParams& p) {
    int half = p.window_size / 2;
    cv::Mat disp(left.size(), CV_32F, -1.0f);

    for (int y = half; y < left.rows - half; ++y) {
        for (int x = half + p.min_disparity + p.num_disparities;
             x < left.cols - half; ++x) {
            float best_cost = std::numeric_limits<float>::max();
            int   best_d    = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d;
                if (xr - half < 0) break;
                float cost = 0;
                for (int dy = -half; dy <= half; ++dy)
                    for (int dx = -half; dx <= half; ++dx) {
                        float diff = (float)left .at<uchar>(y+dy, x+dx) -
                                     (float)right.at<uchar>(y+dy, xr+dx);
                        cost += diff * diff;
                    }
                if (cost < best_cost) { best_cost = cost; best_d = d; }
            }
            disp.at<float>(y, x) = (float)best_d;
        }
    }
    return disp;
}

// ── Manual NCC (DONE) ──────────────────────────────────────────
static cv::Mat computeNCC(const cv::Mat& left, const cv::Mat& right,
                           const MatchParams& p) {
    int half = p.window_size / 2;
    int N    = p.window_size * p.window_size;
    cv::Mat disp(left.size(), CV_32F, -1.0f);

    for (int y = half; y < left.rows - half; ++y) {
        for (int x = half + p.min_disparity + p.num_disparities;
             x < left.cols - half; ++x) {
            float best_ncc = -2.0f;
            int   best_d   = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d;
                if (xr - half < 0) break;
                float sumL=0, sumR=0, sumLL=0, sumRR=0, sumLR=0;
                for (int dy = -half; dy <= half; ++dy)
                    for (int dx = -half; dx <= half; ++dx) {
                        float l = left .at<uchar>(y+dy, x+dx);
                        float r = right.at<uchar>(y+dy, xr+dx);
                        sumL+=l; sumR+=r; sumLL+=l*l; sumRR+=r*r; sumLR+=l*r;
                    }
                float meanL = sumL/N, meanR = sumR/N;
                float num   = sumLR - N*meanL*meanR;
                float den   = std::sqrt((sumLL - N*meanL*meanL) *
                                        (sumRR - N*meanR*meanR));
                float ncc = (den < 1e-6f) ? 0.0f : num / den;
                if (ncc > best_ncc) { best_ncc = ncc; best_d = d; }
            }
            disp.at<float>(y, x) = (float)best_d;
        }
    }
    return disp;
}

// ── Manual Census transform + Hamming cost (Member 3) ─────────────────────
//
// Radiometrically robust: only relative orderings within the window matter,
// so gain/bias changes (different exposures, vignetting) don't affect the cost.
// Zabih & Woodfill, ECCV 1994.

static cv::Mat censusTransform(const cv::Mat& img) {
    // 5×5 window (half=2) → 24 bits → fits in uint32_t (stored as CV_32S)
    // For 7×7 (48 bits) use uint64_t + CV_64F with reinterpret_cast.
    const int half = 2;
    cv::Mat census(img.size(), CV_32S, 0);
    for (int y = half; y < img.rows - half; ++y) {
        for (int x = half; x < img.cols - half; ++x) {
            uint32_t code = 0;
            int bit = 0;
            uchar center = img.at<uchar>(y, x);
            for (int dy = -half; dy <= half; ++dy)
                for (int dx = -half; dx <= half; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (img.at<uchar>(y + dy, x + dx) < center)
                        code |= (1u << bit);
                    ++bit;
                }
            census.at<int>(y, x) = static_cast<int>(code);
        }
    }
    return census;
}

static int hammingDist(int a, int b) {
    return (int)std::bitset<32>((unsigned)a ^ (unsigned)b).count();
}

static cv::Mat computeCensus(const cv::Mat& left, const cv::Mat& right,
                              const MatchParams& p) {
    cv::Mat cen_l = censusTransform(left);
    cv::Mat cen_r = censusTransform(right);

    // Extra border from census window (half=2) + aggregation window
    const int census_half = 2;
    const int agg_half    = p.window_size / 2;
    const int border      = census_half + agg_half;

    cv::Mat disp(left.size(), CV_32F, -1.0f);

    for (int y = border; y < left.rows - border; ++y) {
        for (int x = border + p.min_disparity + p.num_disparities;
             x < left.cols - border; ++x) {
            float best_cost = std::numeric_limits<float>::max();
            int   best_d    = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d;
                if (xr - border < 0) break;
                float cost = 0;
                for (int dy = -agg_half; dy <= agg_half; ++dy)
                    for (int dx = -agg_half; dx <= agg_half; ++dx)
                        cost += hammingDist(cen_l.at<int>(y+dy, x+dx),
                                            cen_r.at<int>(y+dy, xr+dx));
                if (cost < best_cost) { best_cost = cost; best_d = d; }
            }
            disp.at<float>(y, x) = (float)best_d;
        }
    }
    return disp;
}

// ── Manual SGM — Semi-Global Matching (Member 3 — C1 challenge) ───────────
static cv::Mat computeSGM(const cv::Mat& left, const cv::Mat& right,
                           const MatchParams& p) {
    // Hirschmüller, "Stereo Processing by Semiglobal Matching and Mutual
    // Information", PAMI 2008.
    //
    // TODO-SGM-1: Per-pixel cost volume using Census transform
    //   Census encodes the sign pattern of a neighbourhood relative to the
    //   centre pixel as a bit string, then uses Hamming distance as the cost.
    //   For each pixel (y,x) build a 5×5 (or 7×7) bitmask:
    //     uint64_t census(y, x) = 0;
    //     int bit = 0;
    //     for (dy, dx) in neighbourhood:
    //         if left(y+dy, x+dx) < left(y, x): set bit
    //         ++bit;
    //   cost_volume[y][x][d] = hamming_distance(census_l(y,x), census_r(y, x-d))
    //
    // TODO-SGM-2: Path-cost aggregation over 4 or 8 scan directions
    //   For each direction r ∈ {←, →, ↑, ↓, ↖, ↗, ↙, ↘}:
    //     Lr(p, d) = C(p,d) + min(
    //         Lr(p-r, d),
    //         Lr(p-r, d-1) + P1,
    //         Lr(p-r, d+1) + P1,
    //         min_k Lr(p-r, k) + P2
    //     ) - min_k Lr(p-r, k)
    //   Suggested starting values: P1 = 10, P2 = 120.
    //   Aggregate:  S(p, d) = Σ_r Lr(p, d)
    //
    // TODO-SGM-3: Winner-takes-all disparity selection
    //   disp(p) = argmin_d S(p, d)
    //
    // TODO-SGM-4: Post-processing (same as block-matching)
    //   Left-right consistency check (TODO-B below)
    //   Median filter + hole filling (TODO-C below)

    std::cout << "[matching] SGM not yet implemented, falling back to SAD.\n";
    return computeSAD(left, right, p);
}

// ── TODO: Advanced matching improvements ──────────────────────
// TODO-A: Sub-pixel refinement
//   After finding best integer disparity, fit a parabola to cost[d-1], cost[d], cost[d+1]
//   → sub-pixel accuracy: d_sub = d - 0.5 * (cost[d+1]-cost[d-1]) / (cost[d+1]-2*cost[d]+cost[d-1])

// TODO-B: Left-right consistency check
//   Compute disparity from left→right AND right→left
//   Mark pixel as invalid if |disp_lr[x] - disp_rl[x - disp_lr[x]]| > threshold (e.g. 1px)

// TODO-C: Disparity post-processing
//   - Median filter to remove outliers
//   - Hole filling for invalid pixels (interpolate from neighbors)

// TODO-D: Runtime comparison
//   Measure and log time for SAD vs SSD vs NCC vs OpenCV BM vs SGBM
//   Use std::chrono::high_resolution_clock

// ── Public interface ───────────────────────────────────────────
cv::Mat computeDisparity(const cv::Mat& left_rect, const cv::Mat& right_rect,
                          const MatchParams& params) {
    cv::Mat gray_l, gray_r;
    if (left_rect.channels() == 3) {
        cv::cvtColor(left_rect,  gray_l, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_rect, gray_r, cv::COLOR_BGR2GRAY);
    } else {
        gray_l = left_rect;
        gray_r = right_rect;
    }

    switch (params.method) {
    case MatchMethod::OPENCV_BM: {
        auto bm = cv::StereoBM::create(params.num_disparities, params.window_size);
        cv::Mat disp16;
        bm->compute(gray_l, gray_r, disp16);
        cv::Mat disp32;
        disp16.convertTo(disp32, CV_32F, 1.0 / 16.0);
        return disp32;
    }
    case MatchMethod::OPENCV_SGBM: {
        auto sgbm = cv::StereoSGBM::create(
            params.min_disparity, params.num_disparities, params.window_size);
        sgbm->setP1(8  * 3 * params.window_size * params.window_size);
        sgbm->setP2(32 * 3 * params.window_size * params.window_size);
        cv::Mat disp16;
        sgbm->compute(gray_l, gray_r, disp16);
        cv::Mat disp32;
        disp16.convertTo(disp32, CV_32F, 1.0 / 16.0);
        return disp32;
    }
    case MatchMethod::MANUAL_SAD:
        std::cout << "[matching] SAD window=" << params.window_size << "\n";
        return computeSAD(gray_l, gray_r, params);
    case MatchMethod::MANUAL_SSD:
        std::cout << "[matching] SSD window=" << params.window_size << "\n";
        return computeSSD(gray_l, gray_r, params);
    case MatchMethod::MANUAL_NCC:
        std::cout << "[matching] NCC window=" << params.window_size << "\n";
        return computeNCC(gray_l, gray_r, params);
    case MatchMethod::MANUAL_CENSUS:
        std::cout << "[matching] Census window=" << params.window_size << "\n";
        return computeCensus(gray_l, gray_r, params);
    case MatchMethod::MANUAL_SGM:
        std::cout << "[matching] SGM window=" << params.window_size << "\n";
        return computeSGM(gray_l, gray_r, params);
    default:
        throw std::runtime_error("Unknown match method");
    }
}

#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: matching <scene_dir> [sad|ssd|ncc|bm|sgbm]\n";
        return 1;
    }
    fs::path scene(argv[1]);
    std::string method_str = (argc >= 3) ? argv[2] : "sgbm";

    MatchParams params;
    if      (method_str == "sad")  params.method = MatchMethod::MANUAL_SAD;
    else if (method_str == "ssd")  params.method = MatchMethod::MANUAL_SSD;
    else if (method_str == "ncc")  params.method = MatchMethod::MANUAL_NCC;
    else if (method_str == "census") params.method = MatchMethod::MANUAL_CENSUS;
    else if (method_str == "sgm")    params.method = MatchMethod::MANUAL_SGM;
    else if (method_str == "bm")   params.method = MatchMethod::OPENCV_BM;
    else                           params.method = MatchMethod::OPENCV_SGBM;

    CalibData calib = loadCalib(scene);
    params.num_disparities = calib.ndisp;

    cv::Mat left_rect  = cv::imread("results/left_rect.png");
    cv::Mat right_rect = cv::imread("results/right_rect.png");
    if (left_rect.empty()) {
        std::cerr << "Run rectification first.\n"; return 1;
    }

    auto disp = computeDisparity(left_rect, right_rect, params);

    cv::Mat disp_vis;
    cv::normalize(disp, disp_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);
    cv::imwrite("results/disparity.png", disp_vis);
    saveDisparity(disp, "results/disparity_raw.png");
    std::cout << "Saved disparity to results/\n";
    return 0;
}
#endif
