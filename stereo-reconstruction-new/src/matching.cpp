#include "matching.h"
#include "utils.h"
#include <iostream>
#include <filesystem>
#include <bitset>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// NOTE: computeDisparity takes optional mask1/mask2 (CV_8U, 255 = valid
// content, 0 = black warp padding, same contract as RectifyResult::mask1/
// mask2 from rectification.cpp) for backward compatibility with callers,
// but the padding-invalidation logic that used to consume them is
// currently disabled -- see the comments at computeSGM, computeDisparityRL,
// and computeDisparity below.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// SAD
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat computeSAD(const cv::Mat& L, const cv::Mat& R, const MatchParams& p) {
    int half = p.window_size / 2;
    cv::Mat disp(L.size(), CV_32F, cv::Scalar(-1.0f));
    std::vector<float> costs(p.num_disparities);

    for (int y = half; y < L.rows - half; ++y) {
        for (int x = half + p.min_disparity + p.num_disparities;
             x < L.cols - half; ++x) {
            float best = std::numeric_limits<float>::max();
            int   best_d = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d;
                if (xr - half < 0) break;
                float cost = 0;
                for (int dy = -half; dy <= half; ++dy)
                    for (int dx = -half; dx <= half; ++dx)
                        cost += std::abs((float)L.at<uchar>(y+dy, x+dx) -
                                         (float)R.at<uchar>(y+dy, xr+dx));
                costs[d - p.min_disparity] = cost;
                if (cost < best) { best = cost; best_d = d; }
            }
            // Sub-pixel refinement (parabola fit)
            if (p.subpixel && best_d > p.min_disparity &&
                best_d < p.min_disparity + p.num_disparities - 1) {
                float cm = costs[best_d - p.min_disparity - 1];
                float c0 = best;
                float cp = costs[best_d - p.min_disparity + 1];
                float denom = cm - 2.0f*c0 + cp;
                if (std::abs(denom) > 1e-5f)
                    disp.at<float>(y, x) = best_d - 0.5f * (cp - cm) / denom;
                else
                    disp.at<float>(y, x) = (float)best_d;
            } else {
                disp.at<float>(y, x) = (float)best_d;
            }
        }
    }
    return disp;
}

// ─────────────────────────────────────────────────────────────────────────────
// SSD
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat computeSSD(const cv::Mat& L, const cv::Mat& R, const MatchParams& p) {
    int half = p.window_size / 2;
    cv::Mat disp(L.size(), CV_32F, cv::Scalar(-1.0f));

    for (int y = half; y < L.rows - half; ++y)
        for (int x = half + p.min_disparity + p.num_disparities; x < L.cols - half; ++x) {
            float best = std::numeric_limits<float>::max(); int best_d = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d; if (xr - half < 0) break;
                float cost = 0;
                for (int dy = -half; dy <= half; ++dy)
                    for (int dx = -half; dx <= half; ++dx) {
                        float diff = (float)L.at<uchar>(y+dy,x+dx) - (float)R.at<uchar>(y+dy,xr+dx);
                        cost += diff * diff;
                    }
                if (cost < best) { best = cost; best_d = d; }
            }
            disp.at<float>(y, x) = (float)best_d;
        }
    return disp;
}

// ─────────────────────────────────────────────────────────────────────────────
// NCC
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat computeNCC(const cv::Mat& L, const cv::Mat& R, const MatchParams& p) {
    int half = p.window_size / 2;
    int N    = p.window_size * p.window_size;
    cv::Mat disp(L.size(), CV_32F, cv::Scalar(-1.0f));

    for (int y = half; y < L.rows - half; ++y)
        for (int x = half + p.min_disparity + p.num_disparities; x < L.cols - half; ++x) {
            float best_ncc = -2.0f; int best_d = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d; if (xr - half < 0) break;
                float sL=0,sR=0,sLL=0,sRR=0,sLR=0;
                for (int dy = -half; dy <= half; ++dy)
                    for (int dx = -half; dx <= half; ++dx) {
                        float l = L.at<uchar>(y+dy,x+dx), r = R.at<uchar>(y+dy,xr+dx);
                        sL+=l; sR+=r; sLL+=l*l; sRR+=r*r; sLR+=l*r;
                    }
                float mL=sL/N, mR=sR/N;
                float num = sLR - N*mL*mR;
                float den = std::sqrt((sLL - N*mL*mL) * (sRR - N*mR*mR));
                float ncc = (den < 1e-6f) ? 0.0f : num/den;
                if (ncc > best_ncc) { best_ncc = ncc; best_d = d; }
            }
            disp.at<float>(y, x) = (float)best_d;
        }
    return disp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Census transform + Hamming cost  (Zabih & Woodfill 1994)
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat censusTransform(const cv::Mat& img, int half = 2) {
    cv::Mat cen(img.size(), CV_32S, cv::Scalar(0));
    for (int y = half; y < img.rows - half; ++y)
        for (int x = half; x < img.cols - half; ++x) {
            uint32_t code = 0; int bit = 0;
            uchar centre = img.at<uchar>(y, x);
            for (int dy = -half; dy <= half; ++dy)
                for (int dx = -half; dx <= half; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (img.at<uchar>(y+dy, x+dx) < centre) code |= (1u << bit);
                    ++bit;
                }
            cen.at<int>(y, x) = (int)code;
        }
    return cen;
}

static inline int hammingDist(int a, int b) {
    return (int)std::bitset<32>((unsigned)(a ^ b)).count();
}

static cv::Mat computeCensus(const cv::Mat& L, const cv::Mat& R, const MatchParams& p) {
    const int ch = 2, ah = p.window_size / 2, brd = ch + ah;
    cv::Mat cL = censusTransform(L, ch), cR = censusTransform(R, ch);
    cv::Mat disp(L.size(), CV_32F, cv::Scalar(-1.0f));

    for (int y = brd; y < L.rows - brd; ++y)
        for (int x = brd + p.min_disparity + p.num_disparities; x < L.cols - brd; ++x) {
            float best = std::numeric_limits<float>::max(); int best_d = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d; if (xr - brd < 0) break;
                float cost = 0;
                for (int dy = -ah; dy <= ah; ++dy)
                    for (int dx = -ah; dx <= ah; ++dx)
                        cost += hammingDist(cL.at<int>(y+dy,x+dx), cR.at<int>(y+dy,xr+dx));
                if (cost < best) { best = cost; best_d = d; }
            }
            disp.at<float>(y, x) = (float)best_d;
        }
    return disp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Semi-Global Matching  (Hirschmüller 2008)   — C1 challenge
// Uses Census cost volume + 8-direction path aggregation.
//
// mask1/mask2 are accepted for signature compatibility but currently
// unused -- the mask-based padding invalidation this was for (forcing
// pixels under black warp-padding to the same sentinel cost as genuine
// border pixels, so the P1/P2 smoothness aggregation can't drag a
// degenerate zero-Hamming-cost padding match into real neighbouring
// pixels) is disabled. See the commented-out logic further down.
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat computeSGM(const cv::Mat& L, const cv::Mat& R, const MatchParams& p,
                           const cv::Mat& mask1 = cv::Mat(), const cv::Mat& mask2 = cv::Mat()) {
    const int rows = L.rows, cols = L.cols;
    const int D    = p.num_disparities;
    const int dmin = p.min_disparity;
    const int P1   = p.P1;
    const int P2   = p.P2;
    const int ch   = 2;   // census half-window
    // Mask-based padding invalidation disabled, see computeDisparity.
    // const bool has_mask1 = !mask1.empty();
    // const bool has_mask2 = !mask2.empty();

    // Census images (small: rows*cols*4 bytes each)
    cv::Mat cL = censusTransform(L, ch);
    cv::Mat cR = censusTransform(R, ch);

    // ── Aggregated cost  S[y*cols*D + x*D + d]  (uint16_t to halve memory)
    // Max per-direction path cost ≤ 24 + P2 ≈ 144; × 8 dirs ≤ 1152 < 65535.
    const size_t N = (size_t)rows * cols * D;
    std::vector<uint16_t> S(N, 0);   // ~117 MB at 800×600×128

    // ── Per-direction path buffers (cols×D each, tiny compared to S) ──────
    std::vector<uint16_t> Lr_row(cols * D, 0);
    std::vector<uint16_t> Lr_cur(cols * D, 0);
    std::vector<uint16_t> Lr_prev(D, 0);        // for horizontal passes

    const int dy8[] = {0, 0,  1, -1,  1, -1,  1, -1};
    const int dx8[] = {1,-1,  0,  0,  1, -1, -1,  1};

    for (int dir = 0; dir < 8; ++dir) {
        const int dy = dy8[dir], dx = dx8[dir];
        const bool horiz = (dy == 0);

        const int y0 = (dy >= 0) ? 0 : rows-1, y1 = (dy >= 0) ? rows : -1, ys = (dy >= 0) ? 1 : -1;
        const int x0 = (dx >= 0) ? 0 : cols-1, x1 = (dx >= 0) ? cols : -1, xs = (dx >= 0) ? 1 : -1;

        std::fill(Lr_row.begin(), Lr_row.end(), 0);

        for (int y = y0; y != y1; y += ys) {
            std::fill(Lr_cur.begin(), Lr_cur.end(), 0);
            std::fill(Lr_prev.begin(), Lr_prev.end(), 0);

            for (int x = x0; x != x1; x += xs) {
                const int yp = y - dy, xp = x - dx;
                const bool border = (yp < 0 || yp >= rows || xp < 0 || xp >= cols);

                uint16_t* cur = &Lr_cur[x * D];

                // ── Census cost for this pixel (computed on the fly) ──────
                uint16_t c_arr[256];   // D ≤ 256 guaranteed by sane usage
                const bool in_border = (y < ch || y >= rows-ch || x < ch || x >= cols-ch);
                // Mask-based padding invalidation disabled: was
                // `const bool left_masked = has_mask1 && mask1.at<uchar>(y, x) == 0;`
                // added to `in_border` below, plus a `right_masked` check on
                // mask2 inside the d-loop.
                if (in_border) {
                    std::fill(c_arr, c_arr + D, (uint16_t)24);
                } else {
                    int cl = cL.at<int>(y, x);
                    for (int d = 0; d < D; ++d) {
                        int xr = x - (d + dmin);
                        c_arr[d] = (xr < ch || xr >= cols - ch)
                                   ? (uint16_t)24
                                   : (uint16_t)hammingDist(cl, cR.at<int>(y, xr));
                    }
                }

                if (border) {
                    for (int d = 0; d < D; ++d) cur[d] = c_arr[d];
                } else {
                    uint16_t* prev = horiz ? Lr_prev.data() : &Lr_row[xp * D];
                    uint16_t min_prev = *std::min_element(prev, prev + D);
                    for (int d = 0; d < D; ++d) {
                        uint16_t v0 = prev[d];
                        uint16_t v1 = (d > 0)   ? (uint16_t)(prev[d-1] + P1) : 65535u;
                        uint16_t v2 = (d < D-1) ? (uint16_t)(prev[d+1] + P1) : 65535u;
                        uint16_t v3 = (uint16_t)(min_prev + P2);
                        cur[d] = (uint16_t)(c_arr[d] + std::min({v0,v1,v2,v3}) - min_prev);
                    }
                }

                if (horiz) std::copy(cur, cur + D, Lr_prev.begin());

                uint16_t* s = &S[(size_t)(y*cols+x)*D];
                for (int d = 0; d < D; ++d) s[d] += cur[d];
            }
            std::swap(Lr_row, Lr_cur);
        }
    }

    // ── WTA + sub-pixel ──────────────────────────────────────────────────
    cv::Mat disp(rows, cols, CV_32F, cv::Scalar(-1.0f));
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            uint16_t* s = &S[(size_t)(y*cols+x)*D];
            int best_d = (int)(std::min_element(s, s+D) - s);
            float disp_val = (float)(best_d + dmin);
            if (p.subpixel && best_d > 0 && best_d < D-1) {
                float cm = s[best_d-1], c0 = s[best_d], cp = s[best_d+1];
                float denom = cm - 2.0f*c0 + cp;
                if (std::abs(denom) > 1e-3f)
                    disp_val = (float)(best_d + dmin) - 0.5f*(cp-cm)/denom;
            }
            disp.at<float>(y, x) = disp_val;
        }
    return disp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Left-Right consistency check
// ─────────────────────────────────────────────────────────────────────────────
static void lrConsistencyCheck(cv::Mat& disp_lr, const cv::Mat& disp_rl,
                                 float threshold = 1.0f) {
    for (int y = 0; y < disp_lr.rows; ++y)
        for (int x = 0; x < disp_lr.cols; ++x) {
            float d = disp_lr.at<float>(y, x);
            if (d < 0) continue;
            int xr = x - (int)std::round(d);
            if (xr < 0 || xr >= disp_rl.cols) {
                disp_lr.at<float>(y, x) = -1.0f;
                continue;
            }
            float dr = disp_rl.at<float>(y, xr);
            if (dr < 0 || std::abs(d - dr) > threshold)
                disp_lr.at<float>(y, x) = -1.0f;
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mask-based padding invalidation disabled (see computeDisparity). Was:
//
// static void applyValidityMask(cv::Mat& disp, const cv::Mat& mask1) {
//     if (mask1.empty()) return;
//     for (int y = 0; y < disp.rows; ++y)
//         for (int x = 0; x < disp.cols; ++x)
//             if (mask1.at<uchar>(y, x) == 0)
//                 disp.at<float>(y, x) = -1.0f;
// }
// ─────────────────────────────────────────────────────────────────────────────

static cv::Mat computeDisparityRL(const cv::Mat& L, const cv::Mat& R,
                                   const MatchParams& p,
                                   const cv::Mat& mask1, const cv::Mat& mask2) {
    MatchParams pr = p;
    pr.lr_check      = false;
    pr.median_filter = false;

    cv::Mat Lf, Rf;
    cv::flip(L, Lf, 1);
    cv::flip(R, Rf, 1);

    // Mask-based padding invalidation disabled: this used to also flip
    // mask1/mask2 and pass them through (roles swapped: Rf plays "left"
    // here, so it took the flipped RIGHT mask, and Lf took the left mask).
    cv::Mat disp_rl = computeDisparity(Rf, Lf, pr);

    cv::Mat disp_rl_flipped;
    cv::flip(disp_rl, disp_rl_flipped, 1);
    return disp_rl_flipped;
}

// ─────────────────────────────────────────────────────────────────────────────
// Median filter + hole filling
// ─────────────────────────────────────────────────────────────────────────────
static void medianAndFill(cv::Mat& disp) {
    // 1. Median filter (3×3) on valid pixels only
    cv::Mat tmp = disp.clone();
    for (int y = 1; y < disp.rows-1; ++y)
        for (int x = 1; x < disp.cols-1; ++x) {
            if (disp.at<float>(y,x) < 0) continue;
            std::vector<float> v;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    float val = disp.at<float>(y+dy, x+dx);
                    if (val >= 0) v.push_back(val);
                }
            if (!v.empty()) {
                std::nth_element(v.begin(), v.begin()+v.size()/2, v.end());
                tmp.at<float>(y,x) = v[v.size()/2];
            }
        }

    // 2. Fill holes by horizontal interpolation from nearest valid neighbours
    for (int y = 0; y < tmp.rows; ++y) {
        for (int x = 0; x < tmp.cols; ++x) {
            if (tmp.at<float>(y,x) >= 0) continue;
            // find left valid
            float lv = -1; for (int xx = x-1; xx >= 0; --xx)
                if (tmp.at<float>(y,xx) >= 0) { lv = tmp.at<float>(y,xx); break; }
            // find right valid
            float rv = -1; for (int xx = x+1; xx < tmp.cols; ++xx)
                if (tmp.at<float>(y,xx) >= 0) { rv = tmp.at<float>(y,xx); break; }
            if      (lv >= 0 && rv >= 0) tmp.at<float>(y,x) = std::min(lv, rv);
            else if (lv >= 0)            tmp.at<float>(y,x) = lv;
            else if (rv >= 0)            tmp.at<float>(y,x) = rv;
        }
    }
    disp = tmp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public interface
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat computeDisparity(const cv::Mat& left_rect, const cv::Mat& right_rect,
                          const MatchParams& params,
                          const cv::Mat& mask1, const cv::Mat& mask2) {
    cv::Mat gL, gR;
    if (left_rect.channels()  == 3) cv::cvtColor(left_rect,  gL, cv::COLOR_BGR2GRAY);
    else gL = left_rect;
    if (right_rect.channels() == 3) cv::cvtColor(right_rect, gR, cv::COLOR_BGR2GRAY);
    else gR = right_rect;

    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat disp;
    switch (params.method) {
        case MatchMethod::MANUAL_SAD:    disp = computeSAD   (gL, gR, params); break;
        case MatchMethod::MANUAL_SSD:    disp = computeSSD   (gL, gR, params); break;
        case MatchMethod::MANUAL_NCC:    disp = computeNCC   (gL, gR, params); break;
        case MatchMethod::MANUAL_CENSUS: disp = computeCensus(gL, gR, params); break;
        case MatchMethod::MANUAL_SGM:    disp = computeSGM   (gL, gR, params, mask1, mask2); break;
        case MatchMethod::OPENCV_BM: {
            auto bm = cv::StereoBM::create(params.num_disparities, params.window_size);
            cv::Mat d16; bm->compute(gL, gR, d16);
            d16.convertTo(disp, CV_32F, 1.0/16.0); break;
        }
        case MatchMethod::OPENCV_SGBM: {
            auto sgbm = cv::StereoSGBM::create(params.min_disparity, params.num_disparities, params.window_size);
            sgbm->setP1(8  * 3 * params.window_size * params.window_size);
            sgbm->setP2(32 * 3 * params.window_size * params.window_size);
            if (params.lr_check) sgbm->setDisp12MaxDiff(1);
            cv::Mat d16; sgbm->compute(gL, gR, d16);
            // Remove small disconnected speckles (streak outliers at depth boundaries)
            cv::filterSpeckles(d16, -16 * params.num_disparities, 400, params.num_disparities * 16);
            d16.convertTo(disp, CV_32F, 1.0/16.0); break;
        }
        default: throw std::runtime_error("Unknown match method");
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[matching] time = " << ms << " ms\n";

    // Mask-based padding invalidation disabled: this used to call
    // applyValidityMask(disp, mask1) here (before post-processing) and
    // again after, to keep black warp-padding out of the disparity map.

    // ── Post-processing ──────────────────────────────────────────────────
    if (params.lr_check &&
        params.method != MatchMethod::OPENCV_BM &&
        params.method != MatchMethod::OPENCV_SGBM) {
        cv::Mat disp_rl = computeDisparityRL(gL, gR, params, mask1, mask2);
        lrConsistencyCheck(disp, disp_rl);
        std::cout << "[matching] L-R check done.\n";
    }
    if (params.median_filter) {
        medianAndFill(disp);
        std::cout << "[matching] Median + hole fill done.\n";
    }

    return disp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone executable
// ─────────────────────────────────────────────────────────────────────────────
#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: matching <scene_id> <left_id> <right_id> [method] "
                     "[--ndisp N] [--window W] [--tag T]\n"
                     "  methods: sad ssd ncc census sgm bm sgbm\n"
                     "  --tag T: which rectification output to read "
                     "(opencv | loopzhang | loopzhang_correct; default opencv)\n";
        return 1;
    }
    const std::string sceneId(argv[1]);
    const std::string viewL = padViewId(std::stoi(argv[2]));
    const std::string viewR = padViewId(std::stoi(argv[3]));
    std::string method_str = (argc >= 5) ? argv[4] : "sgm";

    MatchParams params;
    if      (method_str == "sad")   params.method = MatchMethod::MANUAL_SAD;
    else if (method_str == "ssd")   params.method = MatchMethod::MANUAL_SSD;
    else if (method_str == "ncc")   params.method = MatchMethod::MANUAL_NCC;
    else if (method_str == "census")params.method = MatchMethod::MANUAL_CENSUS;
    else if (method_str == "sgm")   params.method = MatchMethod::MANUAL_SGM;
    else if (method_str == "bm")    params.method = MatchMethod::OPENCV_BM;
    else                            params.method = MatchMethod::OPENCV_SGBM;

    std::string tag = "opencv";
    for (int i = 5; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--ndisp"  && i+1 < argc) params.num_disparities = std::stoi(argv[++i]);
        else if (a == "--window" && i+1 < argc) params.window_size     = std::stoi(argv[++i]);
        else if (a == "--tag"    && i+1 < argc) tag                    = argv[++i];
    }

    // NOTE: this used to read from "results/scene<id>/rectification" directly
    // -- rectification.cpp now writes into a tagged subdirectory
    // ("rectification/opencv/", "rectification/loopzhang/", etc.), so that
    // flat path either fails to find images or picks up stale ones left
    // over from before the tag subdirectories existed. Fixed below.
    std::string load_path = "results/scene" + sceneId + "/rectification/" + tag;
    cv::Mat lRect = cv::imread(load_path + "/view_" + viewL + ".png");
    cv::Mat rRect = cv::imread(load_path + "/view_" + viewR + ".png");
    if (lRect.empty() || rRect.empty()) {
        std::cerr << "Run rectification first (tag=" << tag << "): " << load_path << "\n";
        return 1;
    }

    // Validity masks written by rectification.cpp alongside the rectified
    // images -- see the padding/SGM-noise discussion this fix addresses.
    cv::Mat mask1 = cv::imread(load_path + "/mask_" + viewL + ".png", cv::IMREAD_GRAYSCALE);
    cv::Mat mask2 = cv::imread(load_path + "/mask_" + viewR + ".png", cv::IMREAD_GRAYSCALE);
    if (mask1.empty() || mask2.empty())
        std::cout << "[matching] WARNING: no validity masks found at " << load_path
                   << " -- proceeding without padding masking.\n";

    std::string calib_path = "results/scene" + sceneId + "/sparse_matching/calib_" + viewL + "_" + viewR + ".yaml";
    CalibData calib = loadCalibData(calib_path);
    if (calib.swapped) {
        std::cout << "[matching] L/R swapped during sparse matching — swapping rectified images.\n";
        std::swap(lRect, rRect);
        std::swap(mask1, mask2);
    }

    cv::Mat disp = computeDisparity(lRect, rRect, params, mask1, mask2);

    cv::Mat vis; cv::normalize(disp, vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(vis, vis, cv::COLORMAP_JET);

    std::string save = "results/scene" + sceneId + "/matching";
    fs::create_directories(save);
    cv::imwrite(save + "/view_" + viewL + "_" + viewR + "_" + method_str + ".png", vis);
    saveDisparity(disp, save + "/view_" + viewL + "_" + viewR + "_" + method_str + "_raw.png");
    std::cout << "Saved to " << save << "\n";
    return 0;
}
#endif