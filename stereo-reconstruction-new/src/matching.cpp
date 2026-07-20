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

/**
 * @brief Rounds up a value to the nearest multiple of 16.
 * 
 * @param[in] value 
 * @return int 
 */
static int roundUpToMultipleOf16(int value) {
    value = std::max(16, value);
    return ((value + 15) / 16) * 16;
}


/**
 * @brief Normalizes and validates the matching parameters.
 * 
 * @param params 
 */
void normalizeMatchParams(MatchParams& params) {
    if (params.window_size <= 0 || params.window_size % 2 == 0)
        throw std::invalid_argument("window_size must be a positive odd number");
    if (params.num_disparities <= 0)
        throw std::invalid_argument("num_disparities must be positive");
    if (params.min_disparity < 0)
        throw std::invalid_argument("min_disparity must be non-negative");

    params.num_disparities = roundUpToMultipleOf16(params.num_disparities);
}

/**
 * @brief Sets the disparity range in the matching parameters.
 *        Based on the image scale, the disparity range is scaled to preserve the depth range.
 * 
 * @param[in, out] params 
 * @param[in] scale 
 */
void configureDisparityRangeForScale(MatchParams& params, double scale) {
    if (!(scale > 0.0) || !std::isfinite(scale))
        throw std::invalid_argument("scale must be finite and positive");
    if (params.num_disparities <= 0)
        throw std::invalid_argument("num_disparities must be positive");

    const int reference_min = params.min_disparity;
    const int reference_count = params.num_disparities;

    // params.min_disparity = static_cast<int>(std::floor(reference_min * scale));
    // params.num_disparities = static_cast<int>(std::ceil(reference_count * scale));
    // -- scaling disabled: min_disparity/num_disparities pass through unscaled.

    normalizeMatchParams(params);

    std::cout << "[matching] disparity range: reference=[" << reference_min
              << ", " << (reference_min + reference_count - 1)
              << "], scale=" << scale
              << ", scaled=[" << params.min_disparity
              << ", " << (params.min_disparity + params.num_disparities - 1)
              << "] (" << params.num_disparities << " disparities)\n";
}

/** 
 * @brief Creates a mask whose valid pixels have a fully valid surrounding window.
 * @param half Radius of the required window around each pixel. 
 */
static cv::Mat createWindowValidityMask(const cv::Mat& mask, int half) {
    if (mask.empty() || half <= 0) return mask;
    
    cv::Mat kernel = cv::Mat::ones(2 * half + 1, 2 * half + 1, CV_8U);
    cv::Mat eroded;
    cv::erode(mask, eroded, kernel);
    
    return eroded;
}

/**
 * @brief Uses SAD cost to compute disparity map from rectified stereo images.
 *        Cost: Sum of Absolute Differences over a square window.
 *
 * @param[in] L Left rectified image.
 * @param[in] R Right rectified image.
 * @param[in] p Matching parameters.
 * @param[in] mask1 Left validity mask. Provided by manual rectificaiton only.
 * @param[in] mask2 Right validity mask. Provided by manual rectificaiton only.
 * @return cv::Mat
 */
static cv::Mat computeSAD(const cv::Mat& L, const cv::Mat& R, const MatchParams& p,
                           const cv::Mat& mask1 = cv::Mat(), const cv::Mat& mask2 = cv::Mat()) {
    int half = p.window_size / 2;
    cv::Mat disp(L.size(), CV_32F, cv::Scalar(-1.0f));
    std::vector<float> costs(p.num_disparities);

    cv::Mat m1 = createWindowValidityMask(mask1, half);
    cv::Mat m2 = createWindowValidityMask(mask2, half);
    const bool has_m1 = !m1.empty(), has_m2 = !m2.empty();

    for (int y = half; y < L.rows - half; ++y) {
        for (int x = half; x < L.cols - half; ++x) {
            if (has_m1 && m1.at<uchar>(y, x) == 0) continue; // Window would touch padding

            float best = std::numeric_limits<float>::max();
            int   best_d = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d;
                // Skip candidates whose right-image window would run off the left edge
                // or fall outside valid content, instead of requiring the whole search
                // range to fit before considering the pixel at all.
                if (xr - half < 0 || (has_m2 && m2.at<uchar>(y, xr) == 0)) { costs[d - p.min_disparity] = std::numeric_limits<float>::max(); continue; }
                float cost = 0;
                for (int dy = -half; dy <= half; ++dy)
                    for (int dx = -half; dx <= half; ++dx)
                        cost += std::abs((float)L.at<uchar>(y+dy, x+dx) -
                                         (float)R.at<uchar>(y+dy, xr+dx));
                costs[d - p.min_disparity] = cost;
                if (cost < best) { best = cost; best_d = d; }
            }
            if (best_d < 0) continue; // Every candidate disparity was masked out

            // Sub-pixel refinement (parabola fit)
            if (p.subpixel && best_d > p.min_disparity &&
                best_d < p.min_disparity + p.num_disparities - 1) {
                float cm = costs[best_d - p.min_disparity - 1];
                float c0 = best;
                float cp = costs[best_d - p.min_disparity + 1];
                float denom = cm - 2.0f*c0 + cp;
                // Skip the fit if denom ~ 0 (degenerate parabola) or a neighbour
                if (std::abs(denom) > 1e-5f && cm < std::numeric_limits<float>::max()
                    && cp < std::numeric_limits<float>::max())
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

/**
 * @brief Uses SSD cost to compute disparity map from rectified stereo images.
 *        Cost: Sum of Squared Differences over a square window.
 *
 * @param[in] L Left rectified image.
 * @param[in] R Right rectified image.
 * @param[in] p Matching parameters.
 * @param[in] mask1 Left validity mask. Provided by manual rectificaiton only.
 * @param[in] mask2 Right validity mask. Provided by manual rectificaiton only.
 * @return cv::Mat
 */
static cv::Mat computeSSD(const cv::Mat& L, const cv::Mat& R, const MatchParams& p,
                           const cv::Mat& mask1 = cv::Mat(), const cv::Mat& mask2 = cv::Mat()) {
    int half = p.window_size / 2;
    cv::Mat disp(L.size(), CV_32F, cv::Scalar(-1.0f));

    cv::Mat m1 = createWindowValidityMask(mask1, half);
    cv::Mat m2 = createWindowValidityMask(mask2, half);
    const bool has_m1 = !m1.empty(), has_m2 = !m2.empty();

    for (int y = half; y < L.rows - half; ++y)
        for (int x = half; x < L.cols - half; ++x) {
            if (has_m1 && m1.at<uchar>(y, x) == 0) continue;

            float best = std::numeric_limits<float>::max(); int best_d = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d;
                if (xr - half < 0) continue;
                if (has_m2 && m2.at<uchar>(y, xr) == 0) continue;
                float cost = 0;
                for (int dy = -half; dy <= half; ++dy)
                    for (int dx = -half; dx <= half; ++dx) {
                        float diff = (float)L.at<uchar>(y+dy,x+dx) - (float)R.at<uchar>(y+dy,xr+dx);
                        cost += diff * diff;
                    }
                if (cost < best) { best = cost; best_d = d; }
            }
            if (best_d >= 0) disp.at<float>(y, x) = (float)best_d;
        }
    return disp;
}

/**
 * @brief Computes the disparity map using NCC cost function.
 *        Cost: Normalized Cross-Correlation over a square window.
 *
 * @param L Left rectified image.
 * @param R Right rectified image.
 * @param p Matching parameters.
 * @param mask1 Left validity mask. Provided by manual rectificaiton only.
 * @param mask2 Right validity mask. Provided by manual rectificaiton only.
 * @return cv::Mat
 */
static cv::Mat computeNCC(const cv::Mat& L, const cv::Mat& R, const MatchParams& p,
                           const cv::Mat& mask1 = cv::Mat(), const cv::Mat& mask2 = cv::Mat()) {
    int half = p.window_size / 2;
    int N    = p.window_size * p.window_size;
    cv::Mat disp(L.size(), CV_32F, cv::Scalar(-1.0f));

    cv::Mat m1 = createWindowValidityMask(mask1, half);
    cv::Mat m2 = createWindowValidityMask(mask2, half);
    const bool has_m1 = !m1.empty(), has_m2 = !m2.empty();

    for (int y = half; y < L.rows - half; ++y)
        for (int x = half; x < L.cols - half; ++x) {
            if (has_m1 && m1.at<uchar>(y, x) == 0) continue;

            float best_ncc = -2.0f; int best_d = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d;
                if (xr - half < 0) continue;
                if (has_m2 && m2.at<uchar>(y, xr) == 0) continue;
                float sum_l=0, sum_r=0, sum_l_sq=0, sum_r_sq=0, sum_lr=0;
                for (int dy = -half; dy <= half; ++dy)
                    for (int dx = -half; dx <= half; ++dx) {
                        float l = L.at<uchar>(y+dy,x+dx), r = R.at<uchar>(y+dy,xr+dx);
                        sum_l+=l; sum_r+=r; sum_l_sq+=l*l; sum_r_sq+=r*r; sum_lr+=l*r;
                    }
                float mean_l=sum_l/N, mean_r=sum_r/N;
                float num = sum_lr - N*mean_l*mean_r;
                float den = std::sqrt((sum_l_sq - N*mean_l*mean_l) * (sum_r_sq - N*mean_r*mean_r));
                float ncc = (den < 1e-6f) ? 0.0f : num/den;
                if (ncc > best_ncc) { best_ncc = ncc; best_d = d; }
            }
            if (best_d >= 0) disp.at<float>(y, x) = (float)best_d;
        }
    return disp;
}

/**
 * @brief Computes the Census transform and Hamming distance for stereo matching.
 *        Census transform encodes the local structure of the image into a bit string.
 *        Strings are compared using Hamming distance to compute disparity.
 *
 * @param img Input image.
 * @param half Half window size.
 * @return cv::Mat
 */
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

/**
 * @brief Computes the Hamming distance between two codes.
 * @param a First code.
 * @param b Second code.
 * @return Hamming distance.
 */
static inline int hammingDist(int a, int b) {
    return (int)std::bitset<32>((unsigned)(a ^ b)).count();
}

/**
 * @brief Computes the disparity map using Census cost function.
 *
 * @param L Left rectified image.
 * @param R Right rectified image.
 * @param p Matching parameters.
 * @param mask1 Left validity mask. Provided by manual rectification only.
 * @param mask2 Right validity mask. Provided by manual rectification only.
 * @return cv::Mat
 */
static cv::Mat computeCensus(const cv::Mat& L, const cv::Mat& R, const MatchParams& p,
                              const cv::Mat& mask1 = cv::Mat(), const cv::Mat& mask2 = cv::Mat()) {
    const int ch = 2, ah = p.window_size / 2, brd = ch + ah;
    cv::Mat cL = censusTransform(L, ch), cR = censusTransform(R, ch);
    cv::Mat disp(L.size(), CV_32F, cv::Scalar(-1.0f));

    // Each aggregated descriptor already depends on a Census neighbourhood,
    // so the complete support radius is the sum of both window radii.
    cv::Mat m1 = createWindowValidityMask(mask1, brd);
    cv::Mat m2 = createWindowValidityMask(mask2, brd);
    const bool has_m1 = !m1.empty(), has_m2 = !m2.empty();

    for (int y = brd; y < L.rows - brd; ++y)
        for (int x = brd; x < L.cols - brd; ++x) {
            if (has_m1 && m1.at<uchar>(y, x) == 0) continue;

            float best = std::numeric_limits<float>::max(); int best_d = -1;
            for (int d = p.min_disparity; d < p.min_disparity + p.num_disparities; ++d) {
                int xr = x - d;
                if (xr - brd < 0) continue;
                if (has_m2 && m2.at<uchar>(y, xr) == 0) continue;
                float cost = 0;
                for (int dy = -ah; dy <= ah; ++dy)
                    for (int dx = -ah; dx <= ah; ++dx)
                        cost += hammingDist(cL.at<int>(y+dy,x+dx), cR.at<int>(y+dy,xr+dx));
                if (cost < best) { best = cost; best_d = d; }
            }
            if (best_d >= 0) disp.at<float>(y, x) = (float)best_d;
        }
    return disp;
}


static cv::Mat computeSGM(const cv::Mat& L, const cv::Mat& R, const MatchParams& p,
                           const cv::Mat& mask1 = cv::Mat(), const cv::Mat& mask2 = cv::Mat()) {
    const int rows = L.rows, cols = L.cols;
    const int D = p.num_disparities;
    const int dmin = p.min_disparity;
    const int P1 = p.P1;
    const int P2 = p.P2;
    const int ch = 2;   // Census half-window

    cv::Mat m1 = createWindowValidityMask(mask1, ch);
    cv::Mat m2 = createWindowValidityMask(mask2, ch);
    const bool has_mask1 = !m1.empty();
    const bool has_mask2 = !m2.empty();

    cv::Mat cL = censusTransform(L, ch);
    cv::Mat cR = censusTransform(R, ch);

    const size_t N = (size_t)rows * cols * D;
    std::vector<uint16_t> S(N, 0);

    std::vector<uint16_t> Lr_row(cols * D, 0);
    std::vector<uint16_t> Lr_cur(cols * D, 0);
    std::vector<uint16_t> Lr_prev(D, 0);

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
                const bool off_canvas = (yp < 0 || yp >= rows || xp < 0 || xp >= cols);

                const bool prev_masked = !off_canvas &&
                    ((has_mask1 && m1.at<uchar>(yp, xp) == 0));
                const bool border = off_canvas || prev_masked;

                uint16_t* cur = &Lr_cur[x * D];

                std::vector<uint16_t> c_arr(D);
                const bool left_masked = has_mask1 && m1.at<uchar>(y, x) == 0;
                const bool in_border = left_masked ||
                    (y < ch || y >= rows-ch || x < ch || x >= cols-ch);
                if (in_border) {
                    std::fill(c_arr.begin(), c_arr.end(), (uint16_t)24);
                } else {
                    int cl = cL.at<int>(y, x);
                    for (int d = 0; d < D; ++d) {
                        int xr = x - (d + dmin);
                        const bool right_masked = has_mask2 &&
                            (xr < 0 || xr >= cols || m2.at<uchar>(y, xr) == 0);
                        c_arr[d] = (right_masked || xr < ch || xr >= cols - ch)
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

    //  WTA + sub-pixel
    cv::Mat disp(rows, cols, CV_32F, cv::Scalar(-1.0f));
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            if (has_mask1 && m1.at<uchar>(y, x) == 0) continue;

            uint16_t* s = &S[(size_t)(y*cols+x)*D];
            int best_d = -1;
            uint16_t best_cost = std::numeric_limits<uint16_t>::max();

            for (int di = 0; di < D; ++di) {
                const int xr = x - (di + dmin);
                const bool valid = xr >= ch && xr < cols - ch &&
                    (!has_mask2 || m2.at<uchar>(y, xr) != 0);
                if (!valid) continue;
                const uint16_t c = s[di];
                if (c < best_cost) {
                    best_cost = c;
                    best_d = di;
                }
            }
            if (best_d < 0) continue;

            float disp_val = (float)(best_d + dmin);
            if (p.subpixel && best_d > 0 && best_d < D-1) {
                const int xrPrev = x - ((best_d - 1) + dmin);
                const int xrNext = x - ((best_d + 1) + dmin);
                const bool neighboursValid =
                    xrPrev >= ch && xrPrev < cols - ch &&
                    xrNext >= ch && xrNext < cols - ch &&
                    (!has_mask2 || (m2.at<uchar>(y, xrPrev) != 0 &&
                                    m2.at<uchar>(y, xrNext) != 0));
                if (!neighboursValid) {
                    disp.at<float>(y, x) = disp_val;
                    continue;
                }
                float cm = s[best_d-1], c0 = s[best_d], cp = s[best_d+1];
                float denom = cm - 2.0f*c0 + cp;
                if (std::abs(denom) > 1e-3f)
                    disp_val = (float)(best_d + dmin) - 0.5f*(cp-cm)/denom;
            }
            disp.at<float>(y, x) = disp_val;
        }
    return disp;
}


/** 
 * @brief Invalidates inconsistent pixels in a dense left-to-right disparity map.
 * @param threshold Maximum allowed disparity difference in pixels. 
 */
static void applyDenseLeftRightConsistencyCheck(cv::Mat& disp_lr, const cv::Mat& disp_rl,
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

/**
 * @brief Invalidate disparities at pixels whose left-image window touched padding.
 *
 * Applied after matching regardless of method, as a catch-all in case a
 * cost function's own masking missed a pixel upstream. mask1 is only
 * non-empty after manual (Loop & Zhang) rectification; a no-op otherwise.
 */
static void applyValidityMask(cv::Mat& disp, const cv::Mat& mask1) {
    if (mask1.empty()) return;
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
            if (mask1.at<uchar>(y, x) == 0)
                disp.at<float>(y, x) = -1.0f;
}

/**
 * @brief Remove disparities whose matched point is outside valid right-image content.
 */
static void applyRightValidityMask(cv::Mat& disp, const cv::Mat& mask2) {
    if (mask2.empty()) return;

    for (int y = 0; y < disp.rows; ++y) {
        for (int x = 0; x < disp.cols; ++x) {
            const float d = disp.at<float>(y, x);
            if (d < 0.0f) continue;

            const int xr = cvRound(static_cast<float>(x) - d);
            if (xr < 0 || xr >= disp.cols || mask2.at<uchar>(y, xr) == 0)
                disp.at<float>(y, x) = -1.0f;
        }
    }
}

/**
 * @brief Computes the right-to-left disparity map by flipping the images and calling computeDisparity.
 * 
 * @param L[in] Left rectified image.
 * @param R[in] Right rectified image.
 * @param p[in] Matching parameters.
 * @param mask1 Validity mask for the left image.  
 * @param mask2 Validity mask for the right image.
 * @return cv::Mat 
 */
static cv::Mat computeDisparityRL(const cv::Mat& L, const cv::Mat& R,
                                   const MatchParams& p,
                                   const cv::Mat& mask1, const cv::Mat& mask2) {
    MatchParams pr = p;
    pr.lr_check      = false;
    pr.median_filter = false;

    cv::Mat Lf, Rf;
    cv::flip(L, Lf, 1);
    cv::flip(R, Rf, 1);

    // Flip the masks the same way as the images
    cv::Mat mask1f, mask2f;
    if (!mask1.empty()) cv::flip(mask1, mask1f, 1);
    if (!mask2.empty()) cv::flip(mask2, mask2f, 1);

    cv::Mat disp_rl = computeDisparity(Rf, Lf, pr, mask2f, mask1f);

    cv::Mat disp_rl_flipped;
    cv::flip(disp_rl, disp_rl_flipped, 1);
    return disp_rl_flipped;
}

/**
 * @brief Applies a 3x3 median filter to the disparity map.
 *        Fills invalid pixels by interpolating from the nearest valid neighbors horizontally.
 * 
 * @param[in,out] disp Disparity map to be filtered and filled. 
 */
static void medianAndFill(cv::Mat& disp) {
    // Median filter (3×3) on valid pixels only
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

    // Fill holes by horizontal interpolation from nearest valid neighbours
    for (int y = 0; y < tmp.rows; ++y) {
        for (int x = 0; x < tmp.cols; ++x) {
            if (tmp.at<float>(y,x) >= 0) continue;
            // Find left valid
            float lv = -1; for (int xx = x-1; xx >= 0; --xx)
                if (tmp.at<float>(y,xx) >= 0) { lv = tmp.at<float>(y,xx); break; }
            // Find right valid
            float rv = -1; for (int xx = x+1; xx < tmp.cols; ++xx)
                if (tmp.at<float>(y,xx) >= 0) { rv = tmp.at<float>(y,xx); break; }
            if      (lv >= 0 && rv >= 0) tmp.at<float>(y,x) = std::min(lv, rv);
            else if (lv >= 0)            tmp.at<float>(y,x) = lv;
            else if (rv >= 0)            tmp.at<float>(y,x) = rv;
        }
    }
    disp = tmp;
}

/**
 * @brief Computes the disparity map for two rectified images.
 * 
 * @param[in] left_rect Left rectified image.
 * @param[in] right_rect Right rectified image.
 * @param[in] params Matching parameters.
 * @param[in] mask1 Left image mask.
 * @param[in] mask2 Right image mask.
 * @return cv::Mat 
 */
cv::Mat computeDisparity(const cv::Mat& left_rect, const cv::Mat& right_rect,
                          const MatchParams& params,
                          const cv::Mat& mask1, const cv::Mat& mask2) {
    MatchParams effective = params;
    normalizeMatchParams(effective);
    std::cout << "[matching] search interval = [" << effective.min_disparity
              << ", " << (effective.min_disparity + effective.num_disparities - 1)
              << "] (" << effective.num_disparities << " disparities)\n";

    cv::Mat gL, gR;
    if (left_rect.channels()  == 3) cv::cvtColor(left_rect,  gL, cv::COLOR_BGR2GRAY);
    else gL = left_rect;
    if (right_rect.channels() == 3) cv::cvtColor(right_rect, gR, cv::COLOR_BGR2GRAY);
    else gR = right_rect;

    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat disp;
    switch (effective.method) {
        case MatchMethod::MANUAL_SAD:    disp = computeSAD   (gL, gR, effective, mask1, mask2); break;
        case MatchMethod::MANUAL_SSD:    disp = computeSSD   (gL, gR, effective, mask1, mask2); break;
        case MatchMethod::MANUAL_NCC:    disp = computeNCC   (gL, gR, effective, mask1, mask2); break;
        case MatchMethod::MANUAL_CENSUS: disp = computeCensus(gL, gR, effective, mask1, mask2); break;
        case MatchMethod::MANUAL_SGM:    disp = computeSGM   (gL, gR, effective, mask1, mask2); break;
        case MatchMethod::OPENCV_BM: {
            auto bm = cv::StereoBM::create(effective.num_disparities, effective.window_size);
            bm->setMinDisparity(effective.min_disparity);
            cv::Mat d16; bm->compute(gL, gR, d16);
            d16.convertTo(disp, CV_32F, 1.0/16.0); break;
        }
        case MatchMethod::OPENCV_SGBM: {
            auto sgbm = cv::StereoSGBM::create(effective.min_disparity, effective.num_disparities, effective.window_size);
            sgbm->setP1(8  * 3 * effective.window_size * effective.window_size);
            sgbm->setP2(32 * 3 * effective.window_size * effective.window_size);
            if (effective.lr_check) sgbm->setDisp12MaxDiff(1);
            cv::Mat d16; sgbm->compute(gL, gR, d16);
            // Remove small disconnected speckles (streak outliers at depth boundaries)
            cv::filterSpeckles(d16, -16 * effective.num_disparities, 400, effective.num_disparities * 16);
            d16.convertTo(disp, CV_32F, 1.0/16.0); break;
        }
        default: throw std::runtime_error("Unknown match method");
    }

    if (effective.method == MatchMethod::OPENCV_BM ||
        effective.method == MatchMethod::OPENCV_SGBM) {
        cv::Mat invalid = disp <= static_cast<float>(effective.min_disparity - 1);
        disp.setTo(-1.0f, invalid);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[matching] time = " << ms << " ms\n";

    applyValidityMask(disp, mask1);
    applyRightValidityMask(disp, mask2);

    if (effective.lr_check &&
        effective.method != MatchMethod::OPENCV_BM &&
        effective.method != MatchMethod::OPENCV_SGBM) {
        
            cv::Mat disp_rl = computeDisparityRL(gL, gR, effective, mask1, mask2);
        applyDenseLeftRightConsistencyCheck(disp, disp_rl);
        std::cout << "[matching] L-R check done.\n";
    }

    if (effective.median_filter) {
        medianAndFill(disp);
        std::cout << "[matching] Median + hole fill done.\n";
    }

    applyValidityMask(disp, mask1);
    applyRightValidityMask(disp, mask2);

    return disp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone executable
// ─────────────────────────────────────────────────────────────────────────────
#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: matching <scene_id> <left_id> <right_id> [method] "
                     "[--min-disp D] [--ndisp N] [--window W] [--tag T]\n"
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
    if (method_str == "sad") params.method = MatchMethod::MANUAL_SAD;
    else if (method_str == "ssd") params.method = MatchMethod::MANUAL_SSD;
    else if (method_str == "ncc") params.method = MatchMethod::MANUAL_NCC;
    else if (method_str == "census") params.method = MatchMethod::MANUAL_CENSUS;
    else if (method_str == "sgm") params.method = MatchMethod::MANUAL_SGM;
    else if (method_str == "bm") params.method = MatchMethod::OPENCV_BM;
    else params.method = MatchMethod::OPENCV_SGBM;

    std::string tag = "opencv";
    for (int i = 5; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--ndisp"   && i+1 < argc) params.num_disparities = std::stoi(argv[++i]);
        else if (a == "--min-disp" && i+1 < argc) params.min_disparity = std::stoi(argv[++i]);
        else if (a == "--window" && i+1 < argc) params.window_size = std::stoi(argv[++i]);
        else if (a == "--tag"&& i+1 < argc) tag = argv[++i];
    }

    std::string load_path = "results/scene" + sceneId + "/rectification/" + tag;
    cv::Mat lRect = cv::imread(load_path + "/view_" + viewL + ".png");
    cv::Mat rRect = cv::imread(load_path + "/view_" + viewR + ".png");
    if (lRect.empty() || rRect.empty()) {
        std::cerr << "Run rectification first (tag=" << tag << "): " << load_path << "\n";
        return 1;
    }

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

    try {
        normalizeMatchParams(params);
    } catch (const std::exception& e) {
        std::cerr << "Invalid matching parameters: " << e.what() << "\n";
        return 1;
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
