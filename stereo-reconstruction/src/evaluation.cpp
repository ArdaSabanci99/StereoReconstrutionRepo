#include "evaluation.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

// ── PFM loader — Middlebury ground-truth format (Member 4) ────────────────

cv::Mat loadPFM(const fs::path& path) {
    // TODO-E1: Open file in binary mode and read header
    //   std::ifstream f(path, std::ios::binary);
    //   std::string magic;      // "Pf" = grayscale, "PF" = color
    //   int w, h; float scale;
    //   f >> magic >> w >> h >> scale;
    //   f.ignore(1);            // skip the single newline after scale
    //   bool little_endian = (scale < 0);

    // TODO-E2: Read pixel data (stored bottom-row-first, 32-bit float)
    //   cv::Mat img(h, w, CV_32F);
    //   for (int y = h - 1; y >= 0; --y)   // bottom-up storage
    //       f.read(reinterpret_cast<char*>(img.ptr<float>(y)),
    //              w * sizeof(float));

    // TODO-E3: Byte-swap if endianness differs from host
    //   if (little_endian != /* host is LE */ true)
    //       for each float: swap bytes

    // TODO-E4: Return the loaded float map
    //   return img;

    std::cerr << "[evaluation] loadPFM not yet implemented.\n";
    return cv::Mat();
}

// ── Disparity evaluation metrics (Member 4) ───────────────────────────────

EvalResult evaluateDisparity(const cv::Mat& estimated,
                              const cv::Mat& ground_truth,
                              int vmin, int vmax) {
    // TODO-F1: Iterate over all pixels and compute error
    //   double sum_sq = 0, sum_abs = 0;
    //   int bad1 = 0, bad2 = 0, bad4 = 0, valid = 0;
    //
    //   for (int y = 0; y < ground_truth.rows; ++y)
    //       for (int x = 0; x < ground_truth.cols; ++x) {
    //           float gt  = ground_truth.at<float>(y, x);
    //           float est = estimated   .at<float>(y, x);
    //           if (gt  <= 0)               continue;  // invalid GT
    //           if (est < vmin || est > vmax) continue;  // out-of-range
    //           float err = std::abs(est - gt);
    //           if (err > 1.0f) ++bad1;
    //           if (err > 2.0f) ++bad2;
    //           if (err > 4.0f) ++bad4;
    //           sum_sq  += err * err;
    //           sum_abs += err;
    //           ++valid;
    //       }

    // TODO-F2: Normalise and return
    //   EvalResult r;
    //   r.bad1   = 100.0 * bad1 / valid;
    //   r.bad2   = 100.0 * bad2 / valid;
    //   r.bad4   = 100.0 * bad4 / valid;
    //   r.rmse   = std::sqrt(sum_sq  / valid);
    //   r.avgerr = sum_abs / valid;
    //   r.valid  = valid;
    //   return r;

    std::cerr << "[evaluation] evaluateDisparity not yet implemented.\n";
    return EvalResult{};
}

void printEvalResult(const EvalResult& r, const std::string& label) {
    if (!label.empty()) std::cout << "=== " << label << " ===\n";
    std::cout << "  bad1  = " << r.bad1   << " %\n"
              << "  bad2  = " << r.bad2   << " %\n"
              << "  bad4  = " << r.bad4   << " %\n"
              << "  RMSE  = " << r.rmse   << " px\n"
              << "  avg   = " << r.avgerr << " px\n"
              << "  valid = " << r.valid  << " px\n";
}

#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: evaluation <scene_dir> [disparity_raw.png]\n"
                  << "  Ground truth loaded from <scene_dir>/disp0GT.pfm\n";
        return 1;
    }
    fs::path scene(argv[1]);
    std::string disp_path = (argc >= 3) ? argv[2] : "results/disparity_raw.png";

    CalibData calib = loadCalib(scene);
    cv::Mat gt   = loadPFM(scene / "disp0GT.pfm");
    cv::Mat disp = loadDisparity(disp_path);

    if (gt.empty())   { std::cerr << "Ground truth not loaded.\n"; return 1; }
    if (disp.empty()) { std::cerr << "Disparity not loaded.\n";    return 1; }

    EvalResult result = evaluateDisparity(disp, gt, calib.vmin, calib.vmax);
    printEvalResult(result, "Disparity Evaluation");
    return 0;
}
#endif
