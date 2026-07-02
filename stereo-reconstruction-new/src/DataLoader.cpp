#include "DataLoader.h"
#include <fstream>
#include <stdexcept>

constexpr double detTolerance = 1e-6;

DTUDataLoader::DTUDataLoader(const std::string & data_path) 
    : m_data_path(data_path) {}

cv::Mat DTUDataLoader::loadImage(const std::string & sceneId, const std::string & viewId, const std::string & lightId) {
    std::string img_path = m_data_path + "/Rectified/scan" + sceneId 
    + "/rect_" + viewId + "_" + lightId + "_r5000.png";

    cv::Mat img = cv::imread(img_path);

    return img;
}


cv::Mat DTUDataLoader::loadCameraProjection(const std::string & viewId) {
    std::string camera_proj_path = m_data_path + "/Calibration/cal18/pos_" + viewId + ".txt";

    std::ifstream file(camera_proj_path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open: " + camera_proj_path);

    // load projection matrix
    cv::Mat projMat(3, 4, CV_64F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {

            double val;
            if (!(file >> val))
                throw std::runtime_error("Unexpected EOF in: " + camera_proj_path);

            projMat.at<double>(i, j) = val;
        }
    }

    return projMat;

}

std::tuple<cv::Mat, cv::Mat, cv::Mat> DTUDataLoader::decomposeProjectionMatrix(const std::string & viewId) {
    cv::Mat projMat = loadCameraProjection(viewId); // no sceneId required (fixed position across scenes)

    // QR decomposition to get intrinsics
    cv::Mat K, R, Qx, Qy, Qz;

    cv::RQDecomp3x3(projMat.colRange(0, 3), K, R, Qx, Qy, Qz);
    K /= K.at<double>(2, 2); // normalising, so at (2, 2) is 1

    cv::Mat t = K.inv() * projMat.col(3);

    return std::make_tuple(K, R, t);
    
}

CalibData DTUDataLoader::loadCalibIntrinsics(const std::string & viewLeftId, const std::string & viewRightId) {
    // Recover calibData from projection matrix (intrinsics + baseline)
    auto [K0, R0, t0] = decomposeProjectionMatrix(viewLeftId);
    auto [K1, R1, t1] = decomposeProjectionMatrix(viewRightId);

    // TODO: Fix if the recovered R0, R1 are not proper rotation matrices (det(R) != 1)
    assert(std::abs(cv::determinant(R0) - 1.0) < detTolerance);
    assert(std::abs(cv::determinant(R1) - 1.0) < detTolerance);

    // intrinsics should be same (same camera), but not same
    printMatInfo("Intrinsics Left", K0);
    std::cout << K0 << std::endl;
    printMatInfo("Intrinsics Right", K1);
    std::cout << K1 << std::endl;

    CalibData calib;
    calib.K0 = K0; calib.K1 = K1;

    // find camera centers in WS
    cv::Mat C0 = -R0.t() * t0;
    cv::Mat C1 = -R1.t() * t1;
    
    // compute baseline
    calib.baseline = cv::norm(C1 - C0);
    std::cout << "Horizontal Baseline: " << calib.baseline << " (mm)" << std::endl;

    return calib;
}

void DTUDataLoader::loadLeftExtrinsics(CalibData& calib,
                                        const std::string& viewLeftId,
                                        const std::string& viewRightId) {
    
                                            // If sparse matching swapped L/R, the effective left camera is the original right.
    const std::string& effectiveId = calib.swapped ? viewRightId : viewLeftId;
    auto [K, R, t] = decomposeProjectionMatrix(effectiveId);
    calib.R0 = R;
    calib.t0 = t;
}
