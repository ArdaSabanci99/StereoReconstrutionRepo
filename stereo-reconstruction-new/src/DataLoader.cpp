#include "DataLoader.h"
#include <fstream>
#include <stdexcept>

constexpr double detTolerance = 1e-6;

DTUDataLoader::DTUDataLoader(const std::string & data_path) 
    : m_data_path(data_path) {}

cv::Mat DTUDataLoader::loadImage(const std::string & scene_id, const std::string & view_id) {
    std::string img_path = m_data_path + "/Rectified/scan" + scene_id 
    + "/rect_" + view_id + "_max.png";

    cv::Mat img = cv::imread(img_path);

    if (img.empty())
        throw std::runtime_error("Cannot open: " + img_path);

    return img;
}

cv::Mat DTUDataLoader::loadCameraProjection(const std::string & view_id) {
    std::string camera_proj_path = m_data_path + "/Calibration/cal18/pos_" + view_id + ".txt";

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

std::tuple<cv::Mat, cv::Mat, cv::Mat> DTUDataLoader::decomposeProjectionMatrix(const std::string & view_id) {
    cv::Mat projMat = loadCameraProjection(view_id);
    cv::Mat K, R, Qx, Qy, Qz;

    cv::RQDecomp3x3(projMat.colRange(0, 3), K, R, Qx, Qy, Qz);
    K /= K.at<double>(2, 2); // normalising, so at (2, 2) is 1

    cv::Mat t = K.inv() * projMat.col(3);

    return std::make_tuple(K, R, t);
    
}

CalibData DTUDataLoader::loadCalibIntrinsics(const std::string & view_left_id, const std::string & view_right_id) {
    // Recover calibData from projection matrix (intrinsics + baseline)
    auto [K0, R0, t0] = decomposeProjectionMatrix(view_left_id);
    auto [K1, R1, t1] = decomposeProjectionMatrix(view_right_id);

    if (std::abs(cv::determinant(R0) - 1.0) >= detTolerance)
        throw std::runtime_error("[calib] Left camera rotation recovered from " + view_left_id
            + "'s projection matrix is not a proper rotation matrix (det != 1).");
    if (std::abs(cv::determinant(R1) - 1.0) >= detTolerance)
        throw std::runtime_error("[calib] Right camera rotation recovered from " + view_right_id
            + "'s projection matrix is not a proper rotation matrix (det != 1).");

    // K0/K1 are physically the same camera, but each is recovered independently via RQ
        // Decomposition of its own view's projection matrix, so expect small numerical differences
    printMatInfo("Intrinsics Left", K0);
    std::cout << K0 << std::endl;
    printMatInfo("Intrinsics Right", K1);
    std::cout << K1 << std::endl;

    CalibData calib;
    calib.K0 = K0; calib.K1 = K1;

    // Find camera centers in WS
    cv::Mat C0 = -R0.t() * t0;
    cv::Mat C1 = -R1.t() * t1;
    
    // Compute baseline
    calib.baseline = cv::norm(C1 - C0);
    std::cout << "Baseline: " << calib.baseline << " (mm)" << std::endl;

    return calib;
}

void DTUDataLoader::loadLeftExtrinsics(CalibData& calib,
                                        const std::string& view_left_id,
                                        const std::string& view_right_id) {

    // Chooding the camera that is effectively the left camera in the stereo pair (geometrically to the left of the other camera)
    const std::string& effectiveId = calib.swapped ? view_right_id : view_left_id;
    auto [K, R, t] = decomposeProjectionMatrix(effectiveId);
    calib.R0 = R;
    calib.t0 = t;
}

void DTUDataLoader::loadGTRelativePose(CalibData& calib,
                                        const std::string& view_left_id,
                                        const std::string& view_right_id) {
    auto [K0_gt, R0_gt, t0_gt] = decomposeProjectionMatrix(view_left_id);
    auto [K1_gt, R1_gt, t1_gt] = decomposeProjectionMatrix(view_right_id);

    calib.R_rel = R1_gt * R0_gt.t();
    calib.t_rel = t1_gt - calib.R_rel * t0_gt;   // in mm (DTU world units)

    const cv::Mat& t = calib.t_rel;
    cv::Mat tx = (cv::Mat_<double>(3, 3) <<
                  0, -t.at<double>(2), t.at<double>(1),
                  t.at<double>(2), 0, -t.at<double>(0),
                  -t.at<double>(1), t.at<double>(0), 0);
    cv::Mat E_gt = tx * calib.R_rel;
    calib.F = calib.K1.inv().t() * E_gt * calib.K0.inv();
    calib.verifyLeftRightCameraOrder();
}
