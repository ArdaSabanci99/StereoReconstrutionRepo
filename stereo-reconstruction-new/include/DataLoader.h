#pragma once
#include <opencv2/core.hpp>
#include "utils.h"
#include <string>
#include <tuple>



class DTUDataLoader {
    std::string m_data_path;

    
public:
    DTUDataLoader(const std::string & data_path);

    cv::Mat loadImage(const std::string & sceneId, const std::string & viewId, const std::string & lightId);
    cv::Mat loadCameraProjection(const std::string & viewId);
    std::tuple<cv::Mat, cv::Mat, cv::Mat> decomposeProjectionMatrix(const std::string & viewId);
    // Loads only K0, K1, baseline from DTU projection matrices.
    // Use loadCalibData() from utils.h to reload a previously saved full CalibData.
    CalibData loadCalibIntrinsics(const std::string & viewLeftId, const std::string & viewRightId);

    // Loads R0, t0 for the effective left camera into calib.
    void loadLeftExtrinsics(CalibData& calib,
                            const std::string& viewLeftId,
                            const std::string& viewRightId);

};  // TODO: resolve issues with loading extrinsics data
        // RectifyOpenCV: one that uses the extrinsics (load correct), second that doesn't but does the rectification still with opencv without it
        // Check: LeftExtrinsics are loaded at correct times
