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
    CalibData loadCalib(const std::string & viewLeftId, const std::string & viewRightId);

};
