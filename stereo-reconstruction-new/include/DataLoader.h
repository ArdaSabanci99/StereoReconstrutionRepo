#pragma once
#include <opencv2/core.hpp>
#include "utils.h"
#include <string>
#include <tuple>


/**
 * @brief Class for loading DTU dataset data.
 */
class DTUDataLoader {
    std::string m_data_path;

    
public:
    /**
     * @brief Constructor for DTUDataLoader.
     * @param data_path Path to the DTU dataset.
     */
    DTUDataLoader(const std::string & data_path);

    /**
     * @brief Loads an image from the dataset.
     * @param scene_id ID of the scene.
     * @param view_id ID of the view.
     * @return Loaded image.
     */
    cv::Mat loadImage(const std::string & scene_id, const std::string & view_id);
    
    /**
     * @brief Loads a camera projection matrix. 
     *        DTU Dataset does not provide camera instrinsics. 
     *        Need to compute them from the projection matrix.
     * @param view_id ID of the view.
     * @return Loaded projection matrix.
     */
     cv::Mat loadCameraProjection(const std::string & view_id);
    
    /**
     * @brief Decomposes a camera projection matrix into intrinsic and extrinsic components.
     * @param view_id ID of the view.
     * @return Tuple containing the intrinsic matrix, rotation matrix, and translation vector.
     */
    std::tuple<cv::Mat, cv::Mat, cv::Mat> decomposeProjectionMatrix(const std::string & viewId);
    
    /**
     * @brief Loads intrinsics for left and right cameras from the DTU dataset. 
     *        Also computes the baseline. 
     * 
     * @param view_left_id ID of the left camera view.
     * @param view_right_id ID of the right camera view.
     * @return CalibData with stored intrinsics and baseline. Used as an input to the stereo reconstruction pipeline.
     */
    CalibData loadCalibIntrinsics(const std::string & view_left_id, const std::string & view_right_id);

    /**
     * @brief Loads the left camera's extrinsic parameters (R0, t0) from the DTU dataset.
     *        This is required for transforming the point cloud from rectified camera space to world space
     *        to perform multi-pair fusion.
     *        Saves the R0 and t0 in the provided CalibData object.
     * 
     * @param calib Data object for storing calibration parameters. R0 and t0 will be updated in this object.
     * @param view_left_id ID of the left camera view.
     * @param view_right_id ID of the right camera view.
     */
    void loadLeftExtrinsics(CalibData& calib,
                            const std::string& view_left_id,
                            const std::string& view_right_id);

};
