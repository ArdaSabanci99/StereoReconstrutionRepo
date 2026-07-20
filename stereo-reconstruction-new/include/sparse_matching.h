#pragma once
#include "utils.h"
#include "Eigen.h"
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Enumeration for selecting the SIFT backend to use in sparse matching.
 *        OpenCV: Uses OpenCV's built-in SIFT implementation.
 *        Custom: Uses a custom SIFT implementation defined in KeypointDetector.h.
 */
enum class SiftBackend { OpenCV, Custom };

/**
 * @brief Enumeration for selecting the fundamental matrix estimation backend in sparse matching.
 *        OpenCVFundamental: Uses OpenCV's built-in fundamental matrix estimation.
 *        CustomRansac: Uses a custom RANSAC implementation.
 */
enum class FMatrixBackend { OpenCVFundamental, CustomRansac };

/**
 * @brief Data structure for sparse matching parameters.
 */
struct SparseMatchParams {
    int sift_features = 0;                  // Maximum number of SIFT features to detect (0 = no limit)
    int sift_octave_layers = 5;             // Number of layers per octave in SIFT
    double sift_contrast_thresh = 0.04;     // Contrast threshold for SIFT keypoint detection
    double sift_edge_thresh = 10.0;         // Edge threshold for SIFT keypoint detection
    double sift_sigma = 1.6;                // Initial Gaussian blur sigma for SIFT
    
    float ratio_threshold = 0.75f;          // Threshold for Lowe's ratio test in descriptor matching
    double ransac_threshold = 1.0;          // Maximum Sampson distance (px) for RANSAC inlier classification
    double chirality_depth = 25.0;          // Depth cap in recoverPose to reject unreliable far/near-parallel-ray points
    size_t custom_ransac_iters = 1000;      // Number of iterations for the custom RANSAC fundamental matrix estimation

    bool use_custom_sift = false;           // Flag to use the custom SIFT implementation instead of OpenCV's SIFT
};

/**
 * @brief Data structure for storing the results of sparse matching.
 */
struct SparseMatchResult {
    std::vector<cv::Point2f> pts_left;      // Matched keypoints in the left image
    std::vector<cv::Point2f> pts_right;     // Matched keypoints in the right image
    std::vector<uchar> inlier_mask;         // Inlier mask after RANSAC (1 = inlier, 0 = outlier)

    cv::Mat F;   
    cv::Mat E;
    cv::Mat R;
    cv::Mat t;

    int n_matches = 0;                  // Number of matches after Lowe ratio test
    int n_inliers = 0;                  // Number of inliers after RANSAC (used for Sampson error)
    int n_pose_inliers = 0;             // Number of inliers after recoverPose chirality check
    double mean_epipolar_error = 0.0;   // Mean Sampson epipolar error over RANSAC inliers (px)
    
    double sift_time_ms    = 0.0;  // Time taken by detectSiftMatches
    double fmatrix_time_ms = 0.0;  // Time taken by computeSparseMatchesOpenCV/Custom
};

/**
 * @brief Computes sparse matches between two images.
 * 
 * @param[in] left Left image
 * @param[in] right Right image
 * @param[in, out] calib Calibration data containing intrinsic matrices and baseline
 * @param[in] sift_backend Which SIFT backend to use (OpenCV or Custom)
 * @param[in] fmatrix_backend Which fundamental matrix estimation backend to use (OpenCV or Custom)
 * @param[in] params Parameters for sparse matching, including SIFT and RANSAC settings
 * @return SparseMatchResult
 */
SparseMatchResult computeSparseMatches(cv::Mat& left,
                                        cv::Mat& right,
                                        CalibData& calib,
                                        SiftBackend sift_backend,
                                        FMatrixBackend fmatrix_backend,
                                        const SparseMatchParams& params = SparseMatchParams{});

/**
 * @brief Computes sparse matches using OpenCV's fundamental matrix estimation.
 *
 * @param[in, out] result SparseMatchResult containing matched points.
 *                        Computes and sets it the fundamental matrix, essential matrix, and pose.
 * @param[in] calib Calibration data containing intrinsic matrices and baseline.
 * @param[in] params Parameters for sparse matching, including SIFT and RANSAC settings.
 */
void computeSparseMatchesOpenCV(SparseMatchResult& result, const CalibData& calib,
                                 const SparseMatchParams& params);

/** 
 * @brief Computes sparse matches using a custom RANSAC implementation for fundamental matrix estimation.
 * @param[in, out] result SparseMatchResult containing matched points.
 *                        Computes and sets it the fundamental matrix, essential matrix, and pose.
 * @param[in] calib Calibration data containing intrinsic matrices and baseline.
 * @param[in] params Parameters for sparse matching, including SIFT and RANSAC settings.
*/
void computeSparseMatchesCustom(SparseMatchResult& result, const CalibData& calib,
                                 const SparseMatchParams& params);

/** 
 * @brief Interface for keypoint detection and matching. 
 *      Detects SIFT matches (using specified backend) between two images.
 * @param[in] img_left Left image.
 * @param[in] img_right Right image.
 * @param[in] params Parameters for SIFT matching.
 * @param[in] sift_backend Which SIFT backend to use.
 * @return SparseMatchResult containing the detected matches.
 */
SparseMatchResult detectSiftMatches(const cv::Mat& img_left, const cv::Mat& img_right,
                                     const SparseMatchParams& params, SiftBackend sift_backend);


/**
 * @brief Computes the Sampson distance (px) between corresponding points and the fundamental matrix.
 * @param[in] pt_left Point in the left image.
 * @param[in] pt_right Point in the right image.
 * @param[in] F Fundamental matrix.
 * @return double Sampson distance (px).
 */
double sampsonDistance(const cv::Point2f& pt_left, const cv::Point2f& pt_right, const cv::Mat& F);

/** 
 * @brief Recovers the relative pose between two cameras given corresponding points and the essential matrix.
 * 
 * @param pts_left_norm Normalized points in the left image.
 * @param pts_right_norm Normalized points in the right image.
 * @param E Essential matrix.
 * @param R Recovered rotation matrix.
 * @param t Recovered translation vector.
 * @param chirality_depth Maximum depth for chirality check.
 * @return size_t Number of inliers.
 */
size_t recoverPose(const std::vector<Eigen::Vector3d>& pts_left_norm, const std::vector<Eigen::Vector3d>& pts_right_norm,
                                                            const Eigen::Matrix3d& E,
                                                            Eigen::Matrix3d & R, Eigen::Vector3d & t,
                                                            double chirality_depth);
                                                            
/**
 * @brief Normalizes a set of points using the intrinsic matrix.
 * 
 * @param pts Points to normalize.
 * @param K Intrinsic matrix.
 * @return std::vector<Eigen::Vector3d> Normalized points.
 */
std::vector<Eigen::Vector3d> normalizePoints(const std::vector<Eigen::Vector3d>& pts, const Eigen::Matrix3d & K);



