#pragma once
#include "Eigen.h"
#include <vector>
#include <tuple>

/**
 * @brief Represents a detected keypoint in the image.
 */
struct Keypoint {
    float x, y;
    double sigma;
    float orientation_deg;
    std::vector<float> descriptor;
};

/**
 * @brief Class implementing the SIFT (Scale-Invariant Feature Transform) algorithm for keypoint detection and descriptor computation.
 */
class SIFT {
public:
    SIFT(int octave_layers_num, double sigma, int max_features = 0,
         double contrast_thresh = 0.03, double edge_thresh_r = 10.0)
    : m_octave_layers_num(octave_layers_num)
    , m_octaves_num(0)
    , m_sigma(sigma)
    , m_max_features(max_features)
    , m_contrast_thresh(contrast_thresh)
    , m_edge_thresh_r(edge_thresh_r) {}

    /**
     * @brief Detect keypoints and compute descriptors for the input image.
     * 
     * @param[in] img 
     * @return std::vector<Keypoint> 
     */
    std::vector<Keypoint> detect_features(const Eigen::MatrixXf & img);


private:
    int m_octave_layers_num;    // Number of layers per octave (how many times is Gaussian applied with the same resolution)
    int m_octaves_num;          // Number of octaves (how many times is the image downsampled by 2)
    double m_sigma;             // Initial blur applied to the base
    int m_max_features;         // 0 = unlimited; else cap on returned keypoints, ranked by response
    double m_contrast_thresh;   // Low-contrast rejection threshold
    double m_edge_thresh_r;     // Edge-response rejection ratio

    // Per-call diagnostic counters, reset at the start of detect_features()
    int m_reject_not_converged = 0;
    int m_reject_low_contrast = 0;
    int m_reject_saddle_point = 0;
    int m_reject_edge_response = 0;

    constexpr static double INPUT_SIGMA = 0.5;      // Assumed blur of the input image (before upsampling)
    constexpr static int MAX_REFINEMENT_STEPS = 5;  // Max Newton iterations in localize_keypoint
    constexpr static int ORIENT_SMOOTH_PASSES = 1;  // Number of 5-tap binomial (1,4,6,4,1)/16 smoothing passes over the orientation histogram
    constexpr static float MAX_PEAK_RATIO = 0.8;    // Threshold for secondary peaks (orientation and descriptor histograms)
    constexpr static int IMAGE_BORDER = 5;          // Min distance (px) from the DoG image edge for a keypoint to be considered (matches OpenCV's SIFT_IMG_BORDER)

    /**
     * @brief Set the number of octaves object
     * 
     * @param[in] img 
     */
    void set_number_of_octaves(const Eigen::MatrixXf & img);

    /**
     * @brief Build the Gaussian and DoG pyramids for the input image.
     * 
     * @param[in] img Input image
     * @param[in] sigmas Blur values for each layer in the Gaussian pyramid
     * @param[out] gauss_pyramid Gaussian pyramid (vector of octaves, each octave is a vector of layers)
     * @param[out] dog_pyramid Difference of Gaussian pyramid
     */
    void build_scale_space(const Eigen::MatrixXf & img,
                            std::vector<double> & sigmas,
                            std::vector<std::vector<Eigen::MatrixXf>> & gauss_pyramid,
                            std::vector<std::vector<Eigen::MatrixXf>> & dog_pyramid);

    /**
     * @brief Refines one DoG extremum candidate to sub-pixel/sub-scale precision and applies the
     * contrast/edge rejection tests.
     * 
     * @param[in] dog_pyramid Difference of Gaussian pyramid
     * @param[in] octave_idx Index of the octave containing the keypoint
     * @param[in] layer_idx Index of the layer containing the keypoint
     * @param[in] x X-coordinate of the keypoint
     * @param[in] y Y-coordinate of the keypoint
     * @param[out] refined_keypoint Refined keypoint coordinates and scale
     * @return true if the keypoint is valid, false otherwise
     */
    bool localize_keypoint(const std::vector<std::vector<Eigen::MatrixXf>> & dog_pyramid,
                            int octave_idx, int layer_idx, int x, int y,
                            std::tuple<int, float, float, float, float> & refined_keypoint);

    /**
     * @brief Detect keypoints in the DoG pyramid
     * 
     * @param[in] dog_pyramid Difference of Gaussian pyramid
     * @return Vector of detected keypoints
     */
    std::vector<std::tuple<int, float, float, float, float>> detect_keypoints(const std::vector<std::vector<Eigen::MatrixXf>> & dog_pyramid);

    /**
     * @brief Finds dominant orientations for the keypoints
     * 
     * @param[in] refined_keypoints Vector of refined keypoint coordinates
     * @param[in] gauss_pyramid Gaussian pyramid
     * @param[in] sigmas Blur values for each layer in the Gaussian pyramid
     * @return Vector of orientation histograms for each keypoint
     */
    std::vector<std::vector<float>> build_orientation_histograms(const std::vector<std::tuple<int, float, float, float>> & refined_keypoints, std::vector<std::vector<Eigen::MatrixXf>> & gauss_pyramid, const std::vector<double> & sigmas);

    /**
     * @brief Build the final keypoints with descriptors from the refined candidates and their orientations.
     * 
     * @param refined_keypoints Vector of refined keypoint coordinates
     * @param gauss_pyramid Gaussian pyramid
     * @param sigmas Blur values for each layer in the Gaussian pyramid
     * @param keypoint_orientations Vector of orientation histograms for each keypoint
     * @return std::vector<Keypoint> Vector of final keypoints with descriptors
     */
    std::vector<Keypoint> generate_descriptors(const std::vector<std::tuple<int, float, float, float>> & refined_keypoints,
                                                const std::vector<std::vector<Eigen::MatrixXf>> & gauss_pyramid,
                                                const std::vector<double> & sigmas,
                                                const std::vector<std::vector<float>> & keypoint_orientations);
};

/**
 * @brief Get suitable kernel size for Gaussian blur based on the given sigma value.
 *
 * @param[in] sigma Blur value
 * @return Kernel size
 */
int get_kernel_size(double sigma);


Eigen::MatrixXf downsample(const Eigen::MatrixXf & img);
Eigen::MatrixXf upsample(const Eigen::MatrixXf & img);

/**
 * @brief Detect local extrema in the DoG pyramid.
 *        Detects whether the pixel at (x, y) in layer layer_idx is a local extremum (either a local maximum or minimum).
 *        By comparing the pixel's value to its 26 neighbors (8 in the same layer, 9 in the layer above, and 9 in the layer below).
 * 
 * @param[in] layers Layers of the DoG pyramid (vector of matrices)
 * @param[in] layer_idx Current layer index (scale) in the DoG pyramid
 * @param[in] x Current x-coordinate of the pixel
 * @param[in] y Current y-coordinate of the pixel
 * @return true if the pixel at (x, y) in layer layer_idx is a local extremum
 * @return false otherwise
 */
bool is_extremum3d(const std::vector<Eigen::MatrixXf> & layers, int layer_idx, int x, int y);


/**
 * @brief Compute the Taylor expansion offset for a keypoint in the DoG pyramid.
 *        Called repeatedly by localize_keypoint until the offset is small enough to accept.
 * 
 * @param[in] dog_octave DoG octave (vector of layers)
 * @param[in] layer_idx Current layer index (scale) in the DoG pyramid
 * @param[in] x Current x-coordinate of the pixel
 * @param[in] y Current y-coordinate of the pixel
 * @param[in] gradient Gradient at the keypoint
 * @param[in] hessian Hessian at the keypoint
 * @return Eigen::Vector3f Taylor expansion offset
 */
Eigen::Vector3f compute_taylor_offset(const std::vector<Eigen::MatrixXf> & dog_octave,
                                    int layer_idx, int x, int y,
                                    Eigen::Vector3f & gradient, Eigen::Matrix3f & hessian);

/**
 * @brief Compute the gradient magnitude and orientation at a pixel (x, y).
 * 
 * @param[in] img Image matrix
 * @param[in] x x-coordinate of the pixel
 * @param[in] y y-coordinate of the pixel
 * @return std::pair<double, double> Pair of gradient magnitude and orientation
 */
std::pair<double, double> compute_pixel_gradient(const Eigen::MatrixXf & img, int x, int y);

/**
 * @brief Get the neighbourhood regions around a pixel with their gradient magnitudes and orientations
 *        Weighting magnitudes by a Gaussian window centered at the keypoint.
 *  
 * @param img Image matrix
 * @param x_center x-coordinate of the center pixel
 * @param y_center y-coordinate of the center pixel
 * @param radius Radius of the neighbourhood
 * @return std::vector<std::tuple<int, int, double, double>> Vector of neighbourhood regions with gradient information
 */
std::vector<std::tuple<int, int, double, double>> get_neighbourhood_regions_with_gradient(const Eigen::MatrixXf & img, int x_center, int y_center, int radius);

/**
 * @brief Compute a Gaussian-weighted histogram of gradient orientations for a keypoint's neighbourhood.
 * 
 * @param neighbours_with_grads Neighbourhood regions with gradient magnitudes and orientations
 * @param weight_sigma Standard deviation of the Gaussian window
 * @param num_bins Number of bins in the histogram
 * @return std::vector<float> Histogram of gradient orientations
 */
std::vector<float> bin_orientation_votes(const std::vector<std::tuple<int, int, double, double>> & neighbours_with_grads, double weight_sigma, int num_bins);

/**
 * @brief Refine the orientation peaks by fitting a parabola through each local maximum and its two neighbours.
 * 
 * @param histogram Histogram of gradient orientations
 * @param local_max_args Indices of local maxima in the histogram
 * @return std::vector<float> Refined peak angles (deg)
 */
std::vector<float> refine_orientation_peaks(const std::vector<float> & histogram, const std::vector<int> & local_max_args);

/**
 * @brief Build the SIFT descriptor for a keypoint given its position, scale, and orientation.
 * 
 * @param gauss_img Image matrix at the keypoint's scale
 * @param x_kp x-coordinate of the keypoint
 * @param y_kp y-coordinate of the keypoint
 * @param sigma Scale of the keypoint
 * @param orientation_deg Orientation of the keypoint (deg)
 * @param subregion_size Size of each subregion in the descriptor
 * @param orientation_bins Number of bins in the orientation histogram
 * @param magnitude_threshold Threshold for considering gradient magnitudes
 * @return std::vector<float> SIFT descriptor
 */
std::vector<float> compute_descriptor(const Eigen::MatrixXf & gauss_img, float x_kp, float y_kp, double sigma, float orientation_deg,
                                       int subregion_size, int orientation_bins, float magnitude_threshold);
