#pragma once
#include "Eigen.h"
#include <vector>
#include <tuple>

// One detected SIFT feature. Public return type of SIFT::detect_features, so it's a named
// struct (not a tuple) the same way SparseMatchResult is in sparse_matching.h.
struct Keypoint {
    float x, y;
    double sigma;
    float orientation_deg;
    std::vector<float> descriptor;
};

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

    // Detect keypoints and compute descriptors for the input image
    std::vector<Keypoint> detect_features(const Eigen::MatrixXf & img);


private:
    int m_octave_layers_num;  // Number of layers per octave (how many times is Gaussian applied with the same resolution)
    int m_octaves_num;  // Number of octaves (how many times is the image downsampled by 2)
    double m_sigma;  // Initial blur applied to the base
    int m_max_features;  // 0 = unlimited; else cap on returned keypoints, ranked by response
    double m_contrast_thresh;  // Low-contrast rejection threshold
    double m_edge_thresh_r;    // Edge-response rejection ratio

    // Per-call diagnostic counters, reset at the start of detect_features()
    int m_reject_not_converged = 0;
    int m_reject_low_contrast = 0;
    int m_reject_saddle_point = 0;
    int m_reject_edge_response = 0;

    constexpr static double INPUT_SIGMA = 0.5;  // Assumed blur of the input image (before upsampling)
    constexpr static int MAX_REFINEMENT_STEPS = 5;  // Max Newton iterations in localize_keypoint
    constexpr static int ORIENT_SMOOTH_PASSES = 1;  // Number of 5-tap binomial (1,4,6,4,1)/16 smoothing passes over the orientation histogram
    constexpr static float MAX_PEAK_RATIO = 0.8;  // Threshold for secondary peaks (orientation and descriptor histograms)
    constexpr static int IMAGE_BORDER = 5;  // Min distance (px) from the DoG image edge for a keypoint to be considered (matches OpenCV's SIFT_IMG_BORDER)

    // Set suitable number of octaves (based on img size, same as in OpenCV)
    void set_number_of_octaves(const Eigen::MatrixXf & img);

    // Build the Gaussian and Difference-of-Gaussian pyramids
    void build_scale_space(const Eigen::MatrixXf & img,
                            std::vector<double> & sigmas,
                            std::vector<std::vector<Eigen::MatrixXf>> & gauss_pyramid,
                            std::vector<std::vector<Eigen::MatrixXf>> & dog_pyramid);

    // Refines one DoG extremum candidate to sub-pixel/sub-scale precision and applies the
    // contrast/edge rejection tests. Returns false if rejected, else fills refined_keypoint.
    bool localize_keypoint(const std::vector<std::vector<Eigen::MatrixXf>> & dog_pyramid,
                            int octave_idx, int layer_idx, int x, int y,
                            std::tuple<int, float, float, float, float> & refined_keypoint);

    // Detect keypoints in the DoG pyramid
    std::vector<std::tuple<int, float, float, float, float>> detect_keypoints(const std::vector<std::vector<Eigen::MatrixXf>> & dog_pyramid);

    // Finds dominant orientations for the keypoints
    std::vector<std::vector<float>> build_orientation_histograms(const std::vector<std::tuple<int, float, float, float>> & refined_keypoints, std::vector<std::vector<Eigen::MatrixXf>> & gauss_pyramid, const std::vector<double> & sigmas);

    // Builds the final Keypoints (with descriptors) from the refined candidates and their orientations
    std::vector<Keypoint> generate_descriptors(const std::vector<std::tuple<int, float, float, float>> & refined_keypoints,
                                                const std::vector<std::vector<Eigen::MatrixXf>> & gauss_pyramid,
                                                const std::vector<double> & sigmas,
                                                const std::vector<std::vector<float>> & keypoint_orientations);
};

// ── Stateless helpers (no SIFT instance state; take everything as parameters, unit-testable
// on their own — same treatment as GaussianBlur's static methods) ────────────────────────────

// Get suitable kernel size (based on sigma, same as in OpenCV)
int get_kernel_size(double sigma);

Eigen::MatrixXf downsample(const Eigen::MatrixXf & img);
Eigen::MatrixXf upsample(const Eigen::MatrixXf & img);

// Detect local extrema in the DoG pyramid
bool is_extremum3d(const std::vector<Eigen::MatrixXf> & layers, int layer_idx, int x, int y);

// One Newton step toward the true DoG extremum near (x, y, layer_idx): fills gradient/hessian
// (3D finite differences over x, y, scale) and returns offset = -hessian^-1 * gradient.
// Called repeatedly by localize_keypoint until the offset is small enough to accept.
Eigen::Vector3f compute_taylor_offset(const std::vector<Eigen::MatrixXf> & dog_octave,
                                    int layer_idx, int x, int y,
                                    Eigen::Vector3f & gradient, Eigen::Matrix3f & hessian);

// Compute the gradient magnitude and orientation at a pixel (x, y)
std::pair<double, double> compute_pixel_gradient(const Eigen::MatrixXf & img, int x, int y);

// Get neighbourhood regions around a pixel with their gradient magnitudes and orientations
    // Taking neighbourhood of radius size
    // Weighting magnitudes by Gaussian window (centered at the keypoint)
std::vector<std::tuple<int, int, double, double>> get_neighbourhood_regions_with_gradient(const Eigen::MatrixXf & img, int x_center, int y_center, int radius);

// Compute a Gaussian-weighted histogram of gradient orientations for a keypoint's neighbourhood.
// num_bins is explicit (not read off SIFT) so this stays testable in isolation.
std::vector<float> bin_orientation_votes(const std::vector<std::tuple<int, int, double, double>> & neighbours_with_grads, double weight_sigma, int num_bins);

// Sub-bin localization of orientation histogram peaks
    // Fits a parabola through each local maximum and its two neighbours
    // Returns interpolated peak angles (deg)
std::vector<float> refine_orientation_peaks(const std::vector<float> & histogram, const std::vector<int> & local_max_args);

// Builds the 128-dim (default) SIFT descriptor for one (keypoint, orientation) pair.
// subregion_size/orientation_bins/magnitude_threshold are explicit (not read off SIFT) so this
// stays testable in isolation.
std::vector<float> compute_descriptor(const Eigen::MatrixXf & gauss_img, float x_kp, float y_kp, double sigma, float orientation_deg,
                                       int subregion_size, int orientation_bins, float magnitude_threshold);
