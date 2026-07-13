#include "KeypointDetector.h"
#include "GaussianBlur.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

// Constants for the free (stateless) helper functions below — passed in explicitly by their
// callers rather than read off SIFT, so bin_orientation_votes/compute_descriptor stay testable
// independent of a SIFT instance.
constexpr int ORIENT_BIN_NUMS = 36;              // Number of bins in the orientation histogram
constexpr int DESC_SUBREGION_SIZE = 4;           // Descriptor is 4x4 subregions
constexpr int DESC_ORIENTATION_BINS = 8;         // Descriptor has 8 orientation bins per subregion
constexpr float DESC_MAGNITUDE_THRESHOLD = 0.2f; // Threshold for descriptor magnitude clipping

// ─────────────────────────────────────────────────────────────────────────────
void SIFT::set_number_of_octaves(const Eigen::MatrixXf & img) {
    int image_height = img.rows();
    int image_width = img.cols();

    m_octaves_num = std::round(std::log(std::min(image_height, image_width)) / std::log(2.0) - 1.0);  // OpenCV's formula 
    m_octaves_num = std::max(m_octaves_num, 1);  // Ensure at least one octave
    std::cout << "[SIFT] Number of octaves: " << m_octaves_num << std::endl;
}

// ─────────────────────────────────────────────────────────────────────────────
int get_kernel_size(double sigma) {
    return 2 * std::ceil(3 * sigma) + 1;  // Standard deviation rule: 3*sigma on each side + 1 for the center pixel
}
// ─────────────────────────────────────────────────────────────────────────────
Eigen::MatrixXf downsample(const Eigen::MatrixXf & img) {
    int rows = img.rows() / 2;
    int cols = img.cols() / 2;
    Eigen::MatrixXf img_down(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Take every second pixel in both dimensions
            img_down(i, j) = img(2 * i, 2 * j);
        }
    }

    return img_down;
}
// ─────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXf upsample(const Eigen::MatrixXf & img) {
    int src_rows = img.rows();
    int src_cols = img.cols();
    int rows = src_rows * 2;
    int cols = src_cols * 2;
    Eigen::MatrixXf img_up(rows, cols);

    for (int i = 0; i < rows; i++) {
        // Coordinate mapping: find the corresponding source pixel in the original image
        double dst_y_center = i + 0.5;
        double src_y_center = dst_y_center * 0.5;
        double src_y = src_y_center - 0.5;  // Turning back to index
        
        // Find neighbouring pixels (rows) for bilinear interpolation
        int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, src_rows - 1);
        int y1 = std::clamp(y0 + 1, 0, src_rows - 1);
        
        double fy = src_y - std::floor(src_y);  // Fractional part for interpolation

        for (int j = 0; j < cols; j++) {
            // Coordinate mapping: find the corresponding source pixel in the original image
            double dst_x_center = j + 0.5;
            double src_x_center = dst_x_center * 0.5;
            double src_x = src_x_center - 0.5;

            // Find neighbouring pixels (columns) for bilinear interpolation
            int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, src_cols - 1);
            int x1 = std::clamp(x0 + 1, 0, src_cols - 1);
            double fx = src_x - std::floor(src_x);

            // Bilinear blend
                // Interpolate along x on the two neighbouring rows
                // Blend those two results along y

            float top = static_cast<float>((1 - fx) * img(y0, x0) + fx * img(y0, x1));
            float bottom = static_cast<float>((1 - fx) * img(y1, x0) + fx * img(y1, x1));
            
            img_up(i, j) = static_cast<float>((1 - fy) * top + fy * bottom);
        }
    }

    return img_up;
}
// ─────────────────────────────────────────────────────────────────────────────
void SIFT::build_scale_space(const Eigen::MatrixXf & up_img,
                            std::vector<double> & sigmas,
                            std::vector<std::vector<Eigen::MatrixXf>> & gauss_pyramid,
                            std::vector<std::vector<Eigen::MatrixXf>> & dog_pyramid) {
    // 1. Base image for the first octave
    // sigmas[i] = m_sigma * k^i -- a uniform geometric sequence starting at the base sigma, so
    // every consecutive Gaussian pair (and therefore every DoG layer) is spaced by the same
    // ratio k. (Previously sigmas[0] held the raw image's assumed un-blurred sigma, which broke
    // that uniformity for the first DoG layer -- see notes/ discussion.)
    sigmas[0] = m_sigma;
    double k = std::pow(2.0, 1.0/m_octave_layers_num);
    for (int i = 1; i < sigmas.size(); i++) {
        sigmas[i] = sigmas[i-1] * k;
    }

    std::cout << "[SIFT] Per-layer sigma (blur level):";
    for (double s : sigmas) std::cout << " " << s;
    std::cout << std::endl;

    // Bring the raw upsampled image (assumed inherent blur INPUT_SIGMA*2) up to the pyramid's
    // base sigma once, before it enters the per-octave loop -- matches OpenCV's
    // createInitialImage, which blurs straight to the base sigma outside buildGaussianPyramid.
    double base_sigma_inc = std::sqrt(std::pow(sigmas[0], 2) - std::pow(INPUT_SIGMA * 2, 2));
    int base_kernel_size = get_kernel_size(base_sigma_inc);
    Eigen::MatrixXf octave_base_img = GaussianBlur::apply_gaussian2d(up_img, base_kernel_size, base_sigma_inc);

    // 2. Create the Gaussian pyramid
    for (int octave = 0; octave < m_octaves_num; octave++) {
        std::vector<Eigen::MatrixXf> octave_images;
        Eigen::MatrixXf current_img = octave_base_img; // Start with the base image for the current octave
        octave_images.push_back(octave_base_img);  // sigmas[0] level, needed for the first DoG pair

        for (int layer = 1; layer < sigmas.size(); layer++) {
            // Blurring the layer to reach sigmas[x] level of blur
            // Image is already blurred to some extent -> blurring by increment only
            double sigma_inc = std::sqrt(std::pow(sigmas[layer], 2) - std::pow(sigmas[layer-1], 2));
            int kernel_size = get_kernel_size(sigma_inc);
            Eigen::MatrixXf blurred_img = GaussianBlur::apply_gaussian2d(current_img, kernel_size, sigma_inc);
            octave_images.push_back(blurred_img);

            current_img = blurred_img;  // Update current image for the next layer
        }
        gauss_pyramid.push_back(octave_images);
        // octave_images is index-aligned with sigmas, so the "double base sigma" level (k^s = 2)
        // is at index m_octave_layers_num -- matches OpenCV's own next-octave-base pick.
        octave_base_img = downsample(octave_images[m_octave_layers_num]);
    }

    // 3. Create the Difference of Gaussian (DoG) pyramid
    for (const auto& octave_images : gauss_pyramid) {
        std::vector<Eigen::MatrixXf> octave_dog_images;
        for (size_t layer = 1; layer < octave_images.size(); layer++) {
            Eigen::MatrixXf dog_img = octave_images[layer] - octave_images[layer - 1];
            octave_dog_images.push_back(dog_img);
        }
        dog_pyramid.push_back(octave_dog_images);
    }
}
// ─────────────────────────────────────────────────────────────────────────────
bool is_extremum3d(const std::vector<Eigen::MatrixXf> & layers, int layer_idx, int x, int y) {
    float center = layers[layer_idx](x, y);
    bool is_min = true;
    bool is_max = true;

    // Compare with neighbours
    for (int dl = -1; dl <= 1; dl++) {  // Go through the layers: below, same, above
        for (int dx = -1; dx <= 1; dx++) { // Column Ofset
            for (int dy = -1; dy <= 1; dy++) { // Row Offset
                if(dl == 0 && dx == 0 && dy == 0) 
                    continue; // Skip the current pixel

                float neighbour = layers[layer_idx + dl](x + dx, y + dy);

                if (neighbour >= center) 
                    is_max = false;

                if (neighbour <= center) 
                    is_min = false;

                if (!is_min && !is_max) 
                    return false;  // Early exit if neither
            }
        }
    }

    return is_max || is_min;
}
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Vector3f compute_taylor_offset(const std::vector<Eigen::MatrixXf> & dog_octave, int layer_idx, int x, int y,
                                        Eigen::Vector3f & gradient, Eigen::Matrix3f & hessian) {
    // Wrapper for better readability
    auto D = [&](int dx, int dy, int dl) {
        return dog_octave[layer_idx + dl](x + dx, y + dy);
    };

    // Compute the gradient at the candidate point
    float grad_x = (D(1, 0, 0) - D(-1, 0, 0)) / 2.0f;
    float grad_y = (D(0, 1, 0) - D(0, -1, 0)) / 2.0f;
    float grad_l = (D(0, 0, 1) - D(0, 0, -1)) / 2.0f;

    gradient = Eigen::Vector3f (grad_x, grad_y, grad_l);

    // Compute the Hessian matrix at the candidate point
    float center = D(0, 0, 0);
    float dxx = D(1, 0, 0) - 2*center + D(-1, 0, 0);

    // x-slope at (y+1) - x-slope at (y-1) = dxy
    float dxy = (D(1, 1, 0) - D(-1, 1, 0) - D(1, -1, 0) + D(-1, -1, 0)) / 4.0f;
    float dxl = (D(1, 0, 1) - D(-1, 0, 1) - D(1, 0, -1) + D(-1, 0, -1)) / 4.0f;

    float dyy = D(0, 1, 0) - 2*center + D(0, -1, 0);
    float dyx = dxy;  // Hessian is symmetric
    float dyl = (D(0, 1, 1) - D(0, -1, 1) - D(0, 1, -1) + D(0, -1, -1)) / 4.0f;

    float dll = D(0, 0, 1) - 2*center + D(0, 0, -1);
    float dlx = dxl;  // Hessian is symmetric
    float dly = dyl;  // Hessian is symmetric

    hessian << dxx, dxy, dxl,
                dyx, dyy, dly,
                dlx, dly, dll;


    // offset = H^-1 * gradient
        // Solving the linear system with LDLT (handles indefinite symmetric matrices)
    Eigen::Vector3f offset = hessian.ldlt().solve(-gradient);

    return offset;
}
// ─────────────────────────────────────────────────────────────────────────────
bool SIFT::localize_keypoint(const std::vector<std::vector<Eigen::MatrixXf>> & dog_pyramid,
                              int octave_idx, int layer_idx, int x, int y,
                              std::tuple<int, float, float, float, float> & refined_keypoint) {
    // Sub-pixel / sub-scale localization
        // Goal: find the exact location and its scale of the keypoint (not just the pixel location)
        // How: fit a local quadratic model to the DoG values

    // Measure how the DoG value is changing in each direction (x, y, scale) around candidate
    int center_x = x;
    int center_y = y;
    int center_layer = layer_idx;

    for (size_t i = 0; i < MAX_REFINEMENT_STEPS; i++) {  // TODO: Add argument controlling the number of itations
        Eigen::Vector3f gradient;
        Eigen::Matrix3f hessian;
        Eigen::Vector3f offset = compute_taylor_offset(dog_pyramid[octave_idx], center_layer, center_x, center_y, gradient, hessian);

        // Check if the offset in all directions is small:
            // if < 0.5 -> accept it
            // if >= 0.5 -> reject it (too far from the candidate point, try in the new closest pixel)

        bool is_accepted = (std::abs(offset(0)) < 0.5 && std::abs(offset(1)) < 0.5 && std::abs(offset(2)) < 0.5);

        if (is_accepted) {  // Interpolate and check contrast and edge response
            float refined_x = center_x + offset(0);
            float refined_y = center_y + offset(1);
            float refined_layer = center_layer + offset(2);

            float center_value = dog_pyramid[octave_idx][center_layer](center_x, center_y);
            float offset_value = center_value + 0.5f * gradient.dot(offset);
            

            // Rescale to [0,255]-pixel-scale DoG units and per-octave-layer, matching OpenCV's convention.
            double effective_contrast_thresh = (m_contrast_thresh * 255.0) / m_octave_layers_num;
            if (std::abs(offset_value) < effective_contrast_thresh) {
                m_reject_low_contrast++;
                return false;  // Reject low contrast keypoints
            }

            // Edge responses
                // Measure if it changes in both x- and y-direction
                // Hessian eigenvalues -> principal curvatures (care only of 2x2 matrix)
                    // lambda1: how steep along the steepest direction, lambda2: how steep along the gentlest direction
                        // Corner/blob: both curvatures are large -> similar magnitude
                        // Edge: curvature large only in one direction

            float trace = hessian(0, 0) + hessian(1, 1);
            float determinant = hessian(0, 0) * hessian(1, 1) - hessian(0, 1) * hessian(1, 0);

            double r = m_edge_thresh_r;
            if (determinant <= 0) {
                m_reject_saddle_point++;
                return false;  // Reject saddle point
            } else if (std::pow(trace, 2) / determinant >= (std::pow(r + 1, 2) / r)) {
                m_reject_edge_response++;
                return false;  // Reject edge response
            }

            // If we reach here, the keypoint is accepted (blob)

            refined_keypoint = std::make_tuple(octave_idx, refined_layer, refined_x, refined_y, std::abs(offset_value));
            
            return true;
        }

        // Move the center to a new integer position
        int shift_x = std::round(offset(0));
        int shift_y = std::round(offset(1));
        int shift_layer = std::round(offset(2));

        // Out of bounds -> discard
        if (shift_x < IMAGE_BORDER || shift_x >= dog_pyramid[octave_idx][center_layer].rows() - IMAGE_BORDER ||
            shift_y < IMAGE_BORDER || shift_y >= dog_pyramid[octave_idx][center_layer].cols() - IMAGE_BORDER ||
            shift_layer < 1 || shift_layer >= (int)dog_pyramid[octave_idx].size() - 1) {
            return false;
        }

        center_x = shift_x;
        center_y = shift_y;
        center_layer = shift_layer;
    }
    m_reject_not_converged++;
    return false;  // Exceeded max iterations without converging
}
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::tuple<int, float, float, float, float>> SIFT::detect_keypoints(const std::vector<std::vector<Eigen::MatrixXf>> & dog_pyramid) {
    // 1. Find candidate points - local extrema in the DoG pyramid
        // check if the candidate is a local minima/maxima -> if yes, add to candidate points
        // check across 26 neighbours: 8 around pixel in the DoG image + 9 in the layer above + 9 in the layer below = 26 neighbors

    std::vector<std::tuple<int, int, int, int>> candidate_points;  // Octave idx, layer idx, x, y
    for (size_t i = 0; i < dog_pyramid.size(); i++) {
        size_t candidates_before = candidate_points.size();

        for (size_t j = 1; j < dog_pyramid[i].size()-1; j++) { // Need to compare with layer above/below
            const Eigen::MatrixXf & current_layer = dog_pyramid[i][j];

            for (int x = IMAGE_BORDER; x < current_layer.rows()-IMAGE_BORDER; x++) {
                for (int y = IMAGE_BORDER; y < current_layer.cols()-IMAGE_BORDER; y++) {
                    if (is_extremum3d(dog_pyramid[i], j, x, y)) {
                        candidate_points.push_back(std::make_tuple(i, j, x, y));
                    }
                }
            }

        }
        std::cout << "[SIFT] Octave " << i << ": " << (candidate_points.size() - candidates_before)
                   << " extrema candidates" << std::endl;
    }
    std::cout << "[SIFT] Total extrema candidates (all octaves): " << candidate_points.size() << std::endl;

    // 2. Refining candidate points
    std::vector<std::tuple<int, float, float, float, float>> refined_keypoints;  // octave_idx, layer, x, y, response
    int rejected_low_contrast_or_edge = 0;
    m_reject_not_converged = 0;
    m_reject_low_contrast = 0;
    m_reject_saddle_point = 0;
    m_reject_edge_response = 0;
    for (const auto & [octave_idx, layer_idx, x, y] : candidate_points) {
        std::tuple<int, float, float, float, float> refined_keypoint;
        if (localize_keypoint(dog_pyramid, octave_idx, layer_idx, x, y, refined_keypoint)) {
            refined_keypoints.push_back(refined_keypoint);
        } else {
            rejected_low_contrast_or_edge++;
        }
    }
    std::cout << "[SIFT] Kept " << refined_keypoints.size() << " keypoints after sub-pixel localization "
               << "(rejected " << rejected_low_contrast_or_edge << " as low-contrast/edge/unstable)" << std::endl;
    std::cout << "[SIFT] Rejections: not_converged=" << m_reject_not_converged
               << " low_contrast=" << m_reject_low_contrast
               << " saddle_point=" << m_reject_saddle_point
               << " edge_response=" << m_reject_edge_response << std::endl;
    return refined_keypoints;
}
// ─────────────────────────────────────────────────────────────────────────────
std::pair<double, double> compute_pixel_gradient(const Eigen::MatrixXf & img, int x, int y) {
    double L_x = (img(x + 1, y) - img(x - 1, y)) / 2.0;
    double L_y = (img(x, y + 1) - img(x, y - 1)) / 2.0;
    double magnitude = std::sqrt(std::pow(L_x, 2) + std::pow(L_y, 2));

    double angle = std::atan2(L_y, L_x);

    return std::make_pair(magnitude, angle);
}
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::tuple<int, int, double, double>> get_neighbourhood_regions_with_gradient(const Eigen::MatrixXf & img, int x_center, int y_center, int radius) {
    std::vector<std::tuple<int, int, double, double>> neighbourhood;
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            int new_x = x_center + dx;
            int new_y = y_center + dy;
            // Keep a 1-pixel margin so compute_pixel_gradient's central differences stay in bounds
            if (new_x >= 1 && new_x < img.rows() - 1 && new_y >= 1 && new_y < img.cols() - 1) {
                auto [magnitude, angle] = compute_pixel_gradient(img, new_x, new_y);
                neighbourhood.push_back(std::make_tuple(dx, dy, magnitude, angle));
            }

        }
    }

    return neighbourhood;
}
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> bin_orientation_votes(const std::vector<std::tuple<int, int, double, double>> & neighbours_with_grads, double weight_sigma, int num_bins) {
    std::vector<float> histogram(num_bins, 0.0f);
    const double bin_width = 360.0 / num_bins;

    for (const auto & [dx, dy, magnitude, angle] : neighbours_with_grads) {
        // Compute Gaussian weight for each point, relative to the keypoint center (dx, dy)
        double weight = std::exp(-(std::pow(dx, 2) + std::pow(dy, 2)) / (2 * std::pow(weight_sigma, 2)));
        double vote = magnitude * weight;

        // Determine the histogram bin for the angle (rad)
        double angle_deg = angle * 180.0 / M_PI;
        if (angle_deg < 0) angle_deg += 360.0;

        int bin_index = static_cast<int>(angle_deg / bin_width) % num_bins;

        histogram[bin_index] += vote;
    }

    return histogram;
}
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> refine_orientation_peaks(const std::vector<float> & histogram, const std::vector<int> & local_max_args) {
    std::vector<float> refined_orientations;
    refined_orientations.reserve(local_max_args.size());
    for (int bin_idx : local_max_args) {
        int left_idx = (bin_idx - 1 + (int)histogram.size()) % (int)histogram.size();
        int right_idx = (bin_idx + 1) % (int)histogram.size();
        float left_value = histogram[left_idx];
        float center_value = histogram[bin_idx];
        float right_value = histogram[right_idx];

        // Fit a parabola: y = ax^2 + bx + c
        float a = (left_value + right_value - 2 * center_value) / 2.0f;
        float b = (right_value - left_value) / 2.0f;

        // The vertex of the parabola (the peak) is at x = -b/(2a); guard a flat peak (a == 0)
        float peak_offset = (std::abs(a) > 1e-12f) ? (-b / (2 * a)) : 0.0f;
        float refined_angle = (bin_idx + peak_offset) * (360.0f / histogram.size());  // Convert bin index to angle

        refined_orientations.push_back(refined_angle);
    }
    return refined_orientations;
}
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<float>> SIFT::build_orientation_histograms(const std::vector<std::tuple<int, float, float, float>> & refined_keypoints, std::vector<std::vector<Eigen::MatrixXf>> & gauss_pyramid, const std::vector<double> & sigmas) {
    std::vector<std::vector<float>> keypoint_orientations(refined_keypoints.size());

    for (size_t k = 0; k < refined_keypoints.size(); k++) {
        const auto & [octave_idx, layer_idx, x, y] = refined_keypoints[k];

        int gauss_layer_idx = static_cast<int>(std::round(layer_idx)) + 1;

        Eigen::MatrixXf & img = gauss_pyramid[octave_idx][gauss_layer_idx];
        double sigma = sigmas[gauss_layer_idx];

        double weight_sigma = 1.5*sigma;  // Weighted kernel for the magnitudes
        int radius = std::round(4.5*sigma);  // Size of neighbourhood (3*weighted sigma = 3 std deviations of weighting kernel)

        int x_center = (int)std::round(x);
        int y_center = (int)std::round(y);

        std::vector<std::tuple<int, int, double, double>> neighbours_with_grads = get_neighbourhood_regions_with_gradient(img, x_center, y_center, radius);

        std::vector<float> histogram = bin_orientation_votes(neighbours_with_grads, weight_sigma, ORIENT_BIN_NUMS);

        // Circular histogram smoothing before peak detection
            // Suppresses bin-quantization noise before applying the 80%-of-max peak rule (same as in OpenCV)
        for (int pass = 0; pass < ORIENT_SMOOTH_PASSES; pass++) {
            std::vector<float> smoothed(histogram.size());
            for (size_t i = 0; i < histogram.size(); i++) {
                size_t idx_left2 = (i - 2 + histogram.size()) % histogram.size();
                size_t idx_left1 = (i - 1 + histogram.size()) % histogram.size();
                size_t idx_right1 = (i + 1) % histogram.size();
                size_t idx_right2 = (i + 2) % histogram.size();
                // Binomial (1,4,6,4,1)/16 weights, matching OpenCV's orientation histogram smoothing
                smoothed[i] = (histogram[idx_left2] + 4.0f*histogram[idx_left1] + 6.0f*histogram[i]
                             + 4.0f*histogram[idx_right1] + histogram[idx_right2]) / 16.0f;
            }
            histogram = smoothed;
        }

        // Find dominant direction
        float global_max = *std::max_element(histogram.begin(), histogram.end());
        std::vector<int> local_max_args;

        
        // Look for local maxima in the histogram that are above 80% of the global maximum
        for (size_t i = 0; i < histogram.size(); i++) {
            size_t idx_left = (i - 1 + histogram.size()) % histogram.size();
            size_t idx_right = (i + 1) % histogram.size();
            
            if (histogram[i] > global_max * MAX_PEAK_RATIO 
                && histogram[i] >= histogram[idx_left] 
                && histogram[i] >= histogram[idx_right]) {
                local_max_args.push_back((int)i);
            }
        }

        // Refine orientation peaks (sub-bin localization)
            // Interpolate by fitting a parabola to the peak and its two neighbors
        std::vector<float> refined_orientations = refine_orientation_peaks(histogram, local_max_args);

        keypoint_orientations[k] = std::move(refined_orientations);
    }

    return keypoint_orientations;
}
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> compute_descriptor(const Eigen::MatrixXf & gauss_img, float x_kp, float y_kp, double sigma, float orientation_deg,
                                       int subregion_size, int orientation_bins, float magnitude_threshold) {
    std::vector<float> descriptor(subregion_size * subregion_size * orientation_bins, 0.0f);

    double orientation_rad = orientation_deg * M_PI / 180.0;
    double hist_width = 3.0 * sigma;
    int radius = static_cast<int>(std::round(hist_width * std::sqrt(2.0) * (subregion_size + 1) / 2.0)); // Radius of the descriptor's support region (same as in OpenCV)
    double weight_sigma = 0.5 * subregion_size;  // Half the width of the descriptor's subregion

    int x_center = static_cast<int>(std::round(x_kp));
    int y_center = static_cast<int>(std::round(y_kp));

    // Iterate over the pixels in the support region of the keypoint
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            // Pixel's coordinates in the image
            int x = x_center + dx;
            int y = y_center + dy;
            if (x < 1 || x >= gauss_img.rows() - 1 || y < 1 || y >= gauss_img.cols() - 1) continue;

            // Offset from the (sub-pixel) keypoint location
            float offset_x = x - x_kp;
            float offset_y = y - y_kp;

            // Step 1: rotate into the keypoint's own frame
                // Rotate so each pixel around the keypoint describes its position relative to the keypoint's orientation
            float x_rot =  offset_x * std::cos(orientation_rad) + offset_y * std::sin(orientation_rad);
            float y_rot = -offset_x * std::sin(orientation_rad) + offset_y * std::cos(orientation_rad);

            // Step 2: fractional subregion coordinates
                // Convert the rotated coordinates into the descriptor's subregion grid (4x4)
            float bin_x = x_rot / hist_width + subregion_size / 2.0f - 0.5f;
            float bin_y = y_rot / hist_width + subregion_size / 2.0f - 0.5f;

            if (bin_x <= -1 || bin_x >= subregion_size || bin_y <= -1 || bin_y >= subregion_size)
                continue;

            // Step 3: express gradient magnitude/angle relative to the keypoint's orientation
            auto [magnitude, angle] = compute_pixel_gradient(gauss_img, x, y);
            double angle_deg = angle * 180.0 / M_PI;
            double angle_rel_deg = std::fmod(angle_deg - orientation_deg, 360.0);
            if (angle_rel_deg < 0)
                angle_rel_deg += 360.0;

            float bin_o = angle_rel_deg / (360.0f / orientation_bins);

            // Gaussian weight centered on the keypoint (in subregion units)
            float rx = x_rot / hist_width;
            float ry = y_rot / hist_width;
            double weight = std::exp(-(std::pow(rx, 2) + std::pow(ry, 2)) / (2.0 * weight_sigma * weight_sigma));
            float vote = magnitude * weight;

            // Step 5: trilinear interpolation across the 8 nearest (subregion_x, subregion_y, angle_bin) bins
            int x0 = static_cast<int>(std::floor(bin_x));
            int y0 = static_cast<int>(std::floor(bin_y));
            int o0 = static_cast<int>(std::floor(bin_o));
            float frac_x = bin_x - x0;
            float frac_y = bin_y - y0;
            float frac_o = bin_o - o0;

            for (int ddx = 0; ddx <= 1; ddx++) {
                for (int ddy = 0; ddy <= 1; ddy++) {
                    for (int ddo = 0; ddo <= 1; ddo++) {
                        int xi = x0 + ddx, yi = y0 + ddy, oi = (o0 + ddo) % orientation_bins;
                        if (xi < 0 || xi >= subregion_size || yi < 0 || yi >= subregion_size)
                            continue;  // outside 4x4 grid

                        float w = (ddx ? frac_x : 1 - frac_x)
                                * (ddy ? frac_y : 1 - frac_y)
                                * (ddo ? frac_o : 1 - frac_o);
                        descriptor[(xi * subregion_size + yi) * orientation_bins + oi] += vote * w;
                    }
                }
            }
        }
    }

    // Step 6: normalize the descriptor to unit length
        // Brightness invariance 
    
    // Wrapper for better readability
    auto normalize = [&descriptor]() {
        float norm = std::sqrt(std::inner_product(descriptor.begin(), descriptor.end(), descriptor.begin(), 0.0f));
        for (auto & v : descriptor) v /= (norm + 1e-7f);
    };
    
    normalize();  // Normalize to unit length

    for (auto & v : descriptor)
        v = std::min(v, magnitude_threshold);  // Clipping to reduce the influence of large gradients (illumination changes)
    
    normalize();  // Need normalization again after clipping

    return descriptor;
}
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Keypoint> SIFT::generate_descriptors(const std::vector<std::tuple<int, float, float, float>> & refined_keypoints,
                                                  const std::vector<std::vector<Eigen::MatrixXf>> & gauss_pyramid,
                                                  const std::vector<double> & sigmas,
                                                  const std::vector<std::vector<float>> & keypoint_orientations) {
    std::vector<Keypoint> keypoints;
    for (size_t k = 0; k < refined_keypoints.size(); k++) {
        const auto & [octave_idx, layer_idx, x, y] = refined_keypoints[k];

        int gauss_layer_idx = static_cast<int>(std::round(layer_idx)) + 1;
        const Eigen::MatrixXf & gauss_img = gauss_pyramid[octave_idx][gauss_layer_idx];
        double sigma = sigmas[gauss_layer_idx];

        // Scale to transform (x, y, s) from octave own pixel grid to the original image's pixel grid
            // Octave o was downscaled by 2^o
            // Original image was upscaled by 2x -> 2^(o-1)
        float to_input_scale = std::pow(2.0f, static_cast<float>(octave_idx) - 1.0f);

        for (float orientation_deg : keypoint_orientations[k]) {
            Keypoint kp;
            kp.x = x * to_input_scale;
            kp.y = y * to_input_scale;
            kp.sigma = sigma * to_input_scale;
            kp.orientation_deg = orientation_deg;
            kp.descriptor = compute_descriptor(gauss_img, x, y, sigma, orientation_deg,
                                                DESC_SUBREGION_SIZE, DESC_ORIENTATION_BINS, DESC_MAGNITUDE_THRESHOLD);
            keypoints.push_back(std::move(kp));
        }
    }

    return keypoints;
}
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Keypoint> SIFT::detect_features(const Eigen::MatrixXf & img) {
    // 1. Build scale space
        // Get Gaussian and DoG pyramid

    Eigen::MatrixXf base_img = upsample(img);
    set_number_of_octaves(base_img);

    std::vector<std::vector<Eigen::MatrixXf>> gauss_pyramid;
    // Total blur at each layer
    std::vector<double> sigmas(m_octave_layers_num + 3);  // (s+3) layers per octave -> extrema detection covers whole octave
    std::vector<std::vector<Eigen::MatrixXf>> dog_pyramid; // Difference of Gaussians between each layer (at octaves)

    build_scale_space(base_img, sigmas, gauss_pyramid, dog_pyramid);

    // 2. Detect keypoints
        // Find candidate points (local extrema)
        // Filter candidate points based on (sub-pixel localization, contrast, edge response)
    std::vector<std::tuple<int, float, float, float, float>> scored_keypoints = detect_keypoints(dog_pyramid);

    // Like cv::SIFT's nfeatures: a cap, not a target. If more candidates passed the
    // contrast/edge filters than max_features, keep only the top-N by contrast response.
    if (m_max_features > 0 && (int)scored_keypoints.size() > m_max_features) {
        std::partial_sort(scored_keypoints.begin(), scored_keypoints.begin() + m_max_features,
                           scored_keypoints.end(),
                           [](const auto & a, const auto & b) { return std::get<4>(a) > std::get<4>(b); });
        scored_keypoints.resize(m_max_features);
        std::cout << "[SIFT] Capped to top " << m_max_features << " keypoints by contrast response" << std::endl;
    }

    std::vector<std::tuple<int, float, float, float>> refined_keypoints;
    refined_keypoints.reserve(scored_keypoints.size());
    for (const auto & [octave_idx, layer_idx, x, y, response] : scored_keypoints)
        refined_keypoints.emplace_back(octave_idx, layer_idx, x, y);

    // 3. Histogram Orientation
        // Assign orientation(s) for rotation-invariance (based on intensity gradients in the local neighborhood)
    std::vector<std::vector<float>> keypoint_orientations = build_orientation_histograms(refined_keypoints, gauss_pyramid, sigmas);

    int multi_orientation_count = 0;
    for (const auto & orientations : keypoint_orientations)
        if (orientations.size() > 1) multi_orientation_count++;
    std::cout << "[SIFT] " << multi_orientation_count << "/" << keypoint_orientations.size()
               << " keypoints got more than one dominant orientation (each becomes a separate descriptor)" << std::endl;

    // 4. Descriptor Generation
        // For every (keypoint, orientation) pair, build a 128-dim gradient histogram descriptor
    std::vector<Keypoint> keypoints = generate_descriptors(refined_keypoints, gauss_pyramid, sigmas, keypoint_orientations);

    std::cout << "[SIFT] detect_features done: " << keypoints.size() << " final (keypoint, orientation) descriptors" << std::endl;

    return keypoints;
}
// ─────────────────────────────────────────────────────────────────────────────

