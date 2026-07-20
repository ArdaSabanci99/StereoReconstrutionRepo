#pragma once
#include "Eigen.h"

/**
 * @brief Class for applying Gaussian blur to images.
 */
class GaussianBlur {
    /** 
     * @brief Computes the 1D Gaussian function. 
     */
    
     static double gaussian1d(int x, double sigma);
    /** 
     * @brief Generates a 1D Gaussian kernel. 
     * 
     * @param[in] kernel_size Size of the kernel. Expecting an odd size.
     * @param[in] sigma Standard deviation of the Gaussian function.
     * @return Eigen::VectorXf 1D Gaussian kernel.
     */
    static Eigen::VectorXf get_kernel1d(int kernel_size, double sigma);

public:
    /** 
     * @brief Applies 2D Gaussian blur to an image.
     *        Separable convolution is used: first horizontal, then vertical.
     * 
     * @param[in] img Input image.
     * @param[in] kernel_size Size of the kernel. Expecting an odd size.
     * @param[in] sigma Standard deviation of the Gaussian function.
     * @return Eigen::MatrixXf Blurred image.
     */
    static Eigen::MatrixXf apply_gaussian2d(const Eigen::MatrixXf & img, int kernel_size, double sigma);
};