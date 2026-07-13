#include "GaussianBlur.h"
#include <cmath>

double GaussianBlur::gaussian1d(int x, double sigma) {
    return std::exp(-(std::pow(x, 2)) / (2.0 * std::pow(sigma, 2)));
}

Eigen::VectorXf GaussianBlur::get_kernel1d(int kernel_size, double sigma) {
    int radius = kernel_size / 2;
    Eigen::VectorXf kernel(kernel_size);

    double sum = 0.0;

    for (int i = -radius; i <= radius; i++) {
        double val = gaussian1d(i, sigma);

        kernel(i + radius) = val;

        sum += val;
    }

    kernel /= sum;
    return kernel;
}
Eigen::MatrixXf GaussianBlur::apply_gaussian2d(const Eigen::MatrixXf & img, int kernel_size, double sigma) {
    Eigen::VectorXf kernel = get_kernel1d(kernel_size, sigma);
    int radius = kernel_size / 2;
    
    // Apply horizontal blur
    Eigen::MatrixXf padded_img_horizontal(img.rows(), img.cols() + 2*radius);
    padded_img_horizontal.block(0, radius, img.rows(), img.cols()) = img;  // copy the current image
    for (int r = 0; r < radius; r++) {
        // Copying the nearest column
        padded_img_horizontal.col(r) = img.col(0);
        padded_img_horizontal.col(radius + img.cols() + r) = img.col(img.cols() - 1);
    }

    Eigen::MatrixXf blurred_img_horizontal = Eigen::MatrixXf::Zero(img.rows(), img.cols());
    for (int i = 0; i < img.rows(); i++) {
        for (int j = 0; j < img.cols(); j++) {
            blurred_img_horizontal(i, j) = padded_img_horizontal.row(i).segment(j, kernel_size).dot(kernel);
        }
    }

    // Apply vertical blur
    Eigen::MatrixXf padded_img_vertical(img.rows() + 2*radius, img.cols());

    // Copy the horizontally blurred image -> adding vertical blur = 2D Gaussian blur
    padded_img_vertical.block(radius, 0, img.rows(), img.cols()) = blurred_img_horizontal;
    for (int r = 0; r < radius; r++) {
        // Copying the nearest row
        padded_img_vertical.row(r) = blurred_img_horizontal.row(0);
        padded_img_vertical.row(radius + img.rows() + r) = blurred_img_horizontal.row(img.rows() - 1);
    }

    Eigen::MatrixXf blurred_img = Eigen::MatrixXf::Zero(img.rows(), img.cols());
    for (int j = 0; j < img.cols(); j++) {
        for (int i = 0; i < img.rows(); i++) {
            blurred_img(i, j) = padded_img_vertical.col(j).segment(i, kernel_size).dot(kernel);
        }
    }

    return blurred_img;
}
