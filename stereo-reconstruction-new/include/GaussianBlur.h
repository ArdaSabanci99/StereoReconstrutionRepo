#pragma once
#include "Eigen.h"

class GaussianBlur {
        static double gaussian1d(int x, double sigma);
        static Eigen::VectorXf get_kernel1d(int kernel_size, double sigma); // TODO: pass as a function

public:
    static Eigen::MatrixXf apply_gaussian2d(const Eigen::MatrixXf & img, int kernel_size, double sigma);
};