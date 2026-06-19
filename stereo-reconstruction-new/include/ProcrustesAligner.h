#pragma once
#include "Eigen.h"

// Closed-form rotation + translation estimation via Procrustes (SVD).
// Used by LinearICPOptimizer for point-to-point step.
class ProcrustesAligner {
public:
    Matrix4f estimatePose(const std::vector<Vector3f>& sourcePoints,
                          const std::vector<Vector3f>& targetPoints) {
        ASSERT(sourcePoints.size() == targetPoints.size() &&
               "Source and target must have the same number of points.");

        Vector3f sourceMean = computeMean(sourcePoints);
        Vector3f targetMean = computeMean(targetPoints);

        Matrix3f rotation    = estimateRotation(sourcePoints, sourceMean,
                                                targetPoints, targetMean);
        Vector3f translation = targetMean - sourceMean;

        Matrix4f pose = Matrix4f::Identity();
        pose.block(0, 0, 3, 3) = rotation;
        pose.block(0, 3, 3, 1) = rotation * translation - rotation * targetMean + targetMean;
        return pose;
    }

private:
    Vector3f computeMean(const std::vector<Vector3f>& pts) {
        Vector3f mean = Vector3f::Zero();
        for (const auto& p : pts) mean += p;
        return mean / (float)pts.size();
    }

    Matrix3f estimateRotation(const std::vector<Vector3f>& src, const Vector3f& srcMean,
                               const std::vector<Vector3f>& tgt, const Vector3f& tgtMean) {
        const int n = (int)src.size();
        MatrixXf S(n, 3), T(n, 3);
        for (int i = 0; i < n; ++i) {
            S.row(i) = (src[i] - srcMean).transpose();
            T.row(i) = (tgt[i] - tgtMean).transpose();
        }

        Matrix3f A = T.transpose() * S;
        JacobiSVD<Matrix3f> svd(A, ComputeFullU | ComputeFullV);

        float d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
        Matrix3f D = Matrix3f::Identity();
        D(2, 2) = d;

        return svd.matrixU() * D * svd.matrixV().transpose();
    }
};
