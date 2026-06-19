#pragma once

// Suppress GLOG/Windows constant conflicts
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <flann/flann.hpp>

#include "Eigen.h"
#include "PointCloud.h"
#include "NearestNeighbor.h"
#include "ProcrustesAligner.h"


// ── Pose increment helper (angle-axis + translation, 6 DOF) ──────────────

template <typename T>
static inline void fillVector(const Vector3f& input, T* output) {
    output[0] = T(input[0]);
    output[1] = T(input[1]);
    output[2] = T(input[2]);
}

template <typename T>
class PoseIncrement {
public:
    explicit PoseIncrement(T* const array) : m_array{array} {}

    void setZero() { for (int i = 0; i < 6; ++i) m_array[i] = T(0); }

    T* getData() const { return m_array; }

    void apply(T* inputPoint, T* outputPoint) const {
        const T* rotation    = m_array;
        const T* translation = m_array + 3;
        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);
        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

    static Matrix4f convertToMatrix(const PoseIncrement<double>& poseIncrement) {
        double* pose        = poseIncrement.getData();
        double* rotation    = pose;
        double* translation = pose + 3;

        double rotMat[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotMat);

        Matrix4f matrix = Matrix4f::Identity();
        matrix(0,0) = float(rotMat[0]); matrix(0,1) = float(rotMat[3]); matrix(0,2) = float(rotMat[6]); matrix(0,3) = float(translation[0]);
        matrix(1,0) = float(rotMat[1]); matrix(1,1) = float(rotMat[4]); matrix(1,2) = float(rotMat[7]); matrix(1,3) = float(translation[1]);
        matrix(2,0) = float(rotMat[2]); matrix(2,1) = float(rotMat[5]); matrix(2,2) = float(rotMat[8]); matrix(2,3) = float(translation[2]);
        return matrix;
    }

private:
    T* m_array;
};


// ── Ceres cost functions ──────────────────────────────────────────────────

class PointToPointConstraint {
public:
    PointToPointConstraint(const Vector3f& src, const Vector3f& tgt, float weight)
        : m_src{src}, m_tgt{tgt}, m_weight{weight} {}

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // TODO: Apply pose to m_src, subtract m_tgt, scale by sqrt(m_weight).
        // PoseIncrement<T> pi(const_cast<T*>(pose));
        // T srcT[3] = {T(m_src.x()), T(m_src.y()), T(m_src.z())};
        // T transformed[3];
        // pi.apply(srcT, transformed);
        // residuals[0] = T(sqrt(m_weight)) * (transformed[0] - T(m_tgt.x()));
        // residuals[1] = T(sqrt(m_weight)) * (transformed[1] - T(m_tgt.y()));
        // residuals[2] = T(sqrt(m_weight)) * (transformed[2] - T(m_tgt.z()));
        residuals[0] = T(0); residuals[1] = T(0); residuals[2] = T(0);
        return true;
    }

    static ceres::CostFunction* create(const Vector3f& src, const Vector3f& tgt, float weight) {
        return new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
            new PointToPointConstraint(src, tgt, weight));
    }

private:
    const Vector3f m_src, m_tgt;
    const float    m_weight;
};


class PointToPlaneConstraint {
public:
    PointToPlaneConstraint(const Vector3f& src, const Vector3f& tgt,
                            const Vector3f& tgtNormal, float weight)
        : m_src{src}, m_tgt{tgt}, m_tgtNormal{tgtNormal}, m_weight{weight} {}

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // TODO: Apply pose to m_src, compute dot product with m_tgtNormal.
        // PoseIncrement<T> pi(const_cast<T*>(pose));
        // T srcT[3] = {T(m_src.x()), T(m_src.y()), T(m_src.z())};
        // T transformed[3];
        // pi.apply(srcT, transformed);
        // residuals[0] = T(sqrt(m_weight)) * (
        //     (transformed[0] - T(m_tgt.x())) * T(m_tgtNormal.x()) +
        //     (transformed[1] - T(m_tgt.y())) * T(m_tgtNormal.y()) +
        //     (transformed[2] - T(m_tgt.z())) * T(m_tgtNormal.z()));
        residuals[0] = T(0);
        return true;
    }

    static ceres::CostFunction* create(const Vector3f& src, const Vector3f& tgt,
                                        const Vector3f& normal, float weight) {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
            new PointToPlaneConstraint(src, tgt, normal, weight));
    }

private:
    const Vector3f m_src, m_tgt, m_tgtNormal;
    const float    m_weight;
};


// ── ICP Optimizer — Abstract Base ────────────────────────────────────────

class ICPOptimizer {
public:
    ICPOptimizer() :
        m_bUsePointToPlane{false},
        m_nIterations{20},
        m_nearestNeighborSearch{std::make_unique<NearestNeighborSearchFlann>()}
    {}

    virtual ~ICPOptimizer() {}

    void setMatchingMaxDistance(float d)           { m_nearestNeighborSearch->setMatchingMaxDistance(d); }
    void usePointToPlaneConstraints(bool b)         { m_bUsePointToPlane = b; }
    void setNbOfIterations(unsigned n)              { m_nIterations = n; }

    virtual void estimatePose(const PointCloud& source,
                               const PointCloud& target,
                               Matrix4f& initialPose) = 0;

protected:
    bool     m_bUsePointToPlane;
    unsigned m_nIterations;
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

    std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& pts, const Matrix4f& pose) {
        std::vector<Vector3f> out;
        out.reserve(pts.size());
        const auto R = pose.block(0,0,3,3);
        const auto t = pose.block(0,3,3,1);
        for (const auto& p : pts) out.push_back(R * p + t);
        return out;
    }

    std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& normals, const Matrix4f& pose) {
        std::vector<Vector3f> out;
        out.reserve(normals.size());
        const auto R = pose.block(0,0,3,3);
        const Matrix3f Rinv_T = R.inverse().transpose();
        for (const auto& n : normals) out.push_back((Rinv_T * n).normalized());
        return out;
    }

    void pruneCorrespondences(const std::vector<Vector3f>& srcNormals,
                               const std::vector<Vector3f>& tgtNormals,
                               std::vector<Match>& matches) {
        for (unsigned i = 0; i < srcNormals.size(); i++) {
            Match& m = matches[i];
            if (m.idx >= 0) {
                // TODO: Invalidate match if angle between normals > 60°
                // float cosAngle = srcNormals[i].dot(tgtNormals[m.idx]);
                // if (cosAngle < 0.5f) m.idx = -1;
            }
        }
    }
};


// ── Ceres-based ICP optimizer ─────────────────────────────────────────────

class CeresICPOptimizer : public ICPOptimizer {
public:
    CeresICPOptimizer() {}

    void estimatePose(const PointCloud& source,
                       const PointCloud& target,
                       Matrix4f& initialPose) override {
        m_nearestNeighborSearch->buildIndex(target.getPoints());
        Matrix4f estimatedPose = initialPose;

        double incrementArray[6];
        auto poseIncrement = PoseIncrement<double>(incrementArray);
        poseIncrement.setZero();

        for (unsigned i = 0; i < m_nIterations; ++i) {
            std::cout << "[ICP-Ceres] Iteration " << i+1 << "/" << m_nIterations << "\n";

            auto transformedPoints  = transformPoints (source.getPoints(),  estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);

            ceres::Problem problem;
            prepareConstraints(transformedPoints, target.getPoints(),
                               target.getNormals(), matches, poseIncrement, problem);

            ceres::Solver::Options options;
            configureSolver(options);

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << "\n";

            estimatedPose = PoseIncrement<double>::convertToMatrix(poseIncrement) * estimatedPose;
            poseIncrement.setZero();
        }
        initialPose = estimatedPose;
    }

private:
    void configureSolver(ceres::Solver::Options& options) {
        options.trust_region_strategy_type   = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps       = false;
        options.linear_solver_type           = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations           = 1;
        options.num_threads                  = 8;
    }

    void prepareConstraints(const std::vector<Vector3f>& srcPts,
                             const std::vector<Vector3f>& tgtPts,
                             const std::vector<Vector3f>& tgtNormals,
                             const std::vector<Match>&    matches,
                             const PoseIncrement<double>& poseIncrement,
                             ceres::Problem&              problem) const {
        for (unsigned i = 0; i < srcPts.size(); ++i) {
            const auto& m = matches[i];
            if (m.idx < 0) continue;
            if (!srcPts[i].allFinite() || !tgtPts[m.idx].allFinite()) continue;

            // TODO: Add PointToPointConstraint
            // problem.AddResidualBlock(
            //     PointToPointConstraint::create(srcPts[i], tgtPts[m.idx], m.weight),
            //     nullptr, poseIncrement.getData());

            if (m_bUsePointToPlane) {
                if (!tgtNormals[m.idx].allFinite()) continue;
                // TODO: Add PointToPlaneConstraint
                // problem.AddResidualBlock(
                //     PointToPlaneConstraint::create(srcPts[i], tgtPts[m.idx],
                //                                   tgtNormals[m.idx], m.weight),
                //     nullptr, poseIncrement.getData());
            }
        }
    }
};


// ── Linear ICP optimizer (closed-form via Procrustes / linear system) ─────

class LinearICPOptimizer : public ICPOptimizer {
public:
    LinearICPOptimizer() {}

    void estimatePose(const PointCloud& source,
                       const PointCloud& target,
                       Matrix4f& initialPose) override {
        m_nearestNeighborSearch->buildIndex(target.getPoints());
        Matrix4f estimatedPose = initialPose;

        for (unsigned i = 0; i < m_nIterations; ++i) {
            std::cout << "[ICP-Linear] Iteration " << i+1 << "/" << m_nIterations << "\n";

            auto transformedPoints  = transformPoints (source.getPoints(),  estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);

            std::vector<Vector3f> srcMatched, tgtMatched;
            for (size_t j = 0; j < transformedPoints.size(); j++) {
                if (matches[j].idx >= 0) {
                    srcMatched.push_back(transformedPoints[j]);
                    tgtMatched.push_back(target.getPoints()[matches[j].idx]);
                }
            }

            if (m_bUsePointToPlane)
                estimatedPose = estimatePosePointToPlane(srcMatched, tgtMatched,
                                                          target.getNormals()) * estimatedPose;
            else
                estimatedPose = estimatePosePointToPoint(srcMatched, tgtMatched) * estimatedPose;
        }
        initialPose = estimatedPose;
    }

private:
    Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& src,
                                       const std::vector<Vector3f>& tgt) {
        ProcrustesAligner aligner;
        return aligner.estimatePose(src, tgt);
    }

    Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& src,
                                       const std::vector<Vector3f>& tgt,
                                       const std::vector<Vector3f>& tgtNormals) {
        const unsigned n = src.size();
        MatrixXf A = MatrixXf::Zero(4 * n, 6);
        VectorXf b = VectorXf::Zero(4 * n);

        for (unsigned i = 0; i < n; i++) {
            const auto& s = src[i];
            const auto& d = tgt[i];
            const auto& normal = tgtNormals[i];

            // TODO: Add point-to-plane constraints to A and b (rows 4*i + 0..2)
            // Row format: [n_z*s_y - n_y*s_z,  n_x*s_z - n_z*s_x,  n_y*s_x - n_x*s_y,  n_x,  n_y,  n_z]
            // b[4*i] = dot(n, d) - dot(n, s)

            // TODO: Add point-to-point constraints (rows 4*i + 3)

            // TODO: Optionally apply higher weight to point-to-plane rows
        }

        // TODO: Solve  A * x = b  (Eigen least-squares)
        // VectorXf x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
        VectorXf x = VectorXf::Zero(6);

        float alpha = x(0), beta = x(1), gamma = x(2);

        Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                            AngleAxisf(beta,  Vector3f::UnitY()).toRotationMatrix() *
                            AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();
        Vector3f translation = x.tail(3);

        Matrix4f pose = Matrix4f::Identity();
        // TODO: fill pose with rotation and translation
        return pose;
    }
};
