#pragma once
#include "Eigen.h"

// Eigen-based point cloud — used throughout the pipeline and by ICPOptimizer.
// Colors are optional (for PLY output); normals are needed for point-to-plane ICP.
struct PointCloud {
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Vector4uc>       colors;   // RGBA, optional

    // Accessors matching the exercise ICPOptimizer interface
    std::vector<Eigen::Vector3f>&       getPoints()        { return points; }
    const std::vector<Eigen::Vector3f>& getPoints()  const { return points; }
    std::vector<Eigen::Vector3f>&       getNormals()       { return normals; }
    const std::vector<Eigen::Vector3f>& getNormals() const { return normals; }

    bool empty() const { return points.empty(); }
    size_t size() const { return points.size(); }
};
