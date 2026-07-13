// poisson_mesh.cpp
// Poisson surface reconstruction from an oriented point cloud.
//
// Pipeline (all in C++, no external meshing library):
//   1. Load PLY point cloud
//   2. Estimate normals via PCA on k-NN (FLANN KD-tree)
//   3. Orient normals consistently (toward centroid heuristic)
//   4. Scatter normals into a voxel normal field V
//   5. Compute divergence  f = div V
//   6. Solve  Lap(chi) = f  via Gauss-Seidel
//   7. Extract iso-surface at chi = isovalue using Marching Cubes (Exercise 2)
//   8. Save mesh as OBJ
//
// Usage:
//   poisson_mesh <pointcloud.ply> [options]
//     --res  N    voxel grid resolution (default 80)
//     --k    N    neighbours for normal PCA (default 15)
//     --iter N    Gauss-Seidel iterations (default 100)
//     --out  path output OBJ path

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include <Eigen/Dense>
#include <flann/flann.hpp>

// Exercise-2 Marching Cubes
#include "../include/mc/Volume.h"
#include "../include/mc/SimpleMesh.h"
#include "../include/mc/MarchingCubes.h"

namespace fs = std::filesystem;

struct ColoredPoint { Eigen::Vector3f pos; uint8_t r, g, b; };

// ── PLY loader (reads XYZ + RGB if present) ───────────────────────────────────
static std::vector<ColoredPoint> loadPLY(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "[poisson] Cannot open " << path << "\n"; return {}; }

    std::string line;
    int n_verts = 0;
    bool has_color = false;
    bool header_done = false;
    while (std::getline(f, line)) {
        if (line.rfind("element vertex", 0) == 0) n_verts = std::stoi(line.substr(15));
        if (line.find("property uchar red") != std::string::npos) has_color = true;
        if (line == "end_header") { header_done = true; break; }
    }
    if (!header_done || n_verts == 0) { std::cerr << "[poisson] Bad PLY header\n"; return {}; }

    std::vector<ColoredPoint> pts;
    pts.reserve(n_verts);
    for (int i = 0; i < n_verts; ++i) {
        float x, y, z; int r = 128, g = 128, b = 128;
        if (!(f >> x >> y >> z)) break;
        if (has_color) f >> r >> g >> b;
        f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        pts.push_back({{x, y, z}, (uint8_t)r, (uint8_t)g, (uint8_t)b});
    }
    std::cout << "[poisson] Loaded " << pts.size() << " points from " << path
              << (has_color ? " (with color)" : "") << "\n";
    return pts;
}

// ── Normal estimation via PCA on k-NN (FLANN) ────────────────────────────────
static std::vector<Eigen::Vector3f> estimateNormals(
        const std::vector<ColoredPoint>& pts, int k) {

    const int N = (int)pts.size();
    // Build FLANN dataset
    std::vector<float> data(N * 3);
    for (int i = 0; i < N; ++i) {
        data[3*i+0] = pts[i].pos.x();
        data[3*i+1] = pts[i].pos.y();
        data[3*i+2] = pts[i].pos.z();
    }
    flann::Matrix<float> dataset(data.data(), N, 3);
    flann::Index<flann::L2<float>> index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();

    k = std::min(k, N - 1);
    std::vector<std::vector<int>>   indices;
    std::vector<std::vector<float>> dists;
    index.knnSearch(dataset, indices, dists, k + 1, flann::SearchParams(128));

    // Centroid for orientation
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (auto& p : pts) centroid += p.pos;
    centroid /= (float)N;

    std::vector<Eigen::Vector3f> normals(N);
    for (int i = 0; i < N; ++i) {
        // Gather neighbours (skip self = index 0)
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (int j = 1; j <= k && j < (int)indices[i].size(); ++j)
            mean += pts[indices[i][j]].pos;
        mean /= (float)k;

        // 3×k covariance matrix
        Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
        for (int j = 1; j <= k && j < (int)indices[i].size(); ++j) {
            Eigen::Vector3f d = pts[indices[i][j]].pos - mean;
            C += d * d.transpose();
        }

        // Smallest eigenvector = normal
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(C);
        Eigen::Vector3f n = eig.eigenvectors().col(0); // smallest eigenvalue

        // Orient toward centroid
        if (n.dot(centroid - pts[i].pos) < 0) n = -n;
        normals[i] = n;
    }
    std::cout << "[poisson] Normal estimation done (k=" << k << ")\n";
    return normals;
}

// ── Grid helpers ──────────────────────────────────────────────────────────────
inline int idx3(int x, int y, int z, int G) {
    return x * G * G + y * G + z;
}

// ── Poisson solve ─────────────────────────────────────────────────────────────
// Solves  Lap(chi) = div(V)  on a G^3 grid with Dirichlet BC = 0 on boundary.
// V is the normal vector field (3 components per voxel, stored interleaved xyz).
static std::vector<double> poissonSolve(
        const std::vector<float>& Vx,
        const std::vector<float>& Vy,
        const std::vector<float>& Vz,
        int G, int iters) {

    // Compute RHS: divergence of V via central finite differences
    std::vector<double> rhs(G*G*G, 0.0);
    float h = 1.0f / (G - 1);
    for (int x = 1; x < G-1; ++x)
        for (int y = 1; y < G-1; ++y)
            for (int z = 1; z < G-1; ++z) {
                float dvx = (Vx[idx3(x+1,y,z,G)] - Vx[idx3(x-1,y,z,G)]) / (2*h);
                float dvy = (Vy[idx3(x,y+1,z,G)] - Vy[idx3(x,y-1,z,G)]) / (2*h);
                float dvz = (Vz[idx3(x,y,z+1,G)] - Vz[idx3(x,y,z-1,G)]) / (2*h);
                rhs[idx3(x,y,z,G)] = dvx + dvy + dvz;
            }

    // Gauss-Seidel: chi_new[i] = (sum of 6 neighbors + h²*rhs[i]) / 6
    double h2 = (double)(h * h);
    std::vector<double> chi(G*G*G, 0.0);
    for (int it = 0; it < iters; ++it) {
        for (int x = 1; x < G-1; ++x)
            for (int y = 1; y < G-1; ++y)
                for (int z = 1; z < G-1; ++z) {
                    double sum =
                        chi[idx3(x-1,y,z,G)] + chi[idx3(x+1,y,z,G)] +
                        chi[idx3(x,y-1,z,G)] + chi[idx3(x,y+1,z,G)] +
                        chi[idx3(x,y,z-1,G)] + chi[idx3(x,y,z+1,G)];
                    chi[idx3(x,y,z,G)] = (sum - h2 * rhs[idx3(x,y,z,G)]) / 6.0;
                }
        if ((it+1) % 20 == 0)
            std::cout << "[poisson] Gauss-Seidel iter " << it+1 << "/" << iters << "\n";
    }
    return chi;
}

// ── Save colored PLY ──────────────────────────────────────────────────────────
static void savePLY(SimpleMesh& mesh,
                    const std::vector<ColoredPoint>& cloud,
                    const std::string& path) {
    // Build FLANN index over original cloud to transfer colors to mesh vertices
    const int N = (int)cloud.size();
    std::vector<float> data(N * 3);
    for (int i = 0; i < N; ++i) {
        data[3*i+0] = cloud[i].pos.x();
        data[3*i+1] = cloud[i].pos.y();
        data[3*i+2] = cloud[i].pos.z();
    }
    flann::Matrix<float> dataset(data.data(), N, 3);
    flann::Index<flann::L2<float>> index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();

    const auto& verts = mesh.GetVertices();
    const auto& tris  = mesh.GetTriangles();

    // Query nearest neighbor for each mesh vertex
    std::vector<float> qdata(verts.size() * 3);
    for (int i = 0; i < (int)verts.size(); ++i) {
        qdata[3*i+0] = verts[i].x(); qdata[3*i+1] = verts[i].y(); qdata[3*i+2] = verts[i].z();
    }
    flann::Matrix<float> queries(qdata.data(), verts.size(), 3);
    std::vector<std::vector<int>>   nn_idx;
    std::vector<std::vector<float>> nn_dist;
    index.knnSearch(queries, nn_idx, nn_dist, 1, flann::SearchParams(64));

    std::ofstream f(path);
    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << verts.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "element face " << tris.size() << "\n";
    f << "property list uchar int vertex_indices\n";
    f << "end_header\n";
    for (int i = 0; i < (int)verts.size(); ++i) {
        int ci = nn_idx[i][0];
        f << verts[i].x() << " " << verts[i].y() << " " << verts[i].z()
          << " " << (int)cloud[ci].r << " " << (int)cloud[ci].g << " " << (int)cloud[ci].b << "\n";
    }
    for (const auto& t : tris)
        f << "3 " << t.idx0 << " " << t.idx1 << " " << t.idx2 << "\n";

    std::cout << "[poisson] Saved " << path
              << " (" << verts.size() << " verts, " << tris.size() << " faces)\n";
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: poisson_mesh <cloud.ply> [--res N] [--k N] [--iter N] [--out path]\n";
        return 1;
    }

    std::string ply_path = argv[1];
    int  G    = 80;
    int  k    = 15;
    int  iters= 100;
    std::string out_path;

    for (int i = 2; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--res"  && i+1<argc) G        = std::stoi(argv[++i]);
        else if (a == "--k"    && i+1<argc) k        = std::stoi(argv[++i]);
        else if (a == "--iter" && i+1<argc) iters    = std::stoi(argv[++i]);
        else if (a == "--out"  && i+1<argc) out_path = argv[++i];
    }

    if (out_path.empty()) {
        fs::path p(ply_path);
        out_path = (p.parent_path() / (p.stem().string() + "_poisson.ply")).string();
    }

    // ── 1. Load ──────────────────────────────────────────────────────────────
    auto pts = loadPLY(ply_path);
    if (pts.empty()) return 1;

    // ── 2. Estimate normals ──────────────────────────────────────────────────
    auto normals = estimateNormals(pts, k);

    // ── 3. Compute bounding box → grid mapping ───────────────────────────────
    Eigen::Vector3f bmin = pts[0].pos, bmax = pts[0].pos;
    for (auto& p : pts) { bmin = bmin.cwiseMin(p.pos); bmax = bmax.cwiseMax(p.pos); }
    // Add 5% padding
    Eigen::Vector3f pad = (bmax - bmin) * 0.05f;
    bmin -= pad; bmax += pad;
    Eigen::Vector3f range = bmax - bmin;

    auto worldToGrid = [&](const Eigen::Vector3f& p) -> Eigen::Vector3f {
        return (p - bmin).cwiseQuotient(range) * (float)(G - 1);
    };

    // ── 4. Scatter normals into voxel field (trilinear splatting) ────────────
    std::vector<float> Vx(G*G*G, 0.f), Vy(G*G*G, 0.f), Vz(G*G*G, 0.f);
    std::vector<float> W (G*G*G, 0.f);

    for (int i = 0; i < (int)pts.size(); ++i) {
        Eigen::Vector3f gp = worldToGrid(pts[i].pos);
        int gx = (int)gp.x(), gy = (int)gp.y(), gz = (int)gp.z();
        if (gx < 0 || gx >= G-1 || gy < 0 || gy >= G-1 || gz < 0 || gz >= G-1) continue;

        float tx = gp.x() - gx, ty = gp.y() - gy, tz = gp.z() - gz;
        // Trilinear weights for 8 corners
        for (int dx = 0; dx <= 1; ++dx)
        for (int dy = 0; dy <= 1; ++dy)
        for (int dz = 0; dz <= 1; ++dz) {
            float w = (dx ? tx : 1-tx) * (dy ? ty : 1-ty) * (dz ? tz : 1-tz);
            int id = idx3(gx+dx, gy+dy, gz+dz, G);
            Vx[id] += w * normals[i].x();
            Vy[id] += w * normals[i].y();
            Vz[id] += w * normals[i].z();
            W [id] += w;
        }
    }
    // Normalize
    for (int i = 0; i < G*G*G; ++i)
        if (W[i] > 1e-6f) { Vx[i] /= W[i]; Vy[i] /= W[i]; Vz[i] /= W[i]; }

    std::cout << "[poisson] Normal field scattered into " << G << "^3 grid\n";

    // ── 5–6. Poisson solve ───────────────────────────────────────────────────
    auto chi = poissonSolve(Vx, Vy, Vz, G, iters);

    // ── 7. Determine isovalue: mean chi at surface points ────────────────────
    double iso_sum = 0; int iso_cnt = 0;
    for (int i = 0; i < (int)pts.size(); ++i) {
        Eigen::Vector3f gp = worldToGrid(pts[i].pos);
        int gx = std::clamp((int)gp.x(), 0, G-1);
        int gy = std::clamp((int)gp.y(), 0, G-1);
        int gz = std::clamp((int)gp.z(), 0, G-1);
        iso_sum += chi[idx3(gx,gy,gz,G)];
        ++iso_cnt;
    }
    double isovalue = (iso_cnt > 0) ? iso_sum / iso_cnt : 0.0;
    std::cout << "[poisson] Isovalue = " << isovalue << "\n";

    // ── 8. Load chi into Volume and run Marching Cubes ───────────────────────
    Vector3d vmin_d(bmin.x(), bmin.y(), bmin.z());
    Vector3d vmax_d(bmax.x(), bmax.y(), bmax.z());
    Volume vol(vmin_d, vmax_d, G, G, G);
    for (int x = 0; x < G; ++x)
        for (int y = 0; y < G; ++y)
            for (int z = 0; z < G; ++z)
                vol.set(x, y, z, chi[idx3(x,y,z,G)] - isovalue);

    SimpleMesh mesh;
    for (int x = 0; x < G-1; ++x)
        for (int y = 0; y < G-1; ++y)
            for (int z = 0; z < G-1; ++z)
                ProcessVolumeCell(&vol, x, y, z, 0.0, &mesh);

    std::cout << "[poisson] Marching Cubes: " << mesh.GetVertices().size()
              << " verts, " << mesh.GetTriangles().size() << " faces\n";

    // ── 9. Save colored PLY ──────────────────────────────────────────────────
    fs::create_directories(fs::path(out_path).parent_path());
    savePLY(mesh, pts, out_path);
    return 0;
}
