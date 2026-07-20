#include "dtu_eval.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>


namespace {

struct KDTree3 {
    struct Node { int idx; int left = -1, right = -1; int axis = 0; };
    const std::vector<Eigen::Vector3f>* pts = nullptr;
    std::vector<int>  indices;
    std::vector<Node> nodes;
    int root = -1;

    void build(const std::vector<Eigen::Vector3f>& p) {
        pts = &p;
        indices.resize(p.size());
        for (size_t i = 0; i < p.size(); ++i) indices[i] = (int)i;
        nodes.clear();
        nodes.reserve(p.size());
        root = buildRec(0, (int)indices.size(), 0);
    }

    int buildRec(int lo, int hi, int depth) {
        if (lo >= hi) return -1;
        const int axis = depth % 3;
        const int mid  = (lo + hi) / 2;
        std::nth_element(indices.begin() + lo, indices.begin() + mid,
                         indices.begin() + hi,
                         [&](int a, int b) {
                             return (*pts)[a][axis] < (*pts)[b][axis];
                         });
        const int nodeId = (int)nodes.size();
        nodes.push_back({ indices[mid], -1, -1, axis });
        const int l = buildRec(lo, mid, depth + 1);   // may grow `nodes`
        const int r = buildRec(mid + 1, hi, depth + 1);
        nodes[nodeId].left  = l;   // index stays valid after reallocation
        nodes[nodeId].right = r;
        return nodeId;
    }

    float nearestSq(const Eigen::Vector3f& q) const {
        float best = std::numeric_limits<float>::max();
        nearestRec(root, q, best);
        return best;
    }

    void nearestRec(int nodeId, const Eigen::Vector3f& q, float& best) const {
        if (nodeId < 0) return;
        const Node& nd = nodes[nodeId];
        const Eigen::Vector3f& p = (*pts)[nd.idx];
        const float d2 = (p - q).squaredNorm();
        if (d2 < best) best = d2;

        const float diff = q[nd.axis] - p[nd.axis];
        const int near = (diff < 0) ? nd.left  : nd.right;
        const int far  = (diff < 0) ? nd.right : nd.left;
        nearestRec(near, q, best);
        if (diff * diff < best) nearestRec(far, q, best);   // prune
    }
};

// One-directional distances: for every query point, distance to nearest target.
std::vector<double> nearestDistances(const std::vector<Eigen::Vector3f>& query,
                                     const KDTree3& targetTree) {
    std::vector<double> d(query.size());
    for (size_t i = 0; i < query.size(); ++i)
        d[i] = std::sqrt((double)targetTree.nearestSq(query[i]));
    return d;
}

double mean(const std::vector<double>& v) {
    if (v.empty()) return 0;
    double s = 0; for (double x : v) s += x; return s / v.size();
}
double cappedMean(const std::vector<double>& v, double cap) {
    if (v.empty()) return 0;
    double s = 0; for (double x : v) s += std::min(x, cap); return s / v.size();
}
double median(std::vector<double> v) {              // by value: sorts a copy
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}
double fractionWithin(const std::vector<double>& v, double tau) {
    if (v.empty()) return 0;
    size_t c = 0; for (double x : v) if (x <= tau) ++c;
    return 100.0 * c / v.size();
}

}

std::vector<Eigen::Vector3f> loadPlyXYZ(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "[dtu_eval] cannot open " << path << "\n";
              return std::vector<Eigen::Vector3f>(); }

    auto typeSize = [](const std::string& t) -> int {
        if (t == "char"  || t == "uchar"  || t == "int8"  || t == "uint8")  return 1;
        if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
        if (t == "int"   || t == "uint"   || t == "int32" || t == "uint32"
                         || t == "float"  || t == "float32")                return 4;
        if (t == "double"|| t == "float64")                                 return 8;
        return 4;
    };

    int  nVerts = 0;
    int  fmt    = 0;            // 0 ascii, 1 binary_le, 2 binary_be
    bool inVertex = false;
    std::vector<std::string> propName, propType;
    std::vector<int>         propSize;

    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string tok; ss >> tok;
        if (tok == "format") {
            std::string ff; ss >> ff;
            fmt = (ff == "ascii") ? 0 : (ff == "binary_little_endian" ? 1 : 2);
        } else if (tok == "element") {
            std::string e; long n; ss >> e >> n;
            inVertex = (e == "vertex");
            if (inVertex) nVerts = (int)n;
        } else if (tok == "property" && inVertex) {
            std::string t1; ss >> t1;
            if (t1 == "list") {                 // unexpected for vertices; skip 3 tokens
                std::string a, b, nm; ss >> a >> b >> nm;
                propName.push_back(nm); propType.push_back("list"); propSize.push_back(0);
            } else {
                std::string nm; ss >> nm;
                propName.push_back(nm); propType.push_back(t1); propSize.push_back(typeSize(t1));
            }
        } else if (tok == "end_header") {
            break;
        }
    }

    int stride = 0, xoff = -1, yoff = -1, zoff = -1, xcol = -1, ycol = -1, zcol = -1;
    std::string xt = "float", yt = "float", zt = "float";
    for (size_t i = 0; i < propName.size(); ++i) {
        if (propName[i] == "x") { xoff = stride; xt = propType[i]; xcol = (int)i; }
        if (propName[i] == "y") { yoff = stride; yt = propType[i]; ycol = (int)i; }
        if (propName[i] == "z") { zoff = stride; zt = propType[i]; zcol = (int)i; }
        stride += propSize[i];
    }
    if (xoff < 0 || yoff < 0 || zoff < 0 || nVerts <= 0) {
        std::cerr << "[dtu_eval] no x/y/z in PLY header: " << path << "\n";
        return std::vector<Eigen::Vector3f>();
    }

    std::vector<Eigen::Vector3f> out;
    out.reserve(nVerts);

    if (fmt == 0) {   // ─ ASCII ─
        for (int i = 0; i < nVerts && std::getline(f, line); ++i) {
            std::istringstream ss(line);
            std::vector<double> vals; double v;
            while (ss >> v) vals.push_back(v);
            int need = std::max({ xcol, ycol, zcol });
            if ((int)vals.size() <= need) continue;
            out.emplace_back((float)vals[xcol], (float)vals[ycol], (float)vals[zcol]);
        }
    } else {          // ─ BINARY ─
        std::vector<char> buf((size_t)nVerts * stride);
        f.read(buf.data(), (std::streamsize)buf.size());
        const bool be = (fmt == 2);
        auto rd = [&](const char* base, int off, const std::string& t) -> float {
            char tmp[8];
            int sz = (t == "double" || t == "float64") ? 8 : 4;
            std::memcpy(tmp, base + off, sz);
            if (be) std::reverse(tmp, tmp + sz);            // big-endian → host LE
            if (sz == 8) { double d; std::memcpy(&d, tmp, 8); return (float)d; }
            float val; std::memcpy(&val, tmp, 4); return val;
        };
        for (int i = 0; i < nVerts; ++i) {
            const char* base = buf.data() + (size_t)i * stride;
            out.emplace_back(rd(base, xoff, xt), rd(base, yoff, yt), rd(base, zoff, zt));
        }
    }

    std::cout << "[dtu_eval] loaded " << out.size() << " pts from " << path
              << "  (" << (fmt == 0 ? "ascii" : "binary") << ")\n";
    return out;
}

static void bbox(const std::vector<Eigen::Vector3f>& p,
                 Eigen::Vector3f& lo, Eigen::Vector3f& hi) {
    lo.setConstant( std::numeric_limits<float>::max());
    hi.setConstant(-std::numeric_limits<float>::max());
    for (const auto& q : p) { lo = lo.cwiseMin(q); hi = hi.cwiseMax(q); }
}

DtuEvalResult evaluateCloudVsReference(const std::vector<Eigen::Vector3f>& ours,
                                       const std::vector<Eigen::Vector3f>& gt,
                                       double tau, double maxdist) {
    DtuEvalResult r;
    r.tau = tau; r.maxdist = maxdist;
    r.n_ours = ours.size(); r.n_gt = gt.size();
    if (ours.empty() || gt.empty()) {
        std::cerr << "[dtu_eval] empty cloud (ours=" << ours.size()
                  << ", gt=" << gt.size() << ")\n";
        return r;
    }

    bbox(ours, r.ours_min, r.ours_max);
    bbox(gt,   r.gt_min,   r.gt_max);
    r.frames_overlap = (r.ours_min.x() <= r.gt_max.x() && r.gt_min.x() <= r.ours_max.x() &&
                        r.ours_min.y() <= r.gt_max.y() && r.gt_min.y() <= r.ours_max.y() &&
                        r.ours_min.z() <= r.gt_max.z() && r.gt_min.z() <= r.ours_max.z());

    std::cout << "[dtu_eval] building KD-tree over GT (" << gt.size() << " pts)…\n";
    KDTree3 gtTree;   gtTree.build(gt);
    std::cout << "[dtu_eval] accuracy: querying " << ours.size() << " points…\n";
    std::vector<double> acc = nearestDistances(ours, gtTree);

    std::cout << "[dtu_eval] building KD-tree over reconstruction (" << ours.size() << " pts)…\n";
    KDTree3 ourTree;  ourTree.build(ours);
    std::cout << "[dtu_eval] completeness: querying " << gt.size() << " points…\n";
    std::vector<double> comp = nearestDistances(gt, ourTree);

    r.accuracy_mean       = mean(acc);
    r.accuracy_median     = median(acc);
    r.accuracy_capped     = cappedMean(acc, maxdist);
    r.precision_tau       = fractionWithin(acc, tau);

    r.completeness_mean   = mean(comp);
    r.completeness_median = median(comp);
    r.completeness_capped = cappedMean(comp, maxdist);
    r.recall_tau          = fractionWithin(comp, tau);

    r.chamfer = 0.5 * (r.accuracy_capped + r.completeness_capped);
    return r;
}

void printDtuEval(const DtuEvalResult& r, const std::string& label) {
    auto& o = std::cout;
    o << std::fixed << std::setprecision(3);
    if (!label.empty()) o << "=== DTU evaluation — " << label << " ===\n";
    o << "  points          ours=" << r.n_ours << "  gt=" << r.n_gt << "\n";
    o << "  ours  bbox      [" << r.ours_min.transpose() << "] .. [" << r.ours_max.transpose() << "]\n";
    o << "  gt    bbox      [" << r.gt_min.transpose()   << "] .. [" << r.gt_max.transpose()   << "]\n";
    if (!r.frames_overlap)
        o << "  ** WARNING: bounding boxes do NOT overlap — clouds may be in different\n"
             "     frames (expected same DTU world frame). Metrics will be meaningless.\n";
    o << "  --- accuracy (ours → GT, mm) ---\n";
    o << "    mean          " << r.accuracy_mean   << "\n";
    o << "    median        " << r.accuracy_median << "\n";
    o << "    mean@" << r.maxdist << "mm     " << r.accuracy_capped << "\n";
    o << "    precision@" << r.tau << "mm  " << std::setprecision(2) << r.precision_tau << " %\n"
      << std::setprecision(3);
    o << "  --- completeness (GT → ours, mm) ---\n";
    o << "    mean          " << r.completeness_mean   << "\n";
    o << "    median        " << r.completeness_median << "\n";
    o << "    mean@" << r.maxdist << "mm     " << r.completeness_capped << "\n";
    o << "    recall@" << r.tau << "mm     " << std::setprecision(2) << r.recall_tau << " %\n"
      << std::setprecision(3);
    o << "  --- chamfer (mm) ---\n";
    o << "    chamfer@" << r.maxdist << "mm   " << r.chamfer << "\n";
}

#ifndef PIPELINE_BUILD
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: dtu_eval <reconstruction.ply> <reference.ply> "
                     "[--tau <mm>] [--maxdist <mm>]\n";
        return 1;
    }
    double tau = 2.0, maxdist = 20.0;
    for (int i = 3; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--tau"     && i + 1 < argc) tau     = std::stod(argv[++i]);
        else if (a == "--maxdist" && i + 1 < argc) maxdist = std::stod(argv[++i]);
    }

    std::vector<Eigen::Vector3f> ours = loadPlyXYZ(argv[1]);
    std::vector<Eigen::Vector3f> gt   = loadPlyXYZ(argv[2]);
    if (ours.empty() || gt.empty()) { std::cerr << "Could not load clouds.\n"; return 1; }

    DtuEvalResult r = evaluateCloudVsReference(ours, gt, tau, maxdist);
    printDtuEval(r, std::string(argv[1]));
    return 0;
}
#endif
