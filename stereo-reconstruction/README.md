# Stereo Reconstruction

Classical stereo reconstruction pipeline in C++.  
Computes a dense 3D point cloud (and optionally a mesh) from a calibrated stereo image pair.

## Pipeline

```
im0.png + im1.png + calib.txt
        │
        ▼
1. Sparse Matching      →  F, E, R, t  (8-point + RANSAC)
        │
        ▼
2. Stereo Rectification →  left_rect.png, right_rect.png
        │
        ▼
3. Dense Matching       →  disparity.png  (SAD / SSD / NCC / SGM)
        │
        ▼
4. Disparity → Depth    →  pointcloud.ply
        │
        ▼
5. ICP Fusion (C2)      →  fused.ply      (multi-scene)
        │
        ▼
6. Mesh Reconstruction  →  mesh.ply       (requires Open3D)
```

Each step has an **OpenCV baseline** and a **manual implementation**.

---

## Dependencies

| Library | Purpose | Version |
|---------|---------|---------|
| OpenCV  | Image I/O, feature detection, rectification baseline, BM/SGBM | ≥ 4.5 |
| Eigen   | Linear algebra (SVD, ICP, pose math) | ≥ 3.4 |
| Ceres   | Non-linear ICP optimization | ≥ 2.0 |
| glog    | Logging (required by Ceres) | ≥ 0.6 |
| FLANN   | Nearest-neighbor search for ICP | 1.8.4 |
| Open3D  | Poisson mesh reconstruction (optional) | ≥ 0.17 |
| CMake   | Build system | ≥ 3.16 |

Eigen, Ceres, glog and FLANN are expected under a shared `Libs/` folder next to
this repository (same layout as Exercise 5):

```
TUM/SEMESTER-3/3D Scanning & Motion Capture/
├── Libs/
│   ├── Eigen/
│   ├── Ceres/
│   ├── glog-lib/
│   └── Flann-1.8.4/
└── Project/
    └── stereo-reconstruction/   ← this repo
```

If your `Libs/` is elsewhere, pass `-DLIBRARY_DIR=<path>` to CMake.

---

## Build

### Windows (Visual Studio)

```bat
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

Or open the folder in Visual Studio / VS Code and let CMake configure automatically.

### Linux / macOS

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Executables are placed in `build/` (Linux/macOS) or `build/Release/` (Windows).

---

## Dataset

Download a scene from [Middlebury Stereo 2014](https://vision.middlebury.edu/stereo/data/scenes2014/)
and place it under `data/`:

```
data/
└── Adirondack/
    ├── im0.png
    ├── im1.png
    ├── calib.txt
    └── disp0GT.pfm   ← ground-truth disparity (for evaluation)
```

---

## Usage

### Full pipeline

```bash
# OpenCV baseline (fast)
./build/pipeline data/Adirondack

# Manual SGM matching
./build/pipeline data/Adirondack --method sgm

# Manual rectification + SAD matching, window size 11
./build/pipeline data/Adirondack --manual-rect --method sad --window 11

# Override number of disparities
./build/pipeline data/Adirondack --disps 128
```

### Individual steps

```bash
# Sparse matching — outputs F, E, R, t to stdout
./build/sparse_matching data/Adirondack

# Rectification
./build/rectification data/Adirondack           # OpenCV baseline
./build/rectification data/Adirondack --manual  # Loop-Zhang (TODO)

# Dense matching
./build/matching data/Adirondack bm     # OpenCV BlockMatching
./build/matching data/Adirondack sgbm   # OpenCV SGBM
./build/matching data/Adirondack sad    # Manual SAD
./build/matching data/Adirondack ssd    # Manual SSD
./build/matching data/Adirondack ncc    # Manual NCC
./build/matching data/Adirondack sgm    # Manual SGM (TODO)

# Depth + point cloud
./build/depth data/Adirondack

# Evaluation against ground truth
./build/evaluation data/Adirondack results/disparity_raw.png

# ICP fusion (C2 challenge)
./build/icp results/scene1/pointcloud.ply results/scene2/pointcloud.ply
```

All outputs go to `results/`.

---

## Available Matching Methods

| Flag   | Type    | Status |
|--------|---------|--------|
| `bm`   | OpenCV BlockMatching | done |
| `sgbm` | OpenCV Semi-Global BM | done |
| `sad`  | Manual Sum of Absolute Differences | done |
| `ssd`  | Manual Sum of Squared Differences | done |
| `ncc`  | Manual Normalized Cross-Correlation | done |
| `sgm`  | Manual Semi-Global Matching | **TODO (C1)** |

---

## Project Structure

```
stereo-reconstruction/
├── include/
│   ├── Eigen.h              # Eigen setup + type aliases (Vector4uc, …)
│   ├── utils.h              # CalibData, loadCalib, save/loadDisparity
│   ├── PointCloud.h         # Eigen-based PointCloud struct (points, normals, colors)
│   ├── NearestNeighbor.h    # FLANN kd-tree nearest-neighbor search
│   ├── ProcrustesAligner.h  # Closed-form rotation via SVD
│   ├── ICPOptimizer.h       # CeresICPOptimizer + LinearICPOptimizer
│   ├── sparse_matching.h    # Member 1 — feature matching, 8-point, RANSAC, pose
│   ├── rectification.h      # Member 2 — Loop-Zhang rectification
│   ├── matching.h           # Member 3 — SAD/SSD/NCC/SGM
│   ├── depth.h              # Member 2 — disparity → depth → point cloud
│   ├── evaluation.h         # Member 4 — PFM loader, bad1/bad2/RMSE
│   └── icp.h                # Member 4 (C2) — multi-cloud ICP fusion
├── src/
│   ├── utils.cpp
│   ├── sparse_matching.cpp  # Member 1 — SIFT + 8-point + RANSAC + E→R,t  (TODO)
│   ├── rectification.cpp    # Member 2 — Loop-Zhang homographies            (TODO)
│   ├── matching.cpp         # Member 3 — SAD/SSD/NCC done; SGM              (TODO)
│   ├── depth.cpp            # Member 2 — disparity→depth→.ply + normals     (done)
│   ├── mesh.cpp             # Member 4 — depth triangulation + Poisson       (TODO)
│   ├── evaluation.cpp       # Member 4 — PFM loader + metrics                (TODO)
│   ├── icp.cpp              # Member 4 (C2) — ICP fusion via ICPOptimizer    (TODO)
│   └── pipeline.cpp         # Full pipeline runner
├── data/                    # Dataset scenes (git-ignored)
├── results/                 # Output files  (git-ignored)
└── CMakeLists.txt
```

---

## References

- Loop & Zhang, *Computing Rectifying Homographies for Stereo Vision*, CVPR 1999
- Hirschmüller, *Stereo Processing by Semiglobal Matching and Mutual Information*, PAMI 2008
- [Middlebury Stereo Evaluation](https://vision.middlebury.edu/stereo/)
