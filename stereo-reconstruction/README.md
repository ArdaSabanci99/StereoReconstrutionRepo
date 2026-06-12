# Stereo Reconstruction

Classical stereo reconstruction pipeline in C++.  
Computes a dense 3D point cloud (and optionally a mesh) from a calibrated stereo image pair.

## Pipeline

```
DTU scan + calibration
        │
        ▼
1. Sparse Matching      →  F, E, R, t  (8-point + RANSAC)
        │
        ▼
2. Stereo Rectification →  results/scene<id>/rectification/view_<L>.png
                                                            view_<R>.png
        │
        ▼
3. Dense Matching       →  results/scene<id>/matching/view_<L>_<R>_disparity.png
                                                       view_<L>_<R>_disparity_raw.png
        │
        ▼
4. Disparity → Depth    →  results/scene<id>/pointcloud/views_<L>_<R>.ply
                                                         views_<L>_<R>.off
        │
        ▼
5. ICP Fusion (C2)      →  results/icp_fused/fused.ply   (multi-scene)
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

This project uses the **DTU MVS** dataset. Download and place it so the directory
tree matches the expected layout:

```
<data_path>/
├── Rectified/
│   └── scan<scene_id>/
│       └── rect_<view_id>_<light_id>_r5000.png
└── Calibration/
    └── cal18/
        └── pos_<view_id>.txt
```

- `scene_id` — integer scene number (e.g. `1`, `24`, `110`)
- `view_id` — zero-padded three-digit integer (e.g. `001`, `025`); passed as a
  plain integer on the command line and padded automatically
- `light_id` — integer lighting condition index (default `0`)

Calibration is extracted per-view from the 3×4 projection matrices in
`Calibration/cal18/pos_<view_id>.txt` via `DTUDataLoader::loadCalib()`.
No `calib.txt` file is needed.

---

## Usage

### Full pipeline

`pipeline` takes `<data_path> <scene_id> <left_view_id> <right_view_id>` as
the first four positional arguments. View IDs are plain integers and are
zero-padded automatically.

```bash
# OpenCV baseline (fast)
./build/pipeline /path/to/dtu 1 1 2

# Manual SGM matching
./build/pipeline /path/to/dtu 1 1 2 --method sgm

# Manual Loop-Zhang rectification + SAD matching, window size 11
./build/pipeline /path/to/dtu 1 1 2 --manual-rect --method sad --window 11

# Override number of disparities and lighting condition
./build/pipeline /path/to/dtu 1 1 2 --ndisp 128 --light 3
```

**Options:**

| Flag | Argument | Default | Description |
|------|----------|---------|-------------|
| `--method` | `bm\|sgbm\|sad\|ssd\|ncc\|census\|sgm` | `sgbm` | Matching method |
| `--manual-rect` | — | off | Use Loop-Zhang manual rectification (instead of OpenCV baseline) |
| `--window` | `<size>` | `5` | Matching window size |
| `--ndisp` | `<n>` | `64` | Number of disparity levels |
| `--light` | `<id>` | `0` | DTU lighting condition index |

### Individual steps

Most executables take `<data_path> <scene_id> <left_view_id> <right_view_id>` as
the first four positional arguments. **`matching` is an exception** — it reads
rectified images from `results/` directly and only needs the scene and view IDs.

```bash
# Sparse matching — outputs F, E, R, t to stdout
./build/sparse_matching /path/to/dtu 1 1 2

# Rectification
./build/rectification /path/to/dtu 1 1 2           # OpenCV baseline
./build/rectification /path/to/dtu 1 1 2 --manual  # Loop-Zhang

# Dense matching (reads rectified images produced by the rectification step)
# Usage: ./build/matching <scene_id> <left_view_id> <right_view_id> [method] [options]
./build/matching 1 1 2 bm     # OpenCV BlockMatching
./build/matching 1 1 2 sgbm   # OpenCV SGBM
./build/matching 1 1 2 sad    # Manual SAD
./build/matching 1 1 2 ssd    # Manual SSD
./build/matching 1 1 2 ncc    # Manual NCC
./build/matching 1 1 2 sgm    # Manual SGM (TODO)

# Depth + point cloud (reads disparity produced by the matching step)
./build/depth /path/to/dtu 1 1 2

# ICP fusion (C2 challenge)
./build/icp results/scene1/pointcloud/views_001_002.ply \
            results/scene1/pointcloud/views_001_003.ply
```

### Output structure

```
results/
└── scene<scene_id>/
    ├── rectification/
    │   ├── view_<L>.png
    │   └── view_<R>.png
    ├── matching/
    │   ├── view_<L>_<R>_disparity.png      ← false-colour visualisation
    │   └── view_<L>_<R>_disparity_raw.png  ← raw disparity (16-bit PNG)
    └── pointcloud/
        ├── views_<L>_<R>.ply
        └── views_<L>_<R>.off
```

ICP fusion writes to `results/icp_fused/`.

---

## Available Matching Methods

| Flag     | Type    | Status |
|----------|---------|--------|
| `bm`     | OpenCV BlockMatching | done |
| `sgbm`   | OpenCV Semi-Global BM | done |
| `sad`    | Manual Sum of Absolute Differences | done |
| `ssd`    | Manual Sum of Squared Differences | done |
| `ncc`    | Manual Normalized Cross-Correlation | done |
| `census` | Manual Census transform + Hamming (Zabih & Woodfill 1994) | done |
| `sgm`    | Manual Semi-Global Matching (Hirschmüller 2008) | **TODO (C1)** |

---

## Project Structure

```
stereo-reconstruction/
├── include/
│   ├── DataLoader.h         # DTUDataLoader — image + calibration loading from DTU layout
│   ├── Eigen.h              # Eigen setup + type aliases (Vector4uc, …)
│   ├── utils.h              # CalibData, save/loadDisparity, padViewId
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
│   ├── DataLoader.cpp       # DTUDataLoader implementation
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
- Aanæs et al., *Large-Scale Data for Multiple-View Stereopsis*, IJCV 2016
- [DTU MVS Dataset](https://roboimagedata.compute.dtu.dk/?page_id=36)
