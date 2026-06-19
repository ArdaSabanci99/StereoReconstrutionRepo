# Stereo Reconstruction — Technical Roadmap

> Group 13 · TU Munich · 3D Scanning & Motion Capture  
> Utku Gürsoy · Arda Sabanci · Kateřina Skotnicová · Christian Gerling

---

## Overview

We build the **full classical stereo pipeline from scratch** in C++, using OpenCV
only for image I/O and SIFT feature detection. Every geometric algorithm
(8-point, RANSAC, rectification, matching, triangulation, evaluation, ICP) is
our own implementation. This makes it easy to compare each stage against the
OpenCV baseline and understand where errors come from.

**Dataset:** DTU MVS — unrectified, calibrated, multi-view (49 positions per scene).  
**Primary dataset for evaluation:** Middlebury 2014 (ground-truth PFM disparity).  
**Build:** C++17, CMake, OpenCV ≥ 4.5, Eigen ≥ 3.4, Ceres (for ICP).

---

## Pipeline

```
DTU / Middlebury images + calibration
            │
            ▼
┌─────────────────────────────────────┐
│  Stage 1 · Sparse Matching          │  → F, E, R, t
│  SIFT · 8-point · RANSAC            │
└──────────────┬──────────────────────┘
               │
            ▼
┌─────────────────────────────────────┐
│  Stage 2 · Stereo Rectification     │  → rectified image pair
│  Loop-Zhang homographies (1999)     │
└──────────────┬──────────────────────┘
               │
            ▼
┌─────────────────────────────────────┐
│  Stage 3 · Dense Matching           │  → disparity map
│  SAD / SSD / NCC / Census / SGM     │
│  + sub-pixel · L-R check · median   │
└──────────────┬──────────────────────┘
               │
            ▼
┌─────────────────────────────────────┐
│  Stage 4 · Depth & Point Cloud      │  → coloured .ply / .off
│  Z = fB/d · DLT triangulation       │
│  normal estimation                  │
└──────────────┬──────────────────────┘
               │
            ▼
┌─────────────────────────────────────┐
│  Stage 5 · Evaluation               │  → bad1/bad2/RMSE/coverage
│  PFM loader · Middlebury metrics    │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐  ┌──────────────────┐
│  Stage 6     │  │  Stage 7         │
│  ICP Fusion  │  │  Poisson Mesh    │
│  multi-pair  │  │  reconstruction  │
└──────────────┘  └──────────────────┘
```

---

## Stage 1 — Sparse Matching

**Goal:** Recover relative rotation R and translation t from two uncalibrated images.

### 1.1 Feature Detection — SIFT

We use OpenCV's SIFT to detect keypoints and compute 128-dim descriptors.
SIFT is scale- and rotation-invariant, which is important for the wide baselines
in DTU.

```
detector = SIFT::create(2000 keypoints)
detect + compute descriptors for left and right images
```

### 1.2 Feature Matching — Lowe Ratio Test

We match descriptors with BFMatcher (L2 norm) using k=2 nearest neighbours.
A match is kept only if:

```
d1 / d2 < 0.75     (Lowe 2004)
```

This rejects ambiguous matches where the second-best is almost as good.

### 1.3 Normalised 8-Point Algorithm (Hartley 1997)

**Why normalise?** The 8-point algorithm is ill-conditioned on pixel coordinates
because image coordinates span [0, W] × [0, H] while the algorithm needs
entries of A to have similar magnitude. Normalisation brings all points to a
unit circle around the origin.

**Normalisation transform T:**
```
centroid = mean(pts)
scale    = sqrt(2) / mean_distance_from_centroid
T = [ scale    0     -scale * cx ]
    [   0    scale   -scale * cy ]
    [   0      0          1      ]
```

**Building the coefficient matrix A (N×9):**

For each correspondence (x₁,y₁) ↔ (x₂,y₂) in normalised coords:
```
row = [ x2·x1,  x2·y1,  x2,  y2·x1,  y2·y1,  y2,  x1,  y1,  1 ]
```

**Solving Af = 0 via SVD:**
```
SVD(A) → f = last column of V  (smallest singular value)
F_hat = reshape(f, 3×3)
```

**Enforcing rank-2 constraint:**
```
SVD(F_hat) → U, S, Vt
S[2] = 0
F = U · diag(S) · Vt
```

**Denormalise:**
```
F = T2^T · F_tilde · T1
```

### 1.4 RANSAC

We wrap the 8-point algorithm in RANSAC to reject outlier correspondences.

```
For N iterations:
  Sample 8 random correspondences
  Estimate F with normalised 8-point
  For each point pair:
    error = |x2^T · F · x1|   (Sampson distance)
    if error < threshold (1.5 px): mark as inlier
  Keep F with most inliers

Re-estimate F on all inliers of best model
```

**Sampson distance** is used instead of symmetric epipolar distance because it
is cheaper and a first-order approximation of the true geometric error:
```
d = (x2^T F x1)^2 / ( (Fx1)_1^2 + (Fx1)_2^2 + (F^T x2)_1^2 + (F^T x2)_2^2 )
```

### 1.5 Essential Matrix → R, t

Given K (known from calibration):
```
E = K2^T · F · K1
```

SVD of E gives four candidate (R, t) solutions. We pick the one where the
triangulated point has positive depth in both cameras (cheirality test):
```
SVD(E) → U, S, Vt
W = [ 0 -1  0; 1  0  0; 0  0  1 ]
R1 = U W  Vt,    R2 = U W^T Vt
t1 =  U[:,2],    t2 = -U[:,2]
→ 4 combinations, keep the valid one
```

---

## Stage 2 — Stereo Rectification (Loop & Zhang 1999)

**Goal:** Find two homographies H1, H2 such that the rectified images have
epipolar lines horizontal and aligned — correspondence search becomes 1D.

### 2.1 Compute Epipoles

```
SVD(F^T) → e2 = last column of V   (left epipole,  F e1 = 0)
SVD(F)   → e1 = last column of V   (right epipole, F^T e2 = 0)
```

### 2.2 Build H2 (map right epipole to infinity on x-axis)

1. **Translate** image so e2 is at centre: T = translate(-cx, -cy)
2. **Rotate** e2 onto the x-axis: R = rotation to align e2 to [1, 0, 0]
3. **Send to infinity**: G = [ 1 0 0; 0 1 0; -1/f 0 1 ] where f = distance to e2
4. H2 = G · R · T

### 2.3 Build H1 (minimise vertical disparity)

Define M = H2 · F (the transfer matrix from right to left).
We want H1 ≈ Ha · M where Ha is affine:
```
Ha = [ a b c; 0 1 0; 0 0 1 ]
```

Minimize Σ ‖Ha·H2·m_left - H1·m_right‖² — leads to a 3×3 linear system
solved with Eigen LLT.

### 2.4 Warp Images

Apply H1 and H2 as perspective warps using `cv::warpPerspective`.
Also compute the disparity-to-depth matrix Q from the rectified parameters.

---

## Stage 3 — Dense Matching

**Goal:** For each pixel in the rectified left image, find the corresponding
pixel in the right image by scanning along the same row (epipolar line).

### 3.1 Cost Functions

| Method | Formula | Property |
|--------|---------|----------|
| **SAD** | Σ\|Il(y+dy, x+dx) − Ir(y+dy, x−d+dx)\| | Fast, sensitive to illumination |
| **SSD** | Σ(Il − Ir)² | Penalises large errors more |
| **NCC** | Σ(Il·Ir) / (‖Il‖·‖Ir‖) | Invariant to gain/bias |
| **Census** | Hamming(bitstring_l, bitstring_r) | Radiometrically robust |

Census transform: encode the sign pattern of a 5×5 neighbourhood relative
to the centre pixel as a 24-bit integer. Cost = Hamming distance.

### 3.2 Semi-Global Matching (Hirschmüller 2008) — C1 Challenge

SGM smooths the disparity map by aggregating costs along multiple 1D paths,
enforcing piecewise-smooth disparity changes.

**Per-pixel cost volume** C(y, x, d): Census cost for each pixel and disparity.

**Path-cost recursion** (for each direction r):
```
Lr(p, d) = C(p,d) + min(
    Lr(p−r, d),
    Lr(p−r, d−1) + P1,
    Lr(p−r, d+1) + P1,
    min_k Lr(p−r, k) + P2
) − min_k Lr(p−r, k)
```
- P1 = 10  (small disparity jump penalty)
- P2 = 120 (large jump penalty — controls smoothness)
- 8 directions: ←, →, ↑, ↓, ↖, ↗, ↙, ↘

**Aggregate:** S(p, d) = Σ_r Lr(p, d)

**Winner-Takes-All:** disp(p) = argmin_d S(p, d)

### 3.3 Post-processing

**Sub-pixel refinement** — parabola fit on the cost curve:
```
d_sub = d − 0.5 · (C[d+1] − C[d−1]) / (C[d+1] − 2·C[d] + C[d−1])
```

**Left-Right consistency check:**
```
Compute disp_LR (left→right) and disp_RL (right→left)
Flag pixel as invalid if |disp_LR[x] − disp_RL[x − disp_LR[x]]| > 1
```

**Median filter + hole filling:** 5×5 median on valid pixels, fill gaps by
interpolating horizontally from valid neighbours.

---

## Stage 4 — Depth & Point Cloud

### 4.1 Disparity to Depth

For a rectified stereo pair with baseline B and focal length f:
```
Z = f · B / d
X = (x − cx) · Z / f
Y = (y − cy) · Z / f
```

For non-rectified pairs (general case): **DLT triangulation**.
Given projection matrices P1, P2 and corresponding points x1, x2:
```
Build A (4×4):
  row 0: x1 * P1[2] − P1[0]
  row 1: y1 * P1[2] − P1[1]
  row 2: x2 * P2[2] − P2[0]
  row 3: y2 * P2[2] − P2[1]
SVD(A) → X = last column of V (homogeneous)
```

### 4.2 Normal Estimation

Estimate surface normals using PCA on a local neighbourhood (k-NN or radius
search). The normal is the eigenvector corresponding to the smallest eigenvalue
of the local covariance matrix. Orient normals towards the camera.

---

## Stage 5 — Evaluation (Middlebury)

### 5.1 PFM Format

PFM stores 32-bit float images, bottom-row first, with optional byte-swap.
```
header: "Pf\n<width> <height>\n<scale>\n"   (scale < 0 = little-endian)
data: w * h * 4 bytes, rows from bottom to top
```

### 5.2 Metrics

Over all pixels where ground-truth disparity is valid (gt > 0):
```
bad1     = % pixels with |est − gt| > 1.0
bad2     = % pixels with |est − gt| > 2.0
bad4     = % pixels with |est − gt| > 4.0
RMSE     = sqrt( Σ (est − gt)² / N_valid )
avg_err  = Σ |est − gt| / N_valid
coverage = 100 * N_valid / N_gt   (% of GT pixels with valid estimate)
```

Pixels where our estimate is out of range [vmin, vmax] count as invalid (not
penalised in metrics but reduce coverage).

---

## Stage 6 — ICP Multi-Pair Fusion (C2)

We generate multiple point clouds from different stereo pairs of the same DTU
scene, then align them with ICP.

**Sequential fusion:**
```
merged = cloud_0
for i = 1..N:
    align cloud_i to merged using ICP
    transform cloud_i by result pose
    append transformed cloud_i to merged
    voxel-downsample merged (cell size = 2 mm)
```

**ICP (Point-to-Point):**
```
Repeat until convergence:
  1. For each point in source: find nearest in target (FLANN kd-tree)
  2. Reject outlier pairs (distance > threshold)
  3. Solve for optimal R, t via closed-form SVD (Procrustes):
     SVD(B = Σ (s_i − ŝ)(t_i − t̄)^T) → R = V U^T,  t = t̄ − R·ŝ
  4. Apply transform, update pose
```

**Point-to-Plane variant** (optional, faster convergence):
Minimise Σ ((R·s_i + t − t_i) · n_i)² — linear in R, t near identity.

---

## Stage 7 — Poisson Mesh Reconstruction

Given coloured point cloud with normals:
1. Screen Poisson reconstruction via Open3D:
   ```python
   mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
   ```
2. Crop to bounding box to remove artefacts
3. Save as `mesh.ply`

Alternatively run from command line with the Open3D CLI.

---

## Milestones

| # | Deadline | Deliverable |
|---|----------|-------------|
| R1 | Jun 15 ✅ | OpenCV end-to-end baseline, first point cloud |
| R2 | Jun 22 | 8-point + RANSAC, evaluation (PFM + metrics), project setup |
| R3 | Jun 29 | Loop-Zhang rectification, cost-function study (tables + figures) |
| R4 | Jul 6  | SGM, ICP fusion, Poisson mesh |
| R5 | Jul 13 | Matching study complete, failure analysis, stretch goals |
| final | Jul 20 | Report + video |

---

## Member Assignments

| Stage | Module | Owner |
|-------|--------|-------|
| 1 | Sparse matching, 8-point, RANSAC | Katerina |
| 2 | Loop-Zhang rectification, depth/triangulation | Utku |
| 3 | SAD/SSD/NCC/Census/SGM, post-processing | Christian |
| 4–7 | Evaluation, ICP fusion, Poisson mesh, figures | Arda |

---

## References

1. Aanæs et al., *Large-Scale Data for Multiple-View Stereopsis*, IJCV 2016
2. Besl & McKay, *A Method for Registration of 3-D Shapes*, IEEE TPAMI 1992
3. Hartley, *In Defense of the Eight-Point Algorithm*, IEEE TPAMI 1997
4. Hirschmüller, *Stereo Processing by Semiglobal Matching*, IEEE PAMI 2008
5. Hirschmüller & Scharstein, *Evaluation of Stereo Matching Costs on Images
   with Radiometric Differences*, IEEE TPAMI 2009
6. Kazhdan & Hoppe, *Screened Poisson Surface Reconstruction*, ACM TOG 2013
7. Lipson et al., *RAFT-Stereo*, 3DV 2021
8. Loop & Zhang, *Computing Rectifying Homographies for Stereo Vision*, CVPR 1999
9. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004
10. Scharstein & Szeliski, *A Taxonomy and Evaluation of Dense Two-Frame
    Stereo Correspondence Algorithms*, IJCV 2002
11. Scharstein et al., *High-Resolution Stereo Datasets with Subpixel-Accurate
    Ground Truth*, GCPR 2014
12. Zabih & Woodfill, *Non-Parametric Local Transforms for Computing Visual
    Correspondence*, ECCV 1994
