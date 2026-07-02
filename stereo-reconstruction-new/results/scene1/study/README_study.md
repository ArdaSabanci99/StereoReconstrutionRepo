# Stage 3 — Dense Matching Study (Contribution C1)

**Owner:** Christian Gerling · **Scene:** DTU scan1, view pair 001–002, light 0 (max)
**Settings:** scale 0.25 (→ ~400×300), ndisp 208, window 5×5, sub-pixel + L-R check + median/hole-fill on.
**Evaluation:** DTU has no ground-truth disparity, so every method is scored in **3D**: the resulting
point cloud is compared against the structured-light reference `stl001_total.ply` (2.88 M pts) with a
KD-tree. We report accuracy (ours→GT), completeness (GT→ours), Chamfer, and precision/recall @ 2 mm.
All custom methods share the *same* rectification, triangulation and post-processing — only the cost
function changes — so the table isolates the matching stage. `bm`/`sgbm` are the OpenCV baseline each
custom method is measured against.

## 1. Cost-function / method comparison

| Method | Impl | Match ms | Points | Acc.med (mm) | Acc.@20 (mm) | Comp.@20 (mm) | Chamfer@20 (mm) | Prec@2mm | Rec@2mm |
|--------|------|---------:|-------:|-------------:|-------------:|--------------:|----------------:|---------:|--------:|
| SAD    | custom | 249 | 117020 | 29.53 | 14.03 | 7.12 | 10.57 | 13.9% | 22.2% |
| SSD    | custom | 258 | 117264 | 29.79 | 13.99 | 6.90 | 10.45 | 15.3% | 22.5% |
| NCC    | custom | 409 | 117506 | 20.85 | 12.35 | 5.85 |  9.10 | 24.4% | 33.5% |
| Census | custom | 610 | 115263 | 13.74 | 11.37 | 5.80 |  8.58 | 28.3% | 34.3% |
| **SGM (ours)** | **custom** | **670** | 114729 | **2.24** | **7.26** | **4.37** | **5.81** | **48.6%** | **62.3%** |
| BM     | OpenCV |   6 | 111220 | 25.50 | 12.13 | 5.07 |  8.60 | 30.1% | 39.8% |
| SGBM   | OpenCV |  20 | 116766 |  8.71 | 10.45 | 7.18 |  8.81 | 30.5% | 28.1% |

(Match ms = one left→right pass; the L-R check runs a second pass. OpenCV timings are for the
library's optimised, multi-threaded routines and are not directly comparable to our single-threaded
from-scratch loops — the point of the runtime column is the *cost of the algorithm*, not tuned speed.)

Per-method disparity figures: `disp_<method>_1_2.png` in this folder.

## 2. Findings

- **Cost-function ordering is textbook.** SAD ≈ SSD are worst (raw-intensity difference, sensitive to
  the DTU LED illumination and gain). NCC improves by normalising for gain/bias. **Census** is clearly
  better again (radiometric robustness via local ordering) — Chamfer drops 10.5 → 8.6 mm and precision
  nearly doubles vs SAD. This is exactly the SAD/SSD → NCC → Census progression the proposal set out to
  demonstrate.
- **SGM is the decisive win.** Adding 8-direction path aggregation on top of the Census cost collapses
  accuracy-median from 13.7 mm (Census WTA) to **2.24 mm** and lifts recall@2mm from 34% to **62%**.
  Global smoothness resolves the textureless/ambiguous regions that break per-pixel winner-takes-all.
- **Our SGM beats the OpenCV baseline.** Custom SGM Chamfer@20 **5.81 mm** vs OpenCV BM 8.60 and OpenCV
  SGBM 8.81 — and higher precision *and* recall than both. This is the C1 headline: the from-scratch
  implementation is not just correct, it is competitive with / better than the library baseline on this
  dataset (our post-processing — Census-SGM + L-R check + honest hole-fill — is tuned for DTU).
- **Runtime is the honest trade-off.** OpenCV BM/SGBM run in 6–20 ms; our loops in 250–670 ms
  single-threaded. Quality parity/superiority at ~30× the cost is the expected outcome of a
  from-scratch classical implementation and is reported as such.

## 3. Window-size ablation (Census cost, same pair)

| Cost | Window | Match ms | Chamfer@20 (mm) | Prec@2mm | Rec@2mm |
|------|-------:|---------:|----------------:|---------:|--------:|
| Census | 3×3 |  235 | 9.28 | 22.9% | 31.5% |
| Census | 5×5 |  616 | 8.58 | 28.3% | 34.4% |
| Census | 7×7 | 1235 | 8.44 | 30.9% | 34.1% |
| Census | 9×9 | 2077 | 8.40 | 33.0% | 34.0% |

Larger windows raise precision (more support ⇒ less ambiguity) with **diminishing quality returns**
(Chamfer flattens 8.58 → 8.40 from 5×5 to 9×9) but **quadratically rising cost** (235 → 2077 ms). Recall
saturates ~34%. **5×5 is the sweet spot** used in the method table above; going to 9×9 nearly quadruples
runtime for ~0.2 mm. This is the classic small-window (detail, noisy) vs large-window (smooth, blurs
depth edges) trade-off.

## 4. Baseline ablation (custom SGM, varying stereo baseline)

DTU is a single camera on a robot arm, so a "stereo pair" is two arm positions. Anchoring the
left view at 001 and widening the right view increases the baseline. Method (SGM), scale (0.20 →
320 px wide) and ndisp (252) are held **fixed** — only the baseline varies. Same scan1 / stl001 GT.

| Pair | Baseline (mm) | ×(1-2) | Points | Chamfer@20 (mm) | Prec@2mm | Rec@2mm |
|------|--------------:|-------:|-------:|----------------:|---------:|--------:|
| 1-2 | 128.9 | 1.0× | 76 207 |  **5.96** | 46.7% | 59.1% |
| 1-3 | 250.5 | 1.9× | 16 102 | 13.55 | 32.2% |  7.8% |
| 1-4 | 356.3 | 2.8× |  6 280 | 20.00 (cap) |  0.0% |  0.0% |

Figures: `disp_baseline_1_{2,3,4}.png`.

**Reading it — this is the classic baseline trade-off, from the wide side:**
- **1-2 (129 mm) is the sweet spot** — dense, accurate (Chamfer 5.96, recall 59%).
- **1-3 (250 mm, ~2×) already degrades sharply** even though disparity (~248 px) still fits the search
  range: valid points collapse 76 k → 16 k and recall 59% → 8%. This is *not* a search-range artifact —
  it is the real physics of a wide baseline: larger occluded regions (rejected by the L-R check),
  stronger perspective foreshortening (windows no longer look alike), and heavier rectification warp.
- **1-4 (356 mm, ~2.8×) breaks down entirely** (Chamfer pinned at the 20 mm cap, 0% precision/recall):
  the near-surface disparity (~343 px) now exceeds both ndisp and the image width, so those pixels are
  literally unmatchable. Wider baseline ⇒ proportionally larger disparity ⇒ needs larger ndisp (more
  compute) *and* eventually can't fit in the image at all.

*Not probed:* the small-baseline side (poor depth precision from tiny disparity / ill-conditioned
triangulation) — 129 mm is already the smallest single-step baseline DTU offers, and it sits near the
optimum. So on DTU the usable-baseline window is narrow, and one-step neighbour pairs are the right
choice — which is exactly why the team fixed on them.

## 5. Failure analysis (qualitative — see disparity figures)

- **Textureless regions** (background wall, matte cup body): SAD/SSD/NCC produce speckle/streak noise;
  Census reduces it; SGM fills them smoothly via the P2 smoothness term.
- **Repetitive texture** (the painted lettering on the base): WTA methods show disparity "jumps" to the
  wrong period; SGM's path costs suppress most of these.
- **Occlusions / depth discontinuities** (cup rim against background): flagged and removed by the L-R
  consistency check, then conservatively filled from the *background* side so the foreground does not
  bleed outward.

## Reproduce

```bash
./run_matching_study.sh 1 1 2 001 0.25 208 5      # method sweep     → study_pair_1_2.md
./run_baseline_ablation.sh                        # baseline sweep   → ablation_baseline_sgm.md
# window ablation: pipeline --method census --window {3,5,7,9}  → ablation_window_census.md
```
