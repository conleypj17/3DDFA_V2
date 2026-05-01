# 3DDFA_V2 — Facial Motion Analysis Pipeline

This is a fork of [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) with a custom facial motion analysis pipeline added on top. The original 3DDFA_V2 code is unchanged — all additions are new files that sit alongside the existing repo structure.

The pipeline extracts per-region facial motion (position, velocity, acceleration), global head pose (roll, pitch, yaw), and 468 canonical 3D/2D landmarks from video frames stored as PNG sequences. It was designed for footage captured at distances ranging from **1 metre to 5 metres** from the camera.

---

## What was added

| File | Description |
|---|---|
| `analyse_video4.py` | Main entry point. Run this on a folder of PNG frames. |
| `utils/region_motion.py` | Computes per-region centroids, nose subtraction, velocity, and acceleration from 68-point sparse landmarks. |
| `utils/fps_points.py` | Runs Farthest Point Sampling on the BFM dense mesh to select 468 canonical landmark points. Run once to generate the index. |
| `process_frames.py` | Utility for preprocessing video into PNG frame sequences before running the main pipeline. |
| `bfm_468_indices.json` | Pre-generated 468-point index file. **You do not need to regenerate this** unless you want a different point distribution. |
| `bfm_468_pointmap.json` | Shareable reference file mapping each of the 468 point IDs to their 3D position on the BFM mean face. Share this with collaborators using other face tracking methods so they can find equivalent points in their own mesh. |

---

## How it works

### Overview

3DDFA_V2 fits a parametric 3D face model (the Basel Face Model, BFM) to each video frame. Every point on that model has a consistent 3D position regardless of which person is being tracked, which makes it possible to compare motion across subjects and distances.

The pipeline runs in two stages per frame:

1. **Face detection** — finds the face bounding box in the image (FaceBoxes or SCRFD)
2. **3DMM regression** — fits the BFM model to the detected face crop and recovers 3D landmark positions

Velocity and acceleration are then computed from the landmark positions across frames using finite differences.

### Nose subtraction

All 3D landmark positions are expressed **relative to the nose tip**, not in absolute model space. This removes global head translation from the signal — so if a person turns their head, the motion of the lip region reflects only the lips moving, not the whole head moving. This is the same approach used in MediaPipe-based pipelines.

### The 468-point set

The BFM dense mesh has 38,365 vertices. We select 468 of them using **Farthest Point Sampling (FPS)**, biased toward the five regions of interest (right eye, left eye, lips, right cheek, left cheek) to ensure good coverage where it matters. The selected indices are saved in `bfm_468_indices.json` and are fixed — the same 468 points are used on every run, making outputs comparable across videos.

---

## Setup

### 1. Follow the original 3DDFA_V2 setup

Clone this fork and follow the original installation instructions:

```bash
git clone https://github.com/yourusername/3DDFA_V2.git
cd 3DDFA_V2
```

Install dependencies and build the Cython extensions:

```bash
pip install -r requirements.txt
sh build.sh
```

Download the pretrained model weights as described in the original README.

> **NumPy version note:** The FaceBoxes Cython extension requires NumPy < 2.0. If you have a newer NumPy installed, downgrade it before building:
> ```bash
> pip install "numpy<2"
> sh build.sh
> ```

### 2. Install additional dependencies for SCRFD (optional)

Only needed if you want to use the SCRFD detector (recommended for footage at 4–5 metres):

```bash
pip install insightface onnxruntime
```

Model weights for SCRFD (~10MB) are downloaded automatically on first use and cached in `~/.insightface/models/`.

---

## Running the pipeline

### Input format

The pipeline expects a **directory of PNG images**, one per frame, named so that alphabetical sorting matches frame order (e.g. `frame_0000.png`, `frame_0001.png`, ...). Use `process_frames.py` to convert a video file into this format if needed.

### Basic usage

```bash
python analyse_video4.py -f path/to/png/directory --onnx
```

### With SCRFD detector (recommended for 4–5 metre footage)

```bash
python analyse_video4.py -f path/to/png/directory --onnx --detector scrfd
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `-f` / `--input` | required | Path to directory containing PNG frames |
| `--onnx` | off | Use ONNX runtime for the 3DMM regressor. Significantly faster on CPU — recommended. |
| `--detector` | `facebox` | Face detector. `facebox` is the default. Use `scrfd` for small/distant faces (4–5m). |
| `--fps` | `30.0` | Frame rate of the source video. Used for velocity/acceleration units. |
| `--config` | `configs/mb1_120x120.yml` | 3DDFA model config file. |
| `--point_index` | `bfm_468_indices.json` | Path to the 468-point index file. |

### Example with all options

```bash
python analyse_video4.py \
    -f /data/subject1/vid_3m_nodding/ \
    --onnx \
    --detector scrfd \
    --fps 30
```

---

## Output files

All output CSVs are saved **inside the input directory**, prefixed with the directory name. For an input folder called `vid_3m_nodding`, the outputs are:

```
vid_3m_nodding/
├── vid_3m_nodding_landmarks.csv
├── vid_3m_nodding_pose.csv
├── vid_3m_nodding_lips.csv
├── vid_3m_nodding_left_eye.csv
├── vid_3m_nodding_right_eye.csv
├── vid_3m_nodding_left_cheek.csv
└── vid_3m_nodding_right_cheek.csv
```

---

### `_landmarks.csv`

One row per frame. Contains all 468 tracked points, each with 5 values:

| Column pattern | Description |
|---|---|
| `pt0_x`, `pt0_y`, `pt0_z` | 3D position in BFM model space, **relative to the nose tip**, in micrometers. X is left/right (negative = left of nose), Y is up/down (negative = above nose), Z is depth (positive = toward camera). |
| `pt0_px`, `pt0_py` | 2D pixel coordinates in the original image frame. Origin is the top-left corner of the image. X increases rightward, Y increases downward. |

This repeats for `pt0` through `pt467`, giving **2,341 columns** total (1 frame column + 468 × 5).

> **3D vs 2D:** The 3D coordinates are distance-independent and suitable for comparing motion across the 1–5m distance range. The 2D pixel coordinates are absolute image positions — they will differ across distances as the face appears at different sizes and locations in the frame. Use 2D coords for visualisation and as input for downstream depth estimation.

---

### `_pose.csv`

One row per frame. Global head orientation extracted from the 3DMM transformation matrix.

| Column | Description |
|---|---|
| `frame` | Frame index |
| `yaw` | Left/right head rotation in degrees |
| `pitch` | Up/down head rotation in degrees |
| `roll` | Tilt head rotation in degrees |

---

### `_lips.csv`, `_left_eye.csv`, `_right_eye.csv`, `_left_cheek.csv`, `_right_cheek.csv`

One row per frame. Per-region kinematics computed from the **centroid** of each region's landmarks, after nose subtraction.

| Column | Description |
|---|---|
| `frame` | Frame index |
| `pos_x`, `pos_y`, `pos_z` | Centroid position relative to nose tip (micrometers) |
| `vel_x`, `vel_y`, `vel_z` | Velocity in micrometers per second |
| `acc_x`, `acc_y`, `acc_z` | Acceleration in micrometers per second² |

Velocity and acceleration are computed using `numpy.gradient` (central differences, one-sided at edges). All values are nose-subtracted — they reflect facial region motion only, not global head movement.

**Region landmark indices (68-point BFM scheme):**

| Region | Landmark indices |
|---|---|
| Right eye | 36–41 |
| Left eye | 42–47 |
| Lips | 48–67 (inner + outer) |
| Right cheek | 1, 2, 3 (upper jaw contour proxy) |
| Left cheek | 13, 14, 15 (upper jaw contour proxy) |

> **Cheek note:** The 68-point BFM landmark set has no explicit cheek landmarks. The upper jaw contour points are the closest anatomical proxy and move with cheek dynamics well enough for velocity and acceleration analysis.

---

## Sharing with collaborators using other methods

`bfm_468_pointmap.json` is the interoperability file. Each entry looks like:

```json
{
  "point_id": 12,
  "bfm_vertex_index": 8423,
  "mean_face_x": -14230.5,
  "mean_face_y": -8810.2,
  "mean_face_z": 3200.1,
  "region": "left_eye"
}
```

- `point_id` — the shared reference number. Your `pt12` in the landmarks CSV corresponds to `point_id: 12` here.
- `mean_face_x/y/z` — the 3D position of this point on the BFM average face, in micrometers.
- `region` — which facial region this point belongs to.

A collaborator using a different face tracking method (e.g. InsightFace) loads this file, normalizes both their landmarks and the `mean_face_x/y/z` coordinates to the same scale, and finds the nearest point in their mesh for each of the 468 entries. This gives them a `point_id`-consistent landmark set that can be compared directly to the output of this pipeline. A helper script `match_landmarks.py` is provided to do this matching.

---

## Regenerating the 468-point index (optional)

The `bfm_468_indices.json` file is already committed and ready to use. You only need to regenerate it if you want a different point distribution or different region budgets. To regenerate:

```bash
python -c "from utils.fps_points import build_and_save_point_index; build_and_save_point_index()"
```

This overwrites both `bfm_468_indices.json` and `bfm_468_pointmap.json`. If you share regenerated files with collaborators, make sure everyone is using the same version — the `point_id` numbers will change if the index is regenerated.

---

## Detector choice guide

| Distance | Recommended detector |
|---|---|
| 1–2 metres | `facebox` (default, no extra dependencies) |
| 3 metres | Either works, `facebox` is fine |
| 4–5 metres | `scrfd` strongly recommended |

SCRFD is better at detecting small faces (down to ~16px wide) and is actually faster than FaceBoxes in most benchmarks. The only reason not to use it by default is the additional `insightface` dependency.

---

## Original 3DDFA_V2

All credit for the underlying 3D face reconstruction goes to the original authors. Please cite their work if you use this in research:

> Guo, J., Zhu, X., Yang, Y., Yang, F., Lei, Z., & Li, S. Z. (2020). Towards Fast, Accurate and Stable 3D Dense Face Alignment. ECCV 2020.
