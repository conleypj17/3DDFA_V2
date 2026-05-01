"""
analyse_video.py

Run this from the 3DDFA_V2 repo root.

Before running for the first time, generate the 468-point index file:
    python -c "from utils.fps_points import build_and_save_point_index; build_and_save_point_index()"

This only needs to be done once. It creates:
    bfm_468_indices.json     — loaded automatically every run
    bfm_468_pointmap.json    — share this with collaborators using other methods

Usage:
    python analyse_video.py -f path/to/image_directory --onnx
    python analyse_video.py -f path/to/image_directory --onnx --detector scrfd
    python analyse_video.py -f path/to/image_directory --onnx --fps 30

Output (saved in the image directory):
    _right_eye.csv   }
    _left_eye.csv    } per-region nose-subtracted kinematics (pos/vel/acc xyz)
    _lips.csv        }
    _right_cheek.csv }
    _left_cheek.csv  }
    _pose.csv          global head yaw, pitch, roll per frame
    _landmarks.csv     all 468 points per frame with two coordinate sets:
                         - 3D mesh coordinates relative to nose (pt#_x, pt#_y, pt#_z)
                         - 2D projected pixel coordinates in original image (pt#_px, pt#_py)

Landmark CSV column layout (per point, repeated 468 times):
    pt0_x, pt0_y, pt0_z   — 3D model space, nose-subtracted (micrometers)
    pt0_px, pt0_py         — 2D pixel location in the original image frame

Why both?
    3D coords: distance-independent, comparable across 1m-5m, has depth.
    2D coords: directly visible in the image, useful for visualisation and
               cross-checking that points land where you expect on the face.
"""

import argparse
import os
import csv
import cv2
import numpy as np
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from TDDFA_ONNX import TDDFA_ONNX
from utils.region_motion import process_video_landmarks
from utils.pose import calc_pose
from utils.fps_points import load_point_index


# ---------------------------------------------------------------------------
# InsightFace / SCRFD wrapper
# ---------------------------------------------------------------------------

def load_scrfd():
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "InsightFace is not installed.\n"
            "Run:  pip install insightface onnxruntime"
        )
    app = FaceAnalysis(
        name="buffalo_sc",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def scrfd_detect(app, frame):
    faces = app.get(frame)
    if not faces:
        return []
    boxes = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(np.float32)
        conf = float(face.det_score)
        boxes.append(np.array([x1, y1, x2, y2, conf], dtype=np.float32))
    return boxes


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Per-region facial motion analyser")
    parser.add_argument("-c", "--config",    type=str, default="configs/mb1_120x120.yml")
    parser.add_argument("-f", "--input",     type=str, required=True,
                        help="Path to directory containing PNG images")
    parser.add_argument("--fps",             type=float, default=30.0,
                        help="Frames per second (default: 30.0)")
    parser.add_argument("--onnx",            action="store_true", default=False,
                        help="Use ONNX runtime for the 3DMM regressor (recommended)")
    parser.add_argument("--detector",        type=str, default="facebox",
                        choices=["facebox", "scrfd"],
                        help="facebox (default) or scrfd (better for 4-5m footage)")
    parser.add_argument("--point_index",     type=str, default="bfm_468_indices.json",
                        help="Path to the 468-point index JSON (from fps_points.py)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CSV savers
# ---------------------------------------------------------------------------

def save_region_results(results, input_path):
    """Save one CSV per region (pos/vel/acc xyz, nose-subtracted)."""
    base = input_path.rstrip(os.sep)
    header = "frame,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z"
    for region_name, data in results.items():
        out_path = f"{base}_{region_name}.csv"
        n_frames = data["position"].shape[0]
        rows = []
        for i in range(n_frames):
            p = data["position"][i]
            v = data["velocity"][i]
            a = data["acceleration"][i]
            rows.append(
                f"{i},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},"
                f"{v[0]:.4f},{v[1]:.4f},{v[2]:.4f},"
                f"{a[0]:.4f},{a[1]:.4f},{a[2]:.4f}"
            )
        with open(out_path, "w") as f:
            f.write(header + "\n")
            f.write("\n".join(rows))
        print(f"  Saved: {out_path}")


def save_pose(pose_sequence, input_path):
    """Save global head pose (yaw, pitch, roll) per frame."""
    base = input_path.rstrip(os.sep)
    out_path = f"{base}_pose.csv"
    with open(out_path, "w") as f:
        f.write("frame,yaw,pitch,roll\n")
        for i, (yaw, pitch, roll) in enumerate(pose_sequence):
            f.write(f"{i},{yaw:.4f},{pitch:.4f},{roll:.4f}\n")
    print(f"  Saved: {out_path}")


def save_landmarks(landmark_sequence, input_path, n_points=468):
    """
    Save all 468 points per frame with two coordinate sets:

      pt#_x, pt#_y, pt#_z  — 3D model space, nose-subtracted (micrometers)
      pt#_px, pt#_py        — 2D pixel location in the original image frame

    Each frame is one row. Columns repeat the 5-value block for each of
    the 468 points.

    How 2D pixel coords are obtained:
        recon_vers() in 3DDFA-V2 returns the 3D mesh already projected back
        into the original image coordinate system — x and y are pixel locations,
        z is model depth. We read x,y before the nose subtraction so the pixel
        coords are absolute image positions, not relative to the nose.

    Args:
        landmark_sequence: list of dicts, one per frame, each with keys:
                           'xyz'  — np.ndarray (468, 3) nose-subtracted 3D coords
                           'pxy'  — np.ndarray (468, 2) pixel coords (x, y)
        input_path:        str, the image directory path (used to build output path)
        n_points:          int, should be 468
    """
    base = input_path.rstrip(os.sep)
    out_path = f"{base}_landmarks.csv"

    # Build header: frame, pt0_x, pt0_y, pt0_z, pt0_px, pt0_py, pt1_x, ...
    header = ["frame"]
    for i in range(n_points):
        header += [f"pt{i}_x", f"pt{i}_y", f"pt{i}_z",   # 3D nose-relative
                   f"pt{i}_px", f"pt{i}_py"]              # 2D pixel

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for frame_idx, data in enumerate(landmark_sequence):
            xyz = data["xyz"]   # (468, 3) — 3D nose-subtracted
            pxy = data["pxy"]   # (468, 2) — 2D pixel coords
            
            # Debug: verify shapes and first point data
            if frame_idx == 0:
                print(f"  DEBUG: xyz shape = {xyz.shape}, pxy shape = {pxy.shape}")
                print(f"  DEBUG: point 0 — xyz = {xyz[0]}, pxy = {pxy[0]}")

            row = [frame_idx]
            for i in range(n_points):
                row += [
                    round(float(xyz[i, 0]), 4),
                    round(float(xyz[i, 1]), 4),
                    round(float(xyz[i, 2]), 4),
                    round(float(pxy[i, 0]), 2),  # pixel — 2 decimal places is plenty
                    round(float(pxy[i, 1]), 2),
                ]
            writer.writerow(row)

    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Load 3DMM config and regressor ---
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    tddfa = TDDFA_ONNX(**cfg) if args.onnx else TDDFA(**cfg)

    # --- Load detector ---
    if args.detector == "scrfd":
        print("Detector: SCRFD (InsightFace)")
        scrfd_app = load_scrfd()
        def detect(frame): return scrfd_detect(scrfd_app, frame)
    else:
        print("Detector: FaceBoxes")
        fb = FaceBoxes()
        def detect(frame): return fb(frame)

    # --- Load 468-point index ---
    print(f"Loading point index: {args.point_index}")
    point_indices, region_map, nose_dense_idx = load_point_index(args.point_index)
    n_points = len(point_indices)
    print(f"  {n_points} points loaded")

    # --- Load PNG images ---
    if not os.path.isdir(args.input):
        raise RuntimeError(f"Directory not found: {args.input}")

    png_files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(".png")])
    if not png_files:
        raise RuntimeError(f"No PNG files found in: {args.input}")

    print(f"Input:  {args.input}")
    print(f"FPS:    {args.fps:.2f}  |  Frames: {len(png_files)}")

    # --- Process frames ---
    ver_sequence_68    = []   # (3, 68) per frame — for region kinematics
    landmark_sequence  = []   # list of {'xyz': (468,3), 'pxy': (468,2)} per frame
    pose_sequence      = []
    param_sequence     = []
    skipped = 0

    for frame_idx, png_file in enumerate(png_files):
        frame = cv2.imread(os.path.join(args.input, png_file))
        if frame is None:
            skipped += 1
            continue

        h, w = frame.shape[:2]
        detected = detect(frame)
        if not detected:
            skipped += 1
            continue

        detected_box = detected[0]
        coords = detected_box[:4]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            skipped += 1
            continue

        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        try:
            param_lst, roi_box_lst = tddfa(frame, [box], crop_policy="box")

            # --- 68-point sparse landmarks (for region kinematics) ---
            ver_68 = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
            ver_sequence_68.append(ver_68)

            # --- Dense mesh (3, 38365) ---
            # recon_vers returns the mesh in original image space:
            #   ver_full[0] = x pixel coordinates in the original frame
            #   ver_full[1] = y pixel coordinates in the original frame
            #   ver_full[2] = z depth in model space
            ver_full = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]

            # --- 2D pixel coordinates (before any subtraction) ---
            # Just read x, y directly — these are already in image pixel space.
            px = ver_full[0, point_indices]   # (468,) x pixel coords
            py = ver_full[1, point_indices]   # (468,) y pixel coords
            pxy = np.stack([px, py], axis=1)  # (468, 2)

            # --- 3D nose-relative coordinates ---
            # Subtract nose tip position in 3D model space so that the 3D
            # coords reflect facial motion, not global head movement.
            nose_pos = ver_full[:, nose_dense_idx]              # (3,)
            ver_relative = ver_full - nose_pos[:, np.newaxis]   # (3, 38365)
            xyz = ver_relative[:, point_indices].T              # (468, 3)

            landmark_sequence.append({"xyz": xyz, "pxy": pxy})
            param_sequence.append(param_lst[0])

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}...")

        except Exception as e:
            print(f"  Frame {frame_idx} failed: {e}")
            skipped += 1

    print(f"Processed {len(ver_sequence_68)} frames, skipped {skipped}")

    if len(ver_sequence_68) < 3:
        print("Not enough frames for kinematics (need at least 3). Exiting.")
        return

    # --- Head pose ---
    print("Extracting head pose...")
    for param in param_sequence:
        _, pose = calc_pose(param)
        pose_sequence.append(pose)

    # --- Region kinematics (from 68-pt sparse landmarks) ---
    print("Computing region kinematics...")
    results = process_video_landmarks(ver_sequence_68, args.fps)

    # --- Save ---
    print("Saving CSVs...")
    save_region_results(results, args.input)
    save_pose(pose_sequence, args.input)
    save_landmarks(landmark_sequence, args.input, n_points=n_points)
    print("Done.")


if __name__ == "__main__":
    main()
