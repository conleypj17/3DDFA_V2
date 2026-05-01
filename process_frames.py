"""
process_frames.py - Run 3DDFA_V2 on a folder of PNG images and save:
  - roll, pitch, yaw per frame to a CSV file
  - dense mesh overlay images (optional)

Usage:
    python process_frames.py --img_dir /path/to/your/frames --csv_out results.csv

Optional flags:
    --save_vis          Save dense-mesh annotated images to a folder
    --vis_dir           Where to save them (default: vis_results/)
    --alpha             Mesh overlay transparency, 0.0-1.0 (default: 0.6)
    --cfg               3DDFA config (default: configs/mb1_120x120.yml)
"""

import os
import csv
import glob
import argparse
import cv2
import yaml
import numpy as np
import sys

# ── make sure we can import from the 3DDFA_V2 root ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import calc_pose
from utils.render import render          # dense mesh renderer (uses render.so)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',  required=True,
                        help='Folder containing PNG (or JPG) frames')
    parser.add_argument('--csv_out',  default='pose_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save dense-mesh annotated images')
    parser.add_argument('--vis_dir',  default='vis_results',
                        help='Where to save annotated images')
    parser.add_argument('--alpha',    type=float, default=0.6,
                        help='Mesh overlay transparency (0=invisible, 1=solid)')
    parser.add_argument('--cfg',      default='configs/mb1_120x120.yml',
                        help='3DDFA config file')
    return parser.parse_args()


def get_sorted_frames(img_dir):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(files)


def main():
    args = parse_args()

    # ── load config & initialise models ─────────────────────────────────────
    cfg = yaml.load(open(args.cfg), Loader=yaml.SafeLoader)
    face_boxes = FaceBoxes()
    tddfa = TDDFA(**cfg)

    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f'Dense mesh images will be saved to: {args.vis_dir}/')

    frames = get_sorted_frames(args.img_dir)
    if not frames:
        print(f'No images found in {args.img_dir}')
        return

    print(f'Found {len(frames)} frames in {args.img_dir}\n')

    with open(args.csv_out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'face_id', 'pitch', 'yaw', 'roll'])

        for frame_idx, img_path in enumerate(frames):
            frame_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f'  [WARN] Could not read {img_path}, skipping.')
                continue

            # ── face detection ───────────────────────────────────────────────
            boxes = face_boxes(img)
            if len(boxes) == 0:
                print(f'  [{frame_idx:04d}] {frame_name}: no faces detected')
                writer.writerow([frame_name, -1, '', '', ''])
                continue

            # ── 3DMM fitting ─────────────────────────────────────────────────
            param_lst, roi_box_lst = tddfa(img, boxes)

            # ── pose estimation ──────────────────────────────────────────────
            for face_id, param in enumerate(param_lst):
                _, pose = calc_pose(param)
                pitch, yaw, roll = pose[0], pose[1], pose[2]
                writer.writerow([frame_name, face_id,
                                 f'{pitch:.4f}', f'{yaw:.4f}', f'{roll:.4f}'])
                print(f'  [{frame_idx:04d}] {frame_name} '
                      f'face={face_id}  '
                      f'pitch={pitch:+6.1f}  yaw={yaw:+6.1f}  roll={roll:+6.1f}')

            # ── dense mesh overlay ───────────────────────────────────────────
            if args.save_vis:
                # recon_vers with dense_flag=True gives the full 38k-vertex mesh
                ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

                # render() blends the mesh onto a copy of the image
                mesh_img = render(img, ver_lst, tddfa.tri, alpha=args.alpha, show_flag=False)

                out_path = os.path.join(args.vis_dir, frame_name)
                cv2.imwrite(out_path, mesh_img)

    print(f'\nDone!')
    print(f'  CSV  -> {args.csv_out}')
    if args.save_vis:
        print(f'  Mesh -> {args.vis_dir}/')


if __name__ == '__main__':
    main()
