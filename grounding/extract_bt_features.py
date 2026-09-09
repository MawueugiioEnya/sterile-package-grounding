#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def frame_number(path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def roi_bounds(shape, fraction=0.70):
    h, w = shape[:2]

    rh = int(h * fraction)
    rw = int(w * fraction)

    y0 = (h - rh) // 2
    x0 = (w - rw) // 2

    return x0, y0, x0 + rw, y0 + rh


def estimate_global_motion(previous, current):

    h, w = previous.shape

    mask = np.full(
        (h, w),
        255,
        dtype=np.uint8,
    )

    x0, y0, x1, y1 = roi_bounds(
        previous.shape,
        0.70,
    )

    # Exclude central hand/package area when estimating head/camera motion.
    mask[y0:y1, x0:x1] = 0

    points0 = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        mask=mask,
    )

    if points0 is None or len(points0) < 10:
        return None

    points1, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points0,
        None,
    )

    if points1 is None:
        return None

    good = status.reshape(-1) == 1

    p0 = points0[good].reshape(-1, 2)
    p1 = points1[good].reshape(-1, 2)

    if len(p0) < 8:
        return None

    M, _ = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )

    return M


def stabilise_previous(previous, current):

    M = estimate_global_motion(
        previous,
        current,
    )

    if M is None:
        return previous

    h, w = current.shape

    return cv2.warpAffine(
        previous,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def central_roi(arr, fraction=0.70):

    x0, y0, x1, y1 = roi_bounds(
        arr.shape,
        fraction,
    )

    return arr[y0:y1, x0:x1]


def rolling_mean(rows, key, end_idx, length):

    start = max(
        0,
        end_idx - length + 1,
    )

    values = [
        rows[i][key]
        for i in range(start, end_idx + 1)
    ]

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def extract(seq_dir, out_path):

    seq_dir = Path(seq_dir)
    color_dir = seq_dir / "color"

    frames = sorted(
        [
            p
            for p in color_dir.iterdir()
            if p.suffix.lower()
            in {".png", ".jpg", ".jpeg"}
        ],
        key=frame_number,
    )

    if len(frames) < 2:
        raise RuntimeError(
            f"Not enough frames in {color_dir}"
        )

    print(
        f"{seq_dir.name}: {len(frames)} source frames"
    )

    first = cv2.imread(
        str(frames[0])
    )

    if first is None:
        raise RuntimeError(
            f"Cannot read {frames[0]}"
        )

    previous = cv2.cvtColor(
        first,
        cv2.COLOR_BGR2GRAY,
    )

    rows = [
        {
            "frame": frame_number(frames[0]),
            "frame_path": str(frames[0]),
            "mean_flow": 0.0,
            "p90_flow": 0.0,
            "opposing_horizontal": 0.0,
            "positive_divergence": 0.0,
            "mean_vertical": 0.0,
            "downward_motion": 0.0,
            "upward_motion": 0.0,
        }
    ]

    for path in frames[1:]:

        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        current = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        aligned_previous = stabilise_previous(
            previous,
            current,
        )

        flow = cv2.calcOpticalFlowFarneback(
            aligned_previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        flow = central_roi(
            flow,
            0.70,
        )

        u = flow[..., 0]
        v = flow[..., 1]

        magnitude = np.sqrt(
            u * u + v * v
        )

        midpoint = u.shape[1] // 2

        left_u = u[:, :midpoint]
        right_u = u[:, midpoint:]

        left_outward = np.maximum(
            -left_u,
            0.0,
        )

        right_outward = np.maximum(
            right_u,
            0.0,
        )

        opposing = (
            np.mean(left_outward)
            + np.mean(right_outward)
        ) / 2.0

        du_dx = cv2.Sobel(
            u,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )

        dv_dy = cv2.Sobel(
            v,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        divergence = np.maximum(
            du_dx + dv_dy,
            0.0,
        )

        rows.append(
            {
                "frame": frame_number(path),
                "frame_path": str(path),

                "mean_flow": float(
                    np.mean(magnitude)
                ),

                "p90_flow": float(
                    np.percentile(
                        magnitude,
                        90,
                    )
                ),

                "opposing_horizontal": float(
                    opposing
                ),

                "positive_divergence": float(
                    np.mean(divergence)
                ),

                # Positive means downward in image coordinates.
                "mean_vertical": float(
                    np.mean(v)
                ),

                "downward_motion": float(
                    np.mean(
                        np.maximum(v, 0.0)
                    )
                ),

                "upward_motion": float(
                    np.mean(
                        np.maximum(-v, 0.0)
                    )
                ),
            }
        )

        previous = current

    # ------------------------------------------------------------
    # Add temporal/trend features.
    # ------------------------------------------------------------

    feature_keys = [
        "mean_flow",
        "p90_flow",
        "opposing_horizontal",
        "positive_divergence",
        "mean_vertical",
        "downward_motion",
        "upward_motion",
    ]

    for i, row in enumerate(rows):

        for key in feature_keys:

            short = rolling_mean(
                rows,
                key,
                i,
                5,
            )

            long = rolling_mean(
                rows,
                key,
                i,
                15,
            )

            row[f"{key}_mean5"] = short
            row[f"{key}_mean15"] = long
            row[f"{key}_trend"] = short - long

    out_path = Path(out_path)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "sequence": seq_dir.name,
        "num_frames": len(rows),
        "features": rows,
    }

    out_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )

    print(
        f"Wrote {out_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seq_dir",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    extract(
        args.seq_dir,
        args.out,
    )


if __name__ == "__main__":
    main()
