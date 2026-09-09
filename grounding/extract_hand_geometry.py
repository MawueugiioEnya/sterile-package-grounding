#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


PALM_IDS = [0, 5, 9, 13, 17]
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9


def frame_number(path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=float)
            - np.asarray(b, dtype=float)
        )
    )


def extract_hand(landmarks):

    points = np.asarray(
        [
            [float(p.x), float(p.y)]
            for p in landmarks
        ],
        dtype=np.float32,
    )

    palm = points[PALM_IDS].mean(axis=0)

    grip = (
        points[THUMB_TIP]
        + points[INDEX_TIP]
    ) / 2.0

    hand_scale = distance(
        points[WRIST],
        points[MIDDLE_MCP],
    )

    pinch_gap = distance(
        points[THUMB_TIP],
        points[INDEX_TIP],
    )

    pinch_ratio = (
        pinch_gap / hand_scale
        if hand_scale > 1e-6
        else np.nan
    )

    return {
        "palm": palm,
        "grip": grip,
        "pinch_ratio": pinch_ratio,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seq_dir",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="models/hand_landmarker.task",
    )

    parser.add_argument(
        "--out_csv",
        required=True,
    )

    parser.add_argument(
        "--out_json",
        required=True,
    )

    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    color_dir = seq_dir / "color"
    model_path = Path(args.model)

    frames = sorted(
        [
            p for p in color_dir.iterdir()
            if p.suffix.lower()
            in {".png", ".jpg", ".jpeg"}
        ],
        key=frame_number,
    )

    print(
        f"{seq_dir.name}: "
        f"{len(frames)} source frames"
    )

    if not frames:
        raise RuntimeError(
            f"No frames in {color_dir}"
        )

    previous = {
        0: None,
        1: None,
    }

    def assign_tracks(detections):

        assigned = {
            0: None,
            1: None,
        }

        if not detections:
            return assigned

        if len(detections) == 1:

            d = detections[0]

            if (
                previous[0] is None
                and previous[1] is None
            ):
                assigned[0] = d

            elif previous[0] is None:
                assigned[1] = d

            elif previous[1] is None:
                assigned[0] = d

            else:

                d0 = distance(
                    d["palm"],
                    previous[0],
                )

                d1 = distance(
                    d["palm"],
                    previous[1],
                )

                assigned[
                    0 if d0 <= d1 else 1
                ] = d

        else:

            d0, d1 = detections[:2]

            if (
                previous[0] is None
                or previous[1] is None
            ):

                ordered = sorted(
                    [d0, d1],
                    key=lambda d:
                    d["palm"][0],
                )

                assigned[0] = ordered[0]
                assigned[1] = ordered[1]

            else:

                normal = (
                    distance(
                        d0["palm"],
                        previous[0],
                    )
                    +
                    distance(
                        d1["palm"],
                        previous[1],
                    )
                )

                swapped = (
                    distance(
                        d0["palm"],
                        previous[1],
                    )
                    +
                    distance(
                        d1["palm"],
                        previous[0],
                    )
                )

                if normal <= swapped:
                    assigned[0] = d0
                    assigned[1] = d1
                else:
                    assigned[0] = d1
                    assigned[1] = d0

        for tid in [0, 1]:
            if assigned[tid] is not None:
                previous[tid] = (
                    assigned[tid]["palm"]
                )

        return assigned

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )
    Options = (
        mp.tasks.vision.HandLandmarkerOptions
    )
    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = Options(
        base_options=BaseOptions(
            model_asset_path=str(
                model_path
            )
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.30,
        min_hand_presence_confidence=0.30,
        min_tracking_confidence=0.30,
    )

    rows = []

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        for i, path in enumerate(frames):

            bgr = cv2.imread(str(path))

            if bgr is None:
                continue

            rgb = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2RGB,
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )

            result = (
                landmarker.detect_for_video(
                    image,
                    i * 33,
                )
            )

            detections = [
                extract_hand(x)
                for x
                in result.hand_landmarks[:2]
            ]

            assigned = assign_tracks(
                detections
            )

            row = {
                "frame": frame_number(path),
                "frame_path": str(path),
                "detected_hands":
                    len(detections),
            }

            for tid in [0, 1]:

                hand = assigned[tid]

                if hand is None:

                    for key in [
                        "palm_x",
                        "palm_y",
                        "grip_x",
                        "grip_y",
                        "pinch_ratio",
                    ]:
                        row[
                            f"h{tid}_{key}"
                        ] = np.nan

                else:

                    row[f"h{tid}_palm_x"] = float(
                        hand["palm"][0]
                    )

                    row[f"h{tid}_palm_y"] = float(
                        hand["palm"][1]
                    )

                    row[f"h{tid}_grip_x"] = float(
                        hand["grip"][0]
                    )

                    row[f"h{tid}_grip_y"] = float(
                        hand["grip"][1]
                    )

                    row[
                        f"h{tid}_pinch_ratio"
                    ] = float(
                        hand["pinch_ratio"]
                    )

            rows.append(row)

    df = pd.DataFrame(rows)

    coordinate_columns = [
        "h0_palm_x",
        "h0_palm_y",
        "h1_palm_x",
        "h1_palm_y",
        "h0_grip_x",
        "h0_grip_y",
        "h1_grip_x",
        "h1_grip_y",
        "h0_pinch_ratio",
        "h1_pinch_ratio",
    ]

    for column in coordinate_columns:

        df[column] = (
            df[column].interpolate(
                method="linear",
                limit=3,
                limit_direction="both",
            )
        )

    df["grip_distance"] = np.sqrt(
        (
            df["h1_grip_x"]
            - df["h0_grip_x"]
        ) ** 2
        +
        (
            df["h1_grip_y"]
            - df["h0_grip_y"]
        ) ** 2
    )

    for tid in [0, 1]:

        dx = (
            df[f"h{tid}_grip_x"]
            .diff()
        )

        dy = (
            df[f"h{tid}_grip_y"]
            .diff()
        )

        df[f"h{tid}_speed"] = np.sqrt(
            dx * dx + dy * dy
        )

    df["combined_hand_speed"] = (
        df["h0_speed"]
        + df["h1_speed"]
    )

    df["distance_delta"] = (
        df["grip_distance"].diff()
    )

    df["separating_speed"] = np.maximum(
        df["distance_delta"],
        0.0,
    )

    df["approaching_speed"] = np.maximum(
        -df["distance_delta"],
        0.0,
    )

    smooth = [
        "grip_distance",
        "distance_delta",
        "separating_speed",
        "approaching_speed",
        "combined_hand_speed",
        "h0_pinch_ratio",
        "h1_pinch_ratio",
    ]

    for key in smooth:

        df[f"{key}_mean5"] = (
            df[key].rolling(
                5,
                min_periods=1,
            ).mean()
        )

    sep5 = (
        df["separating_speed"]
        .rolling(
            5,
            min_periods=1,
        )
        .mean()
    )

    sep15 = (
        df["separating_speed"]
        .rolling(
            15,
            min_periods=1,
        )
        .mean()
    )

    df["separation_trend"] = (
        sep5 - sep15
    )

    df["mean_pinch_ratio"] = (
        df[
            [
                "h0_pinch_ratio",
                "h1_pinch_ratio",
            ]
        ].mean(
            axis=1,
            skipna=False,
        )
    )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_csv,
        index=False,
    )

    records = []

    for record in df.to_dict(
        orient="records"
    ):

        clean = {}

        for key, value in record.items():

            if pd.isna(value):
                clean[key] = None
            elif isinstance(
                value,
                np.integer,
            ):
                clean[key] = int(value)
            elif isinstance(
                value,
                np.floating,
            ):
                clean[key] = float(value)
            else:
                clean[key] = value

        records.append(clean)

    out_json.write_text(
        json.dumps(
            {
                "sequence": seq_dir.name,
                "num_frames": len(records),
                "features": records,
            },
            indent=2,
        )
        + "\n"
    )

    one = int(
        (
            df["detected_hands"] >= 1
        ).sum()
    )

    two = int(
        (
            df["detected_hands"] >= 2
        ).sum()
    )

    print(
        f"One+ hand coverage: "
        f"{one}/{len(df)} "
        f"({100*one/len(df):.1f}%)"
    )

    print(
        f"Two-hand coverage: "
        f"{two}/{len(df)} "
        f"({100*two/len(df):.1f}%)"
    )

    print(
        f"Wrote {out_csv}"
    )

    print(
        f"Wrote {out_json}"
    )


if __name__ == "__main__":
    main()
