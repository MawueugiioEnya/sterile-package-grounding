#!/usr/bin/env python3
"""
Evaluate RGB-only Pixtral event-grounding outputs against the manual subset.

Boundary MAE is reported only where a prediction exists, together with
explicit detection coverage. tIoU uses the fixed GT-valid cohort: a missing
predicted phase receives tIoU = 0 rather than disappearing from evaluation.
"""

import argparse
import json
import statistics
from pathlib import Path


PHASES = (
    "locate_flaps",
    "peel_apart",
    "drop_contents",
)


def starts(phases, label):
    vals = []
    for r in phases.get(label, []):
        if isinstance(r, list) and len(r) >= 1:
            vals.append(int(r[0]))
    return sorted(vals)


def hand_control_start(hand):
    phases = hand.get("phases", {})

    grip_ends = [
        int(r[1])
        for r in phases.get("grip", [])
        if isinstance(r, list) and len(r) >= 2
    ]
    if grip_ends:
        return max(grip_ends)

    hold_starts = [
        int(r[0])
        for r in phases.get("hold", [])
        if isinstance(r, list) and len(r) >= 2
    ]
    if hold_starts:
        return min(hold_starts)

    return None


def gt_grasp_start(gt):
    vals = []
    for side in ("left", "right"):
        frame = hand_control_start(gt.get(side, {}))
        if frame is not None:
            vals.append(frame)

    if len(vals) >= 2:
        return max(vals)
    if len(vals) == 1:
        return vals[0]
    return None


def gt_pull_start(gt):
    vals = []

    for side in ("left", "right"):
        phases = gt.get(side, {}).get("phases", {})
        pulls = starts(phases, "pull")
        if pulls:
            vals.append(min(pulls))

    if len(vals) >= 2:
        return max(vals)
    if len(vals) == 1:
        return vals[0]
    return None


def gt_release_start(gt):
    vals = []

    for side in ("left", "right"):
        phases = gt.get(side, {}).get("phases", {})
        releases = starts(phases, "release")
        if releases:
            vals.append(min(releases))

    return min(vals) if vals else None


def abs_err(pred, gt):
    if pred is None or gt is None:
        return None
    return abs(int(pred) - int(gt))


def tiou(a, b):
    a0, a1 = a
    b0, b1 = b

    intersection = max(
        0,
        min(a1, b1) - max(a0, b0) + 1,
    )

    union = (
        (a1 - a0 + 1)
        + (b1 - b0 + 1)
        - intersection
    )

    return intersection / union if union > 0 else 0.0


def summary(rows, metric):
    gt_key = metric.replace("_error", "_gt")

    eligible = [
        row
        for row in rows
        if row.get(gt_key) is not None
    ]

    vals = [
        int(row[metric])
        for row in eligible
        if row.get(metric) is not None
    ]

    print(f"\n{metric}")
    print(f"  GT eligible : {len(eligible)}")

    if eligible:
        print(
            f"  detected    : {len(vals)}/{len(eligible)} "
            f"({100 * len(vals) / len(eligible):.1f}%)"
        )
    else:
        print("  detected    : 0/0")

    if not vals:
        return

    print(
        f"  MAE         : "
        f"{statistics.mean(vals):.2f} frames"
    )
    print(
        f"  median      : "
        f"{statistics.median(vals):.2f} frames"
    )
    print(f"  min         : {min(vals)}")
    print(f"  max         : {max(vals)}")

    for tol in (2, 5, 10):
        good = sum(v <= tol for v in vals)
        print(
            f"  <= {tol:2d} frames : "
            f"{good}/{len(vals)} "
            f"({100 * good / len(vals):.1f}% of detected)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cras_dir",
        default="cras_annotations",
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
    )
    args = parser.parse_args()

    cras_dir = Path(args.cras_dir)
    pred_dir = Path(args.pred_dir)

    annotation_files = sorted(
        cras_dir.rglob("*_annotations.json")
    )

    boundary_rows = []
    tiou_rows = []
    missing_files = []
    missing_boundaries = []
    invalid_gt_for_tiou = []

    for cp in annotation_files:
        seq = cp.stem.replace("_annotations", "")

        pp = pred_dir / f"{seq}.json"
        if not pp.exists():
            missing_files.append(seq)
            continue

        gt = json.loads(cp.read_text())
        pred_doc = json.loads(pp.read_text())

        boundaries = pred_doc.get("boundaries", {})

        grasp_pred = boundaries.get("grasp_flaps")
        peel_pred = boundaries.get("peel_apart")
        drop_pred = boundaries.get("drop_contents")

        first = int(pred_doc.get("first_frame", 0))
        last = int(pred_doc.get("last_frame", 0))

        grasp_gt = gt_grasp_start(gt)
        peel_gt = gt_pull_start(gt)
        drop_gt = gt_release_start(gt)

        row = {
            "sequence": seq,
            "grasp_gt": grasp_gt,
            "peel_gt": peel_gt,
            "drop_gt": drop_gt,
            "grasp_pred": grasp_pred,
            "peel_pred": peel_pred,
            "drop_pred": drop_pred,
            "grasp_error": abs_err(grasp_pred, grasp_gt),
            "peel_error": abs_err(peel_pred, peel_gt),
            "drop_error": abs_err(drop_pred, drop_gt),
        }
        boundary_rows.append(row)

        missing = []
        if grasp_pred is None:
            missing.append("grasp")
        if peel_pred is None:
            missing.append("peel")
        if drop_pred is None:
            missing.append("drop")

        if missing:
            missing_boundaries.append((seq, missing))

        # Fixed tIoU cohort is defined only by manual GT validity.
        if any(
            x is None
            for x in (grasp_gt, peel_gt, drop_gt)
        ):
            invalid_gt_for_tiou.append(seq)
            continue

        if not (
            0 <= grasp_gt <= peel_gt <= drop_gt <= last
        ):
            invalid_gt_for_tiou.append(seq)
            continue

        gt_intervals = {
            "locate_flaps": (
                0,
                max(0, grasp_gt - 1),
            ),
            "peel_apart": (
                peel_gt,
                max(peel_gt, drop_gt - 1),
            ),
            "drop_contents": (
                drop_gt,
                last,
            ),
        }

        locate_score = 0.0
        peel_score = 0.0
        drop_score = 0.0

        if grasp_pred is not None:
            pred_locate = (
                first,
                max(first, int(grasp_pred) - 1),
            )
            locate_score = tiou(
                gt_intervals["locate_flaps"],
                pred_locate,
            )

        if (
            peel_pred is not None
            and drop_pred is not None
            and int(drop_pred) >= int(peel_pred)
        ):
            pred_peel = (
                int(peel_pred),
                max(
                    int(peel_pred),
                    int(drop_pred) - 1,
                ),
            )
            peel_score = tiou(
                gt_intervals["peel_apart"],
                pred_peel,
            )

        if drop_pred is not None:
            pred_drop = (
                int(drop_pred),
                last,
            )
            drop_score = tiou(
                gt_intervals["drop_contents"],
                pred_drop,
            )

        tiou_rows.append(
            {
                "sequence": seq,
                "locate_flaps": locate_score,
                "peel_apart": peel_score,
                "drop_contents": drop_score,
            }
        )

    print("=" * 72)
    print("ABLATION: PIXTRAL ONLY, RGB EVENT GROUNDING")
    print("=" * 72)
    print("Manual annotation files:", len(annotation_files))
    print("Prediction files evaluated:", len(boundary_rows))
    print("Missing prediction files:", len(missing_files))

    summary(boundary_rows, "grasp_error")
    summary(boundary_rows, "peel_error")
    summary(boundary_rows, "drop_error")

    print()
    print("-" * 72)
    print("TEMPORAL IoU")
    print("-" * 72)
    print(
        "GT-valid tIoU sequences with predictions:",
        len(tiou_rows),
    )

    for phase in PHASES:
        vals = [row[phase] for row in tiou_rows]
        if not vals:
            continue

        good = sum(value >= 0.50 for value in vals)
        zeros = sum(value == 0.0 for value in vals)

        print(f"\n{phase}")
        print(
            f"  mean tIoU    : "
            f"{statistics.mean(vals):.3f}"
        )
        print(
            f"  median tIoU  : "
            f"{statistics.median(vals):.3f}"
        )
        print(
            f"  tIoU >= 0.50 : "
            f"{good}/{len(vals)} "
            f"({100 * good / len(vals):.1f}%)"
        )
        print(
            f"  zero tIoU    : "
            f"{zeros}/{len(vals)} "
            f"({100 * zeros / len(vals):.1f}%)"
        )

    print()
    print("-" * 72)
    print("PER-SEQUENCE BOUNDARY ERRORS")
    print("-" * 72)

    for row in boundary_rows:
        print(
            f"{row['sequence']:16s} "
            f"grasp={str(row['grasp_error']):>4s} "
            f"peel={str(row['peel_error']):>4s} "
            f"drop={str(row['drop_error']):>4s}"
        )

    if missing_files:
        print()
        print("Missing prediction files:")
        for seq in missing_files:
            print(f"  {seq}")

    if missing_boundaries:
        print()
        print("Pixtral event boundaries not detected:")
        for seq, names in missing_boundaries:
            print(
                f"  {seq}: "
                + ", ".join(names)
            )

    if invalid_gt_for_tiou:
        print()
        print(
            "Excluded from tIoU due to incomplete/invalid manual GT:"
        )
        for seq in invalid_gt_for_tiou:
            print(f"  {seq}")


if __name__ == "__main__":
    main()
