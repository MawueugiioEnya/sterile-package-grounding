#!/usr/bin/env python3
"""
Geometry + motion only ablation for sterile-package temporal grounding.

This baseline removes Pixtral semantic resolution and evaluates the
deterministic event proposals produced by detect_bt_events.py.

Prediction mapping:
    grasp_flaps   = grasp candidate preceding first opening burst
    peel_apart    = first opening-burst onset
    drop_contents = terminal release candidate

Ground-truth correspondences and evaluation metrics match the main
CRAS/RA-L temporal-grounding evaluation.
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


# ============================================================
# Ground-truth semantic correspondences
# ============================================================

def starts(phases, label):
    """Return sorted start frames for a manual phase."""
    vals = []

    for r in phases.get(label, []):
        if isinstance(r, list) and len(r) >= 1:
            vals.append(int(r[0]))

    return sorted(vals)


def hand_control_start(hand):
    """
    Secure control is established at completion of the grip action,
    rather than at grip onset.
    """
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
    """
    Secure bimanual control begins when the later required hand
    has established package control.
    """
    vals = []

    for side in ("left", "right"):
        frame = hand_control_start(
            gt.get(side, {})
        )

        if frame is not None:
            vals.append(frame)

    if len(vals) >= 2:
        return max(vals)

    if len(vals) == 1:
        return vals[0]

    return None


def gt_pull_start(gt):
    """
    Peel begins when sustained pulling starts.

    If both hands have pull annotations, use the later onset so that
    both are participating. If only one hand pulls while the other
    maintains hold, use the available pull onset.
    """
    vals = []

    for side in ("left", "right"):
        phases = (
            gt.get(side, {})
              .get("phases", {})
        )

        pulls = starts(
            phases,
            "pull",
        )

        if pulls:
            vals.append(min(pulls))

    if len(vals) >= 2:
        return max(vals)

    if len(vals) == 1:
        return vals[0]

    return None


def gt_release_start(gt):
    """
    Terminal release begins at the earliest annotated release onset.
    """
    vals = []

    for side in ("left", "right"):
        phases = (
            gt.get(side, {})
              .get("phases", {})
        )

        releases = starts(
            phases,
            "release",
        )

        if releases:
            vals.append(min(releases))

    if not vals:
        return None

    return min(vals)


# ============================================================
# Geometry + motion only prediction
# ============================================================

def geometry_prediction(events):
    """
    Convert geometry/motion proposals directly into task boundaries.

    No Pixtral-generated information is used.
    """
    bursts = events.get(
        "opening_burst_candidates",
        [],
    )

    if not bursts:
        return None

    first_burst = bursts[0]

    grasp = first_burst.get(
        "preceding_grasp_candidate"
    )

    peel = first_burst.get(
        "opening_candidate"
    )

    drop = (
        events.get(
            "terminal_release_candidate",
            {},
        )
        .get("drop_candidate")
    )

    if any(
        x is None
        for x in (grasp, peel, drop)
    ):
        return None

    frame_range = events.get(
        "frame_range",
        [0, 0],
    )

    return {
        "grasp": int(grasp),
        "peel": int(peel),
        "drop": int(drop),
        "first": int(frame_range[0]),
        "last": int(frame_range[1]),
    }


# ============================================================
# Metrics
# ============================================================

def abs_err(pred, gt):
    if pred is None or gt is None:
        return None

    return abs(
        int(pred) - int(gt)
    )


def tiou(a, b):
    """Inclusive-frame temporal IoU."""
    a0, a1 = a
    b0, b1 = b

    intersection = max(
        0,
        min(a1, b1)
        - max(a0, b0)
        + 1,
    )

    union = (
        (a1 - a0 + 1)
        + (b1 - b0 + 1)
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def print_boundary_summary(rows, metric):
    vals = [
        int(r[metric])
        for r in rows
        if r[metric] is not None
    ]

    if not vals:
        print(f"\n{metric}: no valid values")
        return

    print(f"\n{metric}")
    print(f"  n      : {len(vals)}")
    print(
        f"  MAE    : "
        f"{statistics.mean(vals):.2f} frames"
    )
    print(
        f"  median : "
        f"{statistics.median(vals):.2f} frames"
    )
    print(f"  min    : {min(vals)}")
    print(f"  max    : {max(vals)}")

    for tol in (2, 5, 10):
        good = sum(
            value <= tol
            for value in vals
        )

        print(
            f"  <= {tol:2d} frames: "
            f"{good}/{len(vals)} "
            f"({100 * good / len(vals):.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cras_dir",
        default="cras_annotations",
        help="Directory containing manual CRAS annotation JSON files.",
    )

    parser.add_argument(
        "--events_dir",
        default="event_candidates",
        help="Directory containing geometry/motion event proposals.",
    )

    args = parser.parse_args()

    cras_dir = Path(args.cras_dir)
    events_dir = Path(args.events_dir)

    boundary_rows = []
    tiou_rows = []

    missing_predictions = []

    for cp in sorted(
        cras_dir.rglob(
            "*_annotations.json"
        )
    ):
        seq = cp.stem.replace(
            "_annotations",
            "",
        )

        ep = (
            events_dir
            / f"{seq}.json"
        )

        if not ep.exists():
            continue

        gt = json.loads(
            cp.read_text()
        )

        events = json.loads(
            ep.read_text()
        )

        pred = geometry_prediction(
            events
        )

        if pred is None:
            missing_predictions.append(
                seq
            )
            continue

        grasp_gt = gt_grasp_start(gt)
        peel_gt = gt_pull_start(gt)
        drop_gt = gt_release_start(gt)

        boundary_rows.append(
            {
                "sequence": seq,
                "grasp_error":
                    abs_err(
                        pred["grasp"],
                        grasp_gt,
                    ),
                "peel_error":
                    abs_err(
                        pred["peel"],
                        peel_gt,
                    ),
                "drop_error":
                    abs_err(
                        pred["drop"],
                        drop_gt,
                    ),
            }
        )

        # --------------------------------------------------------
        # Same ordered semantic interval criterion used by the
        # main tIoU evaluation.
        # --------------------------------------------------------

        if any(
            value is None
            for value in (
                grasp_gt,
                peel_gt,
                drop_gt,
            )
        ):
            continue

        sequence_end = pred["last"]

        if not (
            0
            <= grasp_gt
            <= peel_gt
            <= drop_gt
            <= sequence_end
        ):
            continue

        if not (
            pred["first"]
            <= pred["grasp"]
            <= pred["peel"]
            <= pred["drop"]
            <= pred["last"]
        ):
            continue

        gt_intervals = {
            "locate_flaps": (
                0,
                max(
                    0,
                    grasp_gt - 1,
                ),
            ),
            "peel_apart": (
                peel_gt,
                max(
                    peel_gt,
                    drop_gt - 1,
                ),
            ),
            "drop_contents": (
                drop_gt,
                sequence_end,
            ),
        }

        pred_intervals = {
            "locate_flaps": (
                pred["first"],
                max(
                    pred["first"],
                    pred["grasp"] - 1,
                ),
            ),
            "peel_apart": (
                pred["peel"],
                max(
                    pred["peel"],
                    pred["drop"] - 1,
                ),
            ),
            "drop_contents": (
                pred["drop"],
                pred["last"],
            ),
        }

        scores = {
            phase: tiou(
                gt_intervals[phase],
                pred_intervals[phase],
            )
            for phase in PHASES
        }

        tiou_rows.append(
            {
                "sequence": seq,
                **scores,
            }
        )

    print("=" * 72)
    print(
        "ABLATION: GEOMETRY + MOTION ONLY"
    )
    print("=" * 72)

    print(
        "Boundary-evaluation sequences:",
        len(boundary_rows),
    )

    print_boundary_summary(
        boundary_rows,
        "grasp_error",
    )

    print_boundary_summary(
        boundary_rows,
        "peel_error",
    )

    print_boundary_summary(
        boundary_rows,
        "drop_error",
    )

    print()
    print("-" * 72)
    print("TEMPORAL IoU")
    print("-" * 72)
    print(
        "Evaluated sequences:",
        len(tiou_rows),
    )

    for phase in PHASES:
        vals = [
            row[phase]
            for row in tiou_rows
        ]

        if not vals:
            continue

        good = sum(
            value >= 0.50
            for value in vals
        )

        print(f"\n{phase}")
        print(
            f"  mean tIoU   : "
            f"{statistics.mean(vals):.3f}"
        )
        print(
            f"  median tIoU : "
            f"{statistics.median(vals):.3f}"
        )
        print(
            f"  tIoU >= 0.50: "
            f"{good}/{len(vals)} "
            f"({100 * good / len(vals):.1f}%)"
        )

    print()
    print("-" * 72)
    print(
        "PER-SEQUENCE BOUNDARY ERRORS"
    )
    print("-" * 72)

    for row in boundary_rows:
        print(
            f"{row['sequence']:16s} "
            f"grasp="
            f"{str(row['grasp_error']):>4s} "
            f"peel="
            f"{str(row['peel_error']):>4s} "
            f"drop="
            f"{str(row['drop_error']):>4s}"
        )

    if missing_predictions:
        print()
        print(
            "Sequences without valid geometry/motion prediction:"
        )
        for seq in missing_predictions:
            print(
                f"  {seq}"
            )


if __name__ == "__main__":
    main()
