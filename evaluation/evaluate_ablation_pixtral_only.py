#!/usr/bin/env python3
"""
Pixtral-only ablation for sterile-package temporal grounding.

Input:
    Framewise visual-state predictions produced by pixtral_labeler.py.

No hand geometry, optical flow, or event-candidate proposals are used.

Transitions:
    grasp_flaps:
        stable NO -> YES in both_lips_grasped

    peel_apart:
        stable NO -> YES in package_separated

    drop_contents:
        stable NO -> YES in contents_leaving

Transition persistence and procedural gating reproduce build_sequences.py.
Ground-truth correspondences reproduce the main CRAS/RA-L evaluation.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


YES = "yes"
NO = "no"
UNCERTAIN = "uncertain"

PHASES = (
    "locate_flaps",
    "peel_apart",
    "drop_contents",
)


# ============================================================
# Visual-state transition logic
# Matches build_sequences.py
# ============================================================

def get_state(entry, state):
    states = entry.get("states", {})

    if not isinstance(states, dict):
        return UNCERTAIN

    value = str(
        states.get(state, UNCERTAIN)
    ).strip().lower()

    if value not in {
        YES,
        NO,
        UNCERTAIN,
    }:
        return UNCERTAIN

    return value


def state_series(labels, state):
    return [
        get_state(entry, state)
        for entry in labels
    ]


def initial_state_true(
    statuses,
    min_run=3,
):
    first = statuses[
        :min(
            len(statuses),
            min_run + 1,
        )
    ]

    if not first:
        return False

    yes_count = sum(
        x == YES
        for x in first
    )

    no_count = sum(
        x == NO
        for x in first
    )

    return (
        yes_count >= min_run
        and yes_count > no_count
    )


def first_stable_no_to_yes(
    statuses,
    start_idx=0,
    min_run=3,
):
    """
    Exact transition detector used by build_sequences.py.
    """

    n = len(statuses)

    start_idx = max(
        0,
        int(start_idx),
    )

    for i in range(
        start_idx,
        n,
    ):
        if statuses[i] != YES:
            continue

        pre_start = max(
            start_idx,
            i - max(
                4,
                min_run * 2,
            ),
        )

        previous = statuses[
            pre_start:i
        ]

        if sum(
            x == NO
            for x in previous
        ) < min_run:
            continue

        yes_count = 0

        confirmation_end = min(
            n,
            i + min_run + 2,
        )

        rejected = False

        for j in range(
            i,
            confirmation_end,
        ):
            value = statuses[j]

            if value == YES:
                yes_count += 1

            elif value == NO:
                rejected = True
                break

            if yes_count >= min_run:
                return i

        if rejected:
            continue

    return None


# ============================================================
# Source-frame extraction
# ============================================================

def source_frame(entry):
    """
    Recover the original source-frame integer from the label entry.

    Prefer explicit numeric frame fields where available, otherwise
    parse the final integer from frame_path.
    """

    for key in (
        "frame",
        "source_frame",
        "frame_idx",
        "frame_index",
    ):
        if key in entry:
            try:
                return int(entry[key])
            except Exception:
                pass

    path = str(
        entry.get("frame_path", "")
    )

    name = Path(path).stem

    numbers = re.findall(
        r"\d+",
        name,
    )

    if not numbers:
        raise RuntimeError(
            f"Cannot recover source frame from entry: {entry}"
        )

    return int(numbers[-1])


def pixtral_boundaries(
    labels,
    min_run=3,
):
    series = {
        state: state_series(
            labels,
            state,
        )
        for state in (
            "package_held",
            "both_lips_grasped",
            "package_separated",
            "contents_leaving",
        )
    }

    grasped_at_start = initial_state_true(
        series["both_lips_grasped"],
        min_run=min_run,
    )

    if grasped_at_start:
        grasp_idx = 0
    else:
        grasp_idx = first_stable_no_to_yes(
            series["both_lips_grasped"],
            start_idx=0,
            min_run=min_run,
        )

    if grasp_idx is not None:
        peel_idx = first_stable_no_to_yes(
            series["package_separated"],
            start_idx=grasp_idx,
            min_run=min_run,
        )
    else:
        peel_idx = None

    if peel_idx is not None:
        release_idx = first_stable_no_to_yes(
            series["contents_leaving"],
            start_idx=peel_idx,
            min_run=min_run,
        )
    else:
        release_idx = None

    frames = [
        source_frame(entry)
        for entry in labels
    ]

    return {
        "grasp":
            frames[grasp_idx]
            if grasp_idx is not None
            else None,

        "peel":
            frames[peel_idx]
            if peel_idx is not None
            else None,

        "drop":
            frames[release_idx]
            if release_idx is not None
            else None,

        "first": int(frames[0]),
        "last": int(frames[-1]),
    }


# ============================================================
# Manual GT correspondence
# ============================================================

def starts(phases, label):
    vals = []

    for r in phases.get(label, []):
        if isinstance(r, list) and len(r) >= 1:
            vals.append(int(r[0]))

    return sorted(vals)


def hand_control_start(hand):
    phases = hand.get(
        "phases",
        {},
    )

    grip_ends = [
        int(r[1])
        for r in phases.get("grip", [])
        if isinstance(r, list)
        and len(r) >= 2
    ]

    if grip_ends:
        return max(grip_ends)

    hold_starts = [
        int(r[0])
        for r in phases.get("hold", [])
        if isinstance(r, list)
        and len(r) >= 2
    ]

    if hold_starts:
        return min(hold_starts)

    return None


def gt_grasp_start(gt):
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

    return min(vals) if vals else None


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

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def summary(rows, metric):
    """
    Boundary error summary with explicit detection coverage.

    The denominator is the number of sequences with valid manual GT for the
    boundary. MAE/median/tolerance rates are calculated only where Pixtral
    produced a boundary, while coverage is reported separately.
    """

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
    print(
        f"  detected    : {len(vals)}/{len(eligible)} "
        f"({100 * len(vals) / len(eligible):.1f}%)"
        if eligible
        else "  detected    : 0/0"
    )

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
        good = sum(
            value <= tol
            for value in vals
        )

        print(
            f"  <= {tol:2d} frames : "
            f"{good}/{len(vals)} "
            f"({100 * good / len(vals):.1f}% of detected)"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cras_dir",
        default="cras_annotations",
    )

    parser.add_argument(
        "--labels_dir",
        required=True,
        help=(
            "Directory containing framewise JSON outputs "
            "from pixtral_labeler.py."
        ),
    )

    parser.add_argument(
        "--min_run",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    cras_dir = Path(args.cras_dir)
    labels_dir = Path(args.labels_dir)

    boundary_rows = []
    tiou_rows = []

    missing_files = []
    missing_boundaries = []
    invalid_gt_for_tiou = []

    annotation_files = sorted(
        cras_dir.rglob("*_annotations.json")
    )

    for cp in annotation_files:
        seq = cp.stem.replace(
            "_annotations",
            "",
        )

        gt = json.loads(
            cp.read_text()
        )

        grasp_gt = gt_grasp_start(gt)
        peel_gt = gt_pull_start(gt)
        drop_gt = gt_release_start(gt)

        matches = list(
            labels_dir.rglob(
                f"*{seq}*.json"
            )
        )

        if not matches:
            missing_files.append(seq)
            continue

        # Prefer exact sequence filename where available.
        exact = [
            p
            for p in matches
            if p.stem == seq
        ]

        lp = (
            exact[0]
            if exact
            else matches[0]
        )

        labels = json.loads(
            lp.read_text()
        )

        if not isinstance(labels, list):
            raise RuntimeError(
                f"{lp}: expected a JSON list of framewise labels"
            )

        if not labels:
            raise RuntimeError(
                f"{lp}: empty framewise label list"
            )

        pred = pixtral_boundaries(
            labels,
            min_run=args.min_run,
        )

        boundary_rows.append(
            {
                "sequence": seq,
                "grasp_gt": grasp_gt,
                "peel_gt": peel_gt,
                "drop_gt": drop_gt,
                "grasp_pred": pred["grasp"],
                "peel_pred": pred["peel"],
                "drop_pred": pred["drop"],
                "grasp_error": abs_err(
                    pred["grasp"],
                    grasp_gt,
                ),
                "peel_error": abs_err(
                    pred["peel"],
                    peel_gt,
                ),
                "drop_error": abs_err(
                    pred["drop"],
                    drop_gt,
                ),
            }
        )

        missing = [
            name
            for name in (
                "grasp",
                "peel",
                "drop",
            )
            if pred[name] is None
        ]

        if missing:
            missing_boundaries.append(
                (
                    seq,
                    missing,
                )
            )

        # --------------------------------------------------------
        # Fixed tIoU cohort:
        # determined ONLY by manual GT validity/order, not by
        # whether Pixtral detected a phase.
        # --------------------------------------------------------
        if any(
            x is None
            for x in (
                grasp_gt,
                peel_gt,
                drop_gt,
            )
        ):
            invalid_gt_for_tiou.append(seq)
            continue

        if not (
            0
            <= grasp_gt
            <= peel_gt
            <= drop_gt
            <= pred["last"]
        ):
            invalid_gt_for_tiou.append(seq)
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
                pred["last"],
            ),
        }

        # A missing predicted transition is a grounding failure,
        # not grounds for removing the sequence from the tIoU cohort.
        locate_score = 0.0
        peel_score = 0.0
        drop_score = 0.0

        if pred["grasp"] is not None:
            pred_locate = (
                pred["first"],
                max(
                    pred["first"],
                    pred["grasp"] - 1,
                ),
            )
            locate_score = tiou(
                gt_intervals["locate_flaps"],
                pred_locate,
            )

        if (
            pred["peel"] is not None
            and pred["drop"] is not None
        ):
            pred_peel = (
                pred["peel"],
                max(
                    pred["peel"],
                    pred["drop"] - 1,
                ),
            )
            peel_score = tiou(
                gt_intervals["peel_apart"],
                pred_peel,
            )

        if pred["drop"] is not None:
            pred_drop = (
                pred["drop"],
                pred["last"],
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
    print("ABLATION: PIXTRAL ONLY")
    print("=" * 72)

    print(
        "Manual annotation files:",
        len(annotation_files),
    )
    print(
        "Pixtral label files evaluated:",
        len(boundary_rows),
    )
    print(
        "Missing Pixtral label files:",
        len(missing_files),
    )

    summary(
        boundary_rows,
        "grasp_error",
    )

    summary(
        boundary_rows,
        "peel_error",
    )

    summary(
        boundary_rows,
        "drop_error",
    )

    print()
    print("-" * 72)
    print("TEMPORAL IoU")
    print("-" * 72)

    print(
        "GT-valid tIoU sequences with labels:",
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

        zeros = sum(
            value == 0.0
            for value in vals
        )

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
        print("Missing Pixtral label files:")

        for seq in missing_files:
            print(
                f"  {seq}"
            )

    if missing_boundaries:
        print()
        print("Pixtral transitions not detected:")

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
            print(
                f"  {seq}"
            )


if __name__ == "__main__":
    main()
