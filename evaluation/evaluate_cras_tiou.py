#!/usr/bin/env python3

import json
import statistics
from pathlib import Path

CRAS_DIR = Path("cras_annotations")
PRED_DIR = Path("final_labels")

PHASES = [
    "locate_flaps",
    "grasp_flaps",
    "peel_apart",
    "drop_contents",
]


def phase_ranges(gt, side, phase):
    return (
        gt.get(side, {})
          .get("phases", {})
          .get(phase, [])
    )


def secure_grasp_frame(gt):
    """
    Secure bimanual grasp is defined as completion of the
    required grip establishment.
    """
    ends = []

    for side in ("left", "right"):
        grips = phase_ranges(gt, side, "grip")

        if grips:
            ends.append(
                max(int(r[1]) for r in grips if len(r) >= 2)
            )
        else:
            # Only fall back to hold if that hand has no explicit grip.
            holds = phase_ranges(gt, side, "hold")
            if holds:
                ends.append(
                    min(int(r[0]) for r in holds if len(r) >= 2)
                )

    if not ends:
        return None

    return max(ends)


def peel_start_frame(gt):
    starts = []

    for side in ("left", "right"):
        pulls = phase_ranges(gt, side, "pull")
        if pulls:
            starts.append(
                min(int(r[0]) for r in pulls if len(r) >= 2)
            )

    if not starts:
        return None

    # Both hands actively pulling when both pull annotations exist.
    if len(starts) >= 2:
        return max(starts)

    return starts[0]


def release_start_frame(gt):
    starts = []

    for side in ("left", "right"):
        releases = phase_ranges(gt, side, "release")
        if releases:
            starts.append(
                min(int(r[0]) for r in releases if len(r) >= 2)
            )

    if not starts:
        return None

    return min(starts)


def gt_intervals(gt, sequence_end):
    grasp = secure_grasp_frame(gt)
    peel = peel_start_frame(gt)
    release = release_start_frame(gt)

    if any(x is None for x in (grasp, peel, release)):
        return None

    # Ensure monotonically ordered semantic boundaries.
    if not (0 <= grasp <= peel <= release <= sequence_end):
        return None

    return {
        "locate_flaps": (0, max(0, grasp - 1)),
        "grasp_flaps": (grasp, max(grasp, peel - 1)),
        "peel_apart": (peel, max(peel, release - 1)),
        "drop_contents": (release, sequence_end),
    }


def pred_intervals(pred):
    out = {}

    for ann in pred.get("annotations", []):
        action = ann.get("action")

        if action in PHASES:
            out[action] = (
                int(ann["start_frame"]),
                int(ann["end_frame"]),
            )

    return out


def tiou(a, b):
    """
    Inclusive frame interval temporal IoU.
    """
    a0, a1 = a
    b0, b1 = b

    intersection = max(
        0,
        min(a1, b1) - max(a0, b0) + 1
    )

    union = (
        (a1 - a0 + 1)
        + (b1 - b0 + 1)
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


per_phase = {p: [] for p in PHASES}
sequence_rows = []

for cp in sorted(CRAS_DIR.rglob("*_annotations.json")):

    seq = cp.stem.replace("_annotations", "")
    pp = PRED_DIR / f"{seq}.json"

    if not pp.exists():
        continue

    gt = json.loads(cp.read_text())
    pred = json.loads(pp.read_text())

    sequence_end = int(
        pred.get("frame_range", [0, 0])[1]
    )

    gt_int = gt_intervals(gt, sequence_end)
    pred_int = pred_intervals(pred)

    if gt_int is None:
        print(f"SKIP {seq}: cannot derive ordered GT intervals")
        continue

    scores = {}

    for phase in PHASES:
        if phase in pred_int:
            score = tiou(
                gt_int[phase],
                pred_int[phase],
            )
        else:
            score = 0.0

        scores[phase] = score
        per_phase[phase].append(score)

    predicted_order = [
        ann["action"]
        for ann in pred.get("annotations", [])
        if ann.get("action") in PHASES
    ]

    # Collapse any accidental adjacent duplicates.
    collapsed = []
    for x in predicted_order:
        if not collapsed or collapsed[-1] != x:
            collapsed.append(x)

    exact_order = collapsed == PHASES

    all_present = all(
        p in pred_int
        for p in PHASES
    )

    mean_tiou = statistics.mean(scores.values())

    sequence_rows.append({
        "sequence": seq,
        "exact_order": exact_order,
        "all_present": all_present,
        "mean_tiou": mean_tiou,
        **scores,
    })


print("=" * 66)
print("TEMPORAL PHASE OVERLAP AND PROCEDURAL ORDER")
print("=" * 66)
print("Evaluated sequences:", len(sequence_rows))

for phase in PHASES:
    vals = per_phase[phase]

    print(f"\n{phase}")
    print(f"  mean tIoU   : {statistics.mean(vals):.3f}")
    print(f"  median tIoU : {statistics.median(vals):.3f}")
    print(
        f"  tIoU >= 0.50: "
        f"{sum(v >= 0.50 for v in vals)}/{len(vals)} "
        f"({100*sum(v >= 0.50 for v in vals)/len(vals):.1f}%)"
    )

exact = sum(r["exact_order"] for r in sequence_rows)
present = sum(r["all_present"] for r in sequence_rows)

print("\nPROCEDURAL STRUCTURE")
print(
    f"  All four phases present: "
    f"{present}/{len(sequence_rows)} "
    f"({100*present/len(sequence_rows):.1f}%)"
)
print(
    f"  Exact canonical order:   "
    f"{exact}/{len(sequence_rows)} "
    f"({100*exact/len(sequence_rows):.1f}%)"
)

means = [r["mean_tiou"] for r in sequence_rows]

print("\nOVERALL")
print(f"  Mean phase tIoU   : {statistics.mean(means):.3f}")
print(f"  Median phase tIoU : {statistics.median(means):.3f}")

print("\nLowest-overlap sequences:")
for r in sorted(sequence_rows, key=lambda x: x["mean_tiou"])[:6]:
    print(
        f"  {r['sequence']:16s} "
        f"mean={r['mean_tiou']:.3f} "
        f"locate={r['locate_flaps']:.3f} "
        f"grasp={r['grasp_flaps']:.3f} "
        f"peel={r['peel_apart']:.3f} "
        f"drop={r['drop_contents']:.3f}"
    )
