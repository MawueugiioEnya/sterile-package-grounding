#!/usr/bin/env python3

import csv
import json
import statistics
from pathlib import Path


CRAS_DIR = Path("cras_annotations")
PRED_DIR = Path("final_labels")
OUT_CSV = Path("cras_grounding_results.csv")


def starts(phases, label):
    """Return sorted start frames for a manual phase."""
    ranges = phases.get(label, [])
    vals = []

    for r in ranges:
        if isinstance(r, list) and len(r) >= 1:
            vals.append(int(r[0]))

    return sorted(vals)


def hand_control_start(hand):
    """
    Secure control is established at completion of the grip action,
    rather than at its onset.
    """
    phases = hand.get("phases", {})

    grips = phases.get("grip", [])
    grip_ends = [
        int(r[1])
        for r in grips
        if isinstance(r, list) and len(r) >= 2
    ]

    if grip_ends:
        return max(grip_ends)

    holds = phases.get("hold", [])
    hold_starts = [
        int(r[0])
        for r in holds
        if isinstance(r, list) and len(r) >= 2
    ]

    if hold_starts:
        return min(hold_starts)

    return None


def gt_grasp_start(gt):
    vals = []

    for side in ("left", "right"):
        x = hand_control_start(gt.get(side, {}))
        if x is not None:
            vals.append(x)

    # Secure bimanual control starts when the later required hand
    # has established control.
    if len(vals) >= 2:
        return max(vals)

    if len(vals) == 1:
        return vals[0]

    return None


def gt_pull_start(gt):
    vals = []

    for side in ("left", "right"):
        phases = gt.get(side, {}).get("phases", {})
        p = starts(phases, "pull")

        if p:
            vals.append(min(p))

    # If both hands pull, use the point at which both are participating.
    if len(vals) >= 2:
        return max(vals)

    # Some demonstrations use one pulling hand while the other maintains hold.
    if len(vals) == 1:
        return vals[0]

    return None


def gt_release_start(gt):
    vals = []

    for side in ("left", "right"):
        phases = gt.get(side, {}).get("phases", {})
        r = starts(phases, "release")
        if r:
            vals.append(min(r))

    if not vals:
        return None

    # Release/drop phase begins when release behaviour starts.
    return min(vals)


def first_object_event(gt, name):
    vals = (
        gt.get("object", {})
          .get("events", {})
          .get(name, [])
    )

    if not vals:
        return None

    return min(int(x) for x in vals)


def pred_start(pred, action):
    for ann in pred.get("annotations", []):
        if ann.get("action") == action:
            return int(ann["start_frame"])
    return None


def abs_err(pred, gt):
    if pred is None or gt is None:
        return None
    return abs(pred - gt)


rows = []

for cp in sorted(CRAS_DIR.rglob("*_annotations.json")):

    seq = cp.stem.replace("_annotations", "")
    pp = PRED_DIR / f"{seq}.json"

    if not pp.exists():
        continue

    gt = json.loads(cp.read_text())
    pred = json.loads(pp.read_text())

    grasp_gt = gt_grasp_start(gt)
    peel_gt = gt_pull_start(gt)
    drop_gt = gt_release_start(gt)
    seal_gt = first_object_event(gt, "seal_break")

    grasp_pred = pred_start(pred, "grasp_flaps")
    peel_pred = pred_start(pred, "peel_apart")
    drop_pred = pred_start(pred, "drop_contents")

    row = {
        "sequence": seq,

        "gt_grasp": grasp_gt,
        "pred_grasp": grasp_pred,
        "grasp_error": abs_err(grasp_pred, grasp_gt),

        "gt_peel": peel_gt,
        "pred_peel": peel_pred,
        "peel_error": abs_err(peel_pred, peel_gt),

        "gt_drop": drop_gt,
        "pred_drop": drop_pred,
        "drop_error": abs_err(drop_pred, drop_gt),

        "gt_seal_break": seal_gt,
        "peel_to_seal_error":
            abs_err(peel_pred, seal_gt),

        "pickup_predicted":
            int(pred_start(pred, "pick_up_package") is not None),
    }

    rows.append(row)


fields = list(rows[0].keys())

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def summary(metric):
    vals = [
        int(r[metric])
        for r in rows
        if r[metric] is not None
    ]

    if not vals:
        return

    print(f"\n{metric}")
    print(f"  n      : {len(vals)}")
    print(f"  MAE    : {statistics.mean(vals):.2f} frames")
    print(f"  median : {statistics.median(vals):.2f} frames")
    print(f"  min    : {min(vals)}")
    print(f"  max    : {max(vals)}")

    for tol in (2, 5, 10):
        good = sum(v <= tol for v in vals)
        print(
            f"  <= {tol:2d} frames: "
            f"{good}/{len(vals)} "
            f"({100*good/len(vals):.1f}%)"
        )


print("=" * 60)
print("CRAS MANUAL vs AUTOMATIC TEMPORAL GROUNDING")
print("=" * 60)
print("Evaluated sequences:", len(rows))

summary("grasp_error")
summary("peel_error")
summary("drop_error")

print("\nSaved:", OUT_CSV)

print("\nPer-sequence:")
for r in rows:
    print(
        f"{r['sequence']:16s} "
        f"grasp={str(r['grasp_error']):>4s}  "
        f"peel={str(r['peel_error']):>4s}  "
        f"drop={str(r['drop_error']):>4s}"
    )
