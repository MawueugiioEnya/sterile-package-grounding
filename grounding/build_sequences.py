#!/usr/bin/env python3

"""
Construct the procedural action sequence from concrete visual-state changes.

Expected visual progression:

package_held
    ->
both_lips_grasped
    ->
package_separated
    ->
contents_leaving

The recording may begin with package_held already true.

No Viterbi decoder.
No action classification.
No cumulative VLM event memory.
"""

import argparse
import json
import os
from collections import Counter


YES = "yes"
NO = "no"
UNCERTAIN = "uncertain"


STATES = [
    "package_held",
    "both_lips_grasped",
    "package_separated",
    "contents_leaving",
]


def get_state(entry, state):

    states = entry.get(
        "states",
        {},
    )

    if not isinstance(states, dict):
        return UNCERTAIN

    value = str(
        states.get(
            state,
            UNCERTAIN,
        )
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
    Detect a robust NO -> YES state change.

    Requirements:
    - definite NO evidence before candidate
    - at least min_run YES observations immediately after/around candidate
    - tolerate at most one UNCERTAIN
    - a definite NO inside the confirmation region rejects candidate
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

        # Need real negative evidence before claiming a transition.
        if sum(
            x == NO
            for x in previous
        ) < min_run:
            continue

        yes_count = 0
        uncertain_count = 0

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

            elif value == UNCERTAIN:
                uncertain_count += 1

            elif value == NO:
                # Allow no definite reversal while establishing boundary.
                rejected = True
                break

            if yes_count >= min_run:
                return i

        if rejected:
            continue

    return None


def frame_name(entry):

    return os.path.basename(
        entry.get(
            "frame_path",
            "",
        )
    )


def majority_condition(frames):

    values = [
        x.get(
            "condition",
            "not_applicable",
        )
        for x in frames
    ]

    valid = [
        x
        for x in values
        if x in {
            "contents_touched",
            "contents_untouched",
        }
    ]

    if not valid:
        return "not_applicable"

    return Counter(
        valid
    ).most_common(1)[0][0]


def add_step(
    steps,
    labels,
    action,
    start_idx,
    end_idx,
    source,
):

    if (
        start_idx is None
        or end_idx is None
    ):
        return

    start_idx = max(
        0,
        start_idx,
    )

    end_idx = min(
        len(labels) - 1,
        end_idx,
    )

    if start_idx > end_idx:
        return

    frames = labels[
        start_idx:
        end_idx + 1
    ]

    steps.append(
        {
            "action": action,
            "condition": majority_condition(
                frames
            ),
            "start_frame": frame_name(
                frames[0]
            ),
            "end_frame": frame_name(
                frames[-1]
            ),
            "start_s": frames[0].get(
                "timestamp_s"
            ),
            "end_s": frames[-1].get(
                "timestamp_s"
            ),
            "num_frames": len(frames),
            "boundary_source": source,
        }
    )


def describe_boundary(
    labels,
    idx,
):

    if idx is None:
        return "NOT DETECTED"

    return (
        f"sampled_index={idx}, "
        f"source_frame={frame_name(labels[idx])}"
    )


def build_sequence(
    labels_path,
    out_path,
    min_run=3,
    min_confidence=0.0,
    smooth_radius=1,
    visualize=False,
):

    # Compatibility with run_pipeline.py.
    del min_confidence
    del smooth_radius
    del visualize

    with open(labels_path, "r") as f:
        labels = json.load(f)

    if not labels:
        raise ValueError(
            f"No labels in {labels_path}"
        )

    series = {
        state: state_series(
            labels,
            state,
        )
        for state in STATES
    }

    print()
    print("=" * 70)
    print("RAW VISUAL STATES")
    print("=" * 70)

    for state in STATES:

        values = series[
            state
        ]

        print(
            f"{state:24s} "
            f"YES={sum(x == YES for x in values):3d} "
            f"NO={sum(x == NO for x in values):3d} "
            f"UNCERTAIN={sum(x == UNCERTAIN for x in values):3d}"
        )

    held_at_start = initial_state_true(
        series["package_held"],
        min_run=min_run,
    )

    grasped_at_start = initial_state_true(
        series["both_lips_grasped"],
        min_run=min_run,
    )

    # ------------------------------------------------------------
    # Pickup
    # ------------------------------------------------------------

    if held_at_start:

        pickup_idx = None

    else:

        pickup_idx = first_stable_no_to_yes(
            series["package_held"],
            start_idx=0,
            min_run=min_run,
        )

    # ------------------------------------------------------------
    # BOTH lips secured.
    # ------------------------------------------------------------

    if grasped_at_start:

        grasp_idx = 0

    else:

        grasp_search_start = (
            pickup_idx
            if pickup_idx is not None
            else 0
        )

        grasp_idx = first_stable_no_to_yes(
            series["both_lips_grasped"],
            start_idx=grasp_search_start,
            min_run=min_run,
        )

    # ------------------------------------------------------------
    # Actual material separation.
    #
    # Do not even search before two-sided grasp has been established.
    # ------------------------------------------------------------

    if grasp_idx is not None:

        peel_idx = first_stable_no_to_yes(
            series["package_separated"],
            start_idx=grasp_idx,
            min_run=min_run,
        )

    else:

        peel_idx = None

    # ------------------------------------------------------------
    # Content release.
    #
    # Do not search before actual opening.
    # ------------------------------------------------------------

    if peel_idx is not None:

        release_idx = first_stable_no_to_yes(
            series["contents_leaving"],
            start_idx=peel_idx,
            min_run=min_run,
        )

    else:

        release_idx = None

    print()
    print("=" * 70)
    print("DETECTED VISUAL-STATE BOUNDARIES")
    print("=" * 70)

    if held_at_start:
        print(
            "package_held:         "
            "ALREADY TRUE AT RECORDING START"
        )
    else:
        print(
            "package_held:         "
            + describe_boundary(
                labels,
                pickup_idx,
            )
        )

    print(
        "both_lips_grasped:     "
        + describe_boundary(
            labels,
            grasp_idx,
        )
    )

    print(
        "package_separated:     "
        + describe_boundary(
            labels,
            peel_idx,
        )
    )

    print(
        "contents_leaving:      "
        + describe_boundary(
            labels,
            release_idx,
        )
    )

    steps = []

    last_idx = len(labels) - 1

    # ------------------------------------------------------------
    # PICK UP
    # ------------------------------------------------------------

    if pickup_idx is not None:

        add_step(
            steps,
            labels,
            "pick_up_package",
            0,
            pickup_idx,
            "package_held NO->YES",
        )

    # ------------------------------------------------------------
    # LOCATE FLAPS
    #
    # For sequence 9 this should run from the beginning until around
    # source frame 60.
    # ------------------------------------------------------------

    if grasp_idx is not None:

        if held_at_start:
            locate_start = 0
        elif pickup_idx is not None:
            locate_start = pickup_idx + 1
        else:
            locate_start = None

        if (
            locate_start is not None
            and locate_start <= grasp_idx - 1
        ):

            add_step(
                steps,
                labels,
                "locate_flaps",
                locate_start,
                grasp_idx - 1,
                "before both lips are securely grasped",
            )

    # ------------------------------------------------------------
    # GRASP FLAPS
    # ------------------------------------------------------------

    if grasp_idx is not None:

        grasp_end = (
            peel_idx - 1
            if peel_idx is not None
            else last_idx
        )

        add_step(
            steps,
            labels,
            "grasp_flaps",
            grasp_idx,
            grasp_end,
            "both_lips_grasped NO->YES",
        )

    # ------------------------------------------------------------
    # PEEL
    # ------------------------------------------------------------

    if peel_idx is not None:

        peel_end = (
            release_idx - 1
            if release_idx is not None
            else last_idx
        )

        add_step(
            steps,
            labels,
            "peel_apart",
            peel_idx,
            peel_end,
            "package_separated NO->YES",
        )

    # ------------------------------------------------------------
    # RELEASE
    # ------------------------------------------------------------

    if release_idx is not None:

        add_step(
            steps,
            labels,
            "drop_contents",
            release_idx,
            last_idx,
            "contents_leaving NO->YES",
        )

    print()
    print("=" * 70)
    print("VISUAL-STATE ACTION SEQUENCE")
    print("=" * 70)

    if steps:

        print(
            " -> ".join(
                x["action"]
                for x in steps
            )
        )

    else:

        print("(no reliable sequence)")

    print()
    print("=" * 70)
    print("WARNINGS")
    print("=" * 70)

    warnings = []

    if grasp_idx is None:
        warnings.append(
            "both-lips grasp boundary not detected"
        )

    if peel_idx is None:
        warnings.append(
            "package-separation boundary not detected"
        )

    if release_idx is None:
        warnings.append(
            "contents-release boundary not detected"
        )

    if warnings:

        for warning in warnings:
            print(
                "WARNING:",
                warning,
            )

    else:
        print("None")

    os.makedirs(
        os.path.dirname(out_path) or ".",
        exist_ok=True,
    )

    with open(out_path, "w") as f:
        json.dump(
            steps,
            f,
            indent=2,
        )

    print()
    print(
        f"Wrote {len(steps)} stages to {out_path}"
    )

    return out_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--min_run",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--smooth_radius",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
    )

    args = parser.parse_args()

    build_sequence(
        labels_path=args.labels,
        out_path=args.out,
        min_run=args.min_run,
        min_confidence=args.min_confidence,
        smooth_radius=args.smooth_radius,
        visualize=args.visualize,
    )
