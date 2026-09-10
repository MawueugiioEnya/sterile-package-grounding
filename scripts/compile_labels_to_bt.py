#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


COMPILER_VERSION = "0.1"

VALID_SKILLS = [
    "pick_up_package",
    "locate_flaps",
    "grasp_flaps",
    "peel_apart",
    "drop_contents",
]

SKILL_ORDER = {
    name: i
    for i, name in enumerate(VALID_SKILLS)
}


# ============================================================
# BT BUILDING BLOCKS
# ============================================================

def condition(name):
    return {
        "type": "Condition",
        "name": name,
    }


def action(name):
    return {
        "type": "Action",
        "name": name,
    }


def selector(name, children):
    return {
        "type": "Selector",
        "name": name,
        "children": children,
    }


def sequence(name, children):
    return {
        "type": "Sequence",
        "name": name,
        "children": children,
    }


def retry(name, child, max_attempts=3):
    return {
        "type": "Retry",
        "name": name,
        "max_attempts": max_attempts,
        "child": child,
    }


# ============================================================
# ROBOT EXECUTION SUBTREES
# ============================================================

def build_package_acquisition():
    return selector(
        "EnsurePackageHeld",
        [
            condition("package_held"),
            sequence(
                "AcquirePackage",
                [
                    action("LocatePackage"),
                    action("ApproachPackage"),
                    action("CloseGripper"),
                    action("LiftAndPresent"),
                ],
            ),
        ],
    )


def build_package_held_precondition():
    """
    Used when pickup is NOT visible in the demonstration.

    We do not invent a demonstrated pickup.
    The generated execution tree therefore requires the
    package to already be held at entry.
    """
    return condition("package_held")


def build_peel_region():
    return selector(
        "EnsurePeelRegion",
        [
            condition("peel_region_found"),
            action("DetectPeelRegion"),
        ],
    )


def build_flap_acquisition():
    return selector(
        "EnsureFlapsSecured",
        [
            condition("both_flaps_secured"),
            retry(
                "RetryFlapAcquisition",
                sequence(
                    "AcquireFlaps",
                    [
                        build_peel_region(),

                        action("ApproachFlapPair"),
                        condition("flap_region_reached"),

                        action("EngageFlapPair"),
                        condition("flap_pair_engaged"),

                        action("PositionGripperAroundFlaps"),

                        action("VerifyFlapsBetweenFingers"),
                        condition("flaps_between_fingers"),

                        action("SeparateLayers"),

                        action("VerifyInitialSeparation"),
                        condition("initial_separation_observed"),

                        condition("both_flaps_secured"),
                    ],
                ),
                max_attempts=3,
            ),
        ],
    )


def build_open_package():
    return selector(
        "EnsurePackageOpen",
        [
            condition("package_open"),
            sequence(
                "OpenPackage",
                [
                    selector(
                        "EnsureFlapsStillSecured",
                        [
                            condition("both_flaps_secured"),
                            sequence(
                                "ReacquireFlaps",
                                [
                                    build_peel_region(),

                                    action("ApproachFlapPair"),
                                    condition("flap_region_reached"),

                                    action("EngageFlapPair"),
                                    condition("flap_pair_engaged"),

                                    action("PositionGripperAroundFlaps"),

                                    action("VerifyFlapsBetweenFingers"),
                                    condition("flaps_between_fingers"),

                                    action("SeparateLayers"),

                                    action("VerifyInitialSeparation"),
                                    condition("initial_separation_observed"),

                                    condition("both_flaps_secured"),
                                ],
                            ),
                        ],
                    ),
                    action("PullApart"),
                ],
            ),
        ],
    )


def build_release():
    return selector(
        "EnsureContentsReleased",
        [
            condition("contents_released"),
            action("ReleaseContents"),
        ],
    )


def build_safe_finish():
    return selector(
        "EnsureSafeFinish",
        [
            condition("robot_pose_safe"),
            action("ReturnSafe"),
        ],
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_annotations(data, path):
    annotations = data.get("annotations")

    if not isinstance(annotations, list) or not annotations:
        raise ValueError(
            f"{path.name}: missing/non-list annotations"
        )

    previous_end = None
    previous_order = -1

    for ann in annotations:
        skill = ann.get("action")
        start = ann.get("start_frame")
        end = ann.get("end_frame")

        if skill not in VALID_SKILLS:
            raise ValueError(
                f"{path.name}: unknown skill {skill!r}"
            )

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(
                f"{path.name}: invalid frame interval for {skill}"
            )

        if start > end:
            raise ValueError(
                f"{path.name}: start > end for {skill}"
            )

        order = SKILL_ORDER[skill]

        if order < previous_order:
            raise ValueError(
                f"{path.name}: invalid skill order at {skill}"
            )

        if previous_end is not None:
            if start != previous_end + 1:
                raise ValueError(
                    f"{path.name}: non-contiguous annotations: "
                    f"previous_end={previous_end}, start={start}"
                )

        previous_end = end
        previous_order = order

    frame_range = data.get("frame_range")

    if (
        isinstance(frame_range, list)
        and len(frame_range) == 2
    ):
        if annotations[0]["start_frame"] != frame_range[0]:
            raise ValueError(
                f"{path.name}: first annotation does not "
                "match frame_range start"
            )

        if annotations[-1]["end_frame"] != frame_range[1]:
            raise ValueError(
                f"{path.name}: final annotation does not "
                "match frame_range end"
            )


# ============================================================
# PROVENANCE EXTRACTION
# ============================================================

def extract_inter_burst_decisions(data):
    steps = (
        data
        .get("provenance", {})
        .get("event_resolution", {})
        .get("resolver_steps", [])
    )

    output = []

    for step in steps:
        if step.get("type") != "inter_burst_resolution":
            continue

        parsed = (
            step
            .get("result", {})
            .get("parsed", {})
        )

        output.append(
            {
                "from_burst": step.get("from_burst"),
                "to_burst": step.get("to_burst"),
                "decision": step.get("decision"),
                "release_observed":
                    parsed.get(
                        "release_from_previous_peel_observed"
                    ),
                "reacquisition_observed":
                    parsed.get("reacquisition_observed"),
                "approx_regrasp_frame":
                    parsed.get("approx_regrasp_frame"),
                "approx_peel_frame":
                    parsed.get("approx_peel_frame"),
                "semantic_action":
                    step.get("semantic_action"),
            }
        )

    return output


def extract_feature_burst_count(data):
    return (
        data
        .get("provenance", {})
        .get("event_resolution", {})
        .get("feature_burst_count")
    )


# ============================================================
# TRACE EXTRACTION
# ============================================================

def build_demonstration_trace(data):
    trace = []

    for ann in data["annotations"]:
        start = ann["start_frame"]
        end = ann["end_frame"]

        trace.append(
            {
                "skill": ann["action"],
                "start_frame": start,
                "end_frame": end,
                "duration_frames": end - start + 1,
                "source": ann.get("source"),
            }
        )

    return trace


# ============================================================
# BT COMPILATION
# ============================================================

def compile_bt(data):
    observed = [
        ann["action"]
        for ann in data["annotations"]
    ]

    observed_set = set(observed)

    children = []

    pickup_observed = bool(
        data.get("pickup_observed", False)
    )

    pickup_label_present = (
        "pick_up_package" in observed_set
    )

    # --------------------------------------------------------
    # PACKAGE ENTRY CONDITION
    # --------------------------------------------------------
    #
    # If pickup was demonstrated:
    #   compile the acquisition subtree.
    #
    # If the video starts already holding the package:
    #   do NOT invent pickup.
    #   Instead require package_held as an entry condition.
    # --------------------------------------------------------

    if pickup_observed or pickup_label_present:
        children.append(
            build_package_acquisition()
        )
    else:
        children.append(
            build_package_held_precondition()
        )

    # --------------------------------------------------------
    # LOCATE FLAPS
    # --------------------------------------------------------

    if "locate_flaps" in observed_set:
        children.append(
            build_peel_region()
        )

    # --------------------------------------------------------
    # GRASP FLAPS
    # --------------------------------------------------------

    if "grasp_flaps" in observed_set:
        children.append(
            build_flap_acquisition()
        )

    # --------------------------------------------------------
    # PEEL
    # --------------------------------------------------------
    #
    # Multiple feature bursts do NOT create repeated PullApart
    # nodes unless the semantic resolver finds a genuine
    # release/reacquisition transition.
    #
    # Current publication run has CONTINUOUS_PEEL transitions,
    # so one demonstrated peel interval compiles to one
    # reactive opening subtree.
    # --------------------------------------------------------

    if "peel_apart" in observed_set:
        children.append(
            build_open_package()
        )

    # --------------------------------------------------------
    # RELEASE CONTENTS
    # --------------------------------------------------------

    if "drop_contents" in observed_set:
        children.append(
            build_release()
        )

    # Robot-specific safe completion is an execution invariant,
    # not a demonstrated human action.
    children.append(
        build_safe_finish()
    )

    return sequence(
        "OpenSterilePackage",
        children,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compile final sterile-opening video labels "
            "into executable Behaviour Trees."
        )
    )

    parser.add_argument(
        "--labels",
        default="final_labels",
        help="Directory containing final label JSON files",
    )

    parser.add_argument(
        "--out",
        default="generated_bts",
        help="Output directory",
    )

    args = parser.parse_args()

    labels_dir = Path(args.labels)
    out_dir = Path(args.out)

    metadata_dir = (
        out_dir
        / "metadata"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        labels_dir.glob("*.json")
    )

    if not files:
        raise SystemExit(
            f"No JSON files found in {labels_dir}"
        )

    stats = Counter()

    summary_rows = []

    for path in files:
        with path.open() as f:
            data = json.load(f)

        validate_annotations(
            data,
            path,
        )

        sequence_id = (
            data.get("sequence")
            or path.stem
        )

        trace = build_demonstration_trace(
            data
        )

        decisions = (
            extract_inter_burst_decisions(
                data
            )
        )

        tree = compile_bt(
            data
        )

        # ----------------------------------------------------
        # EXECUTABLE BT
        # ----------------------------------------------------

        tree_path = (
            out_dir
            / f"{sequence_id}.json"
        )

        with tree_path.open(
            "w"
        ) as f:
            json.dump(
                tree,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # VIDEO / COMPILATION METADATA
        # ----------------------------------------------------

        metadata = {
            "compiler_version": COMPILER_VERSION,
            "sequence": sequence_id,
            "source_labels_file": str(path),
            "method": data.get("method"),
            "frame_range": data.get("frame_range"),
            "pickup_observed":
                data.get("pickup_observed"),
            "review_required":
                data.get("review_required", False),
            "demonstration_trace": trace,
            "semantic_events":
                data.get("semantic_events", []),
            "feature_burst_count":
                extract_feature_burst_count(data),
            "inter_burst_decisions":
                decisions,
            "generated_tree":
                str(tree_path),
            "compiler_notes": [
                (
                    "Feature bursts are not compiled as "
                    "separate robot actions unless semantic "
                    "resolution indicates genuine "
                    "release/reacquisition."
                ),
                (
                    "ReturnSafe is an execution invariant "
                    "added by the robot domain, not a "
                    "demonstrated human skill."
                ),
            ],
        }

        metadata_path = (
            metadata_dir
            / f"{sequence_id}.json"
        )

        with metadata_path.open(
            "w"
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        stats["compiled"] += 1

        if data.get(
            "pickup_observed",
            False,
        ):
            stats[
                "pickup_observed"
            ] += 1
        else:
            stats[
                "pickup_not_observed"
            ] += 1

        if data.get(
            "review_required",
            False,
        ):
            stats[
                "review_required"
            ] += 1

        for decision in decisions:
            stats[
                "decision:"
                + str(
                    decision.get(
                        "decision"
                    )
                )
            ] += 1

        summary_rows.append(
            {
                "sequence":
                    sequence_id,
                "pickup_observed":
                    data.get(
                        "pickup_observed"
                    ),
                "review_required":
                    data.get(
                        "review_required",
                        False,
                    ),
                "skills": [
                    x["skill"]
                    for x in trace
                ],
                "feature_burst_count":
                    extract_feature_burst_count(
                        data
                    ),
                "inter_burst_decisions": [
                    x["decision"]
                    for x in decisions
                ],
            }
        )

        print(
            f"[OK] {sequence_id} "
            f"skills="
            f"{' -> '.join(x['skill'] for x in trace)}"
        )

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    summary = {
        "compiler_version":
            COMPILER_VERSION,
        "input_directory":
            str(labels_dir),
        "output_directory":
            str(out_dir),
        "statistics":
            dict(stats),
        "sequences":
            summary_rows,
    }

    summary_path = (
        out_dir
        / "compilation_summary.json"
    )

    with summary_path.open(
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("BT COMPILATION FINISHED")
    print("=" * 80)

    print(
        "Input labels :",
        len(files),
    )

    print(
        "Compiled BTs :",
        stats["compiled"],
    )

    print(
        "Pickup seen  :",
        stats["pickup_observed"],
    )

    print(
        "Pickup absent:",
        stats["pickup_not_observed"],
    )

    print(
        "Review req.  :",
        stats["review_required"],
    )

    print()

    for key in sorted(stats):
        if key.startswith(
            "decision:"
        ):
            print(
                key,
                "=",
                stats[key],
            )

    print()
    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()
