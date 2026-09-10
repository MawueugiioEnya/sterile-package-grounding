#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


METHODS = [
    "geometry_motion",
    "pixtral_only",
    "hybrid",
]

EVENT_ORDER = [
    "grasp_flaps",
    "peel_apart",
    "drop_contents",
]


def load_sequence_info(sped_root, sequence):
    path = sped_root / sequence / "sequence_info.json"

    if not path.exists():
        raise FileNotFoundError(
            f"{sequence}: missing {path}"
        )

    with path.open() as f:
        info = json.load(f)

    total_frames = int(info["total_frames"])

    if total_frames <= 0:
        raise ValueError(
            f"{sequence}: invalid total_frames={total_frames}"
        )

    return {
        "total_frames": total_frames,
        "last_frame": total_frames - 1,
        "object_name": info.get("object_name"),
    }


def build_annotations(sequence, method, events, last_frame):
    """
    Convert method-specific semantic event onsets into the
    contiguous interval representation expected by
    compile_labels_to_bt.py.

    Grounding predictions are NOT changed.

    If grasp_flaps and peel_apart occur on the same frame,
    grasp_flaps is assigned the immediately preceding frame
    for BT interval construction only. The original grounding
    frame remains preserved in grounding_events and
    source_frame_original.
    """

    present = []

    for event in EVENT_ORDER:
        frame = events.get(event)

        if frame is None:
            continue

        frame = int(frame)

        present.append(
            {
                "action": event,
                "start_frame": frame,
                "source_frame_original": frame,
                "source": method,
            }
        )

    if not present:
        raise ValueError(
            f"{sequence}/{method}: no semantic events present"
        )

    adjustments = []

    # --------------------------------------------------------
    # SAME-FRAME GRASP -> PEEL REPRESENTATION FIX
    # --------------------------------------------------------
    #
    # Mirrors the established logic in finalize_bt_labels.py.
    # It does NOT modify the grounding prediction itself.
    # --------------------------------------------------------

    for i in range(len(present) - 1):
        current = present[i]
        nxt = present[i + 1]

        if (
            current["action"] == "grasp_flaps"
            and nxt["action"] == "peel_apart"
            and current["start_frame"] == nxt["start_frame"]
        ):
            original_frame = int(
                current["start_frame"]
            )

            compiler_frame = original_frame - 1

            if compiler_frame < 0:
                raise ValueError(
                    f"{sequence}/{method}: "
                    f"same-frame grasp/peel at frame "
                    f"{original_frame}, but no preceding "
                    f"frame is available"
                )

            current["start_frame"] = compiler_frame

            current["source"] = (
                f"{method}:same_frame_grasp_adjustment"
            )

            adjustments.append(
                {
                    "event": "grasp_flaps",
                    "original_frame": original_frame,
                    "compiler_frame": compiler_frame,
                    "peel_frame": int(
                        nxt["start_frame"]
                    ),
                    "reason": "same_frame_grasp_peel",
                }
            )

    # --------------------------------------------------------
    # VALIDATE ORDER AFTER REPRESENTATION FIX
    # --------------------------------------------------------

    frames = [
        int(item["start_frame"])
        for item in present
    ]

    for previous, current in zip(
        frames,
        frames[1:],
    ):
        if current <= previous:
            raise ValueError(
                f"{sequence}/{method}: "
                f"unresolved event ordering {frames}. "
                f"Grounding predictions were not altered."
            )

    for item in present:
        frame = int(item["start_frame"])

        if frame < 0 or frame > last_frame:
            raise ValueError(
                f"{sequence}/{method}: "
                f"{item['action']} compiler frame "
                f"{frame} outside valid range "
                f"0..{last_frame}"
            )

    # --------------------------------------------------------
    # BUILD CONTIGUOUS INTERVALS FOR EXISTING BT COMPILER
    # --------------------------------------------------------

    annotations = []

    for i, item in enumerate(present):
        start = int(
            item["start_frame"]
        )

        if i + 1 < len(present):
            end = (
                int(
                    present[i + 1]["start_frame"]
                )
                - 1
            )
        else:
            end = int(last_frame)

        if end < start:
            raise ValueError(
                f"{sequence}/{method}: "
                f"invalid interval for "
                f"{item['action']}: "
                f"{start}..{end}"
            )

        annotations.append(
            {
                **item,
                "start_frame": start,
                "end_frame": end,
            }
        )

    return annotations, adjustments


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert matched three-method grounding events "
            "into the common label schema used by the "
            "reactive Behaviour Tree compiler."
        )
    )

    parser.add_argument(
        "--cohort",
        required=True,
        help="matched_grounding_cohort.json",
    )

    parser.add_argument(
        "--sped",
        required=True,
        help="SPED dataset root",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output root for method-specific BT inputs",
    )

    args = parser.parse_args()

    cohort_path = Path(
        args.cohort
    ).expanduser().resolve()

    sped_root = Path(
        args.sped
    ).expanduser().resolve()

    out_root = Path(
        args.out
    ).expanduser().resolve()

    with cohort_path.open() as f:
        cohort = json.load(f)

    sequences = cohort["sequences"]

    print("=" * 80)
    print("BUILD METHOD-SPECIFIC BT INPUTS")
    print("=" * 80)
    print("Cohort :", cohort_path)
    print("SPED   :", sped_root)
    print("Output :", out_root)
    print("N      :", len(sequences))
    print()

    counts = {
        method: {
            "written": 0,
            "missing_grasp": 0,
            "missing_peel": 0,
            "missing_release": 0,
            "frame_adjustments": 0,
        }
        for method in METHODS
    }

    for method in METHODS:
        (
            out_root
            / method
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    for record in sequences:
        sequence = record["sequence"]

        seq_info = load_sequence_info(
            sped_root,
            sequence,
        )

        last_frame = seq_info[
            "last_frame"
        ]

        object_name = seq_info[
            "object_name"
        ]

        for method in METHODS:
            events = record[
                "methods"
            ][method]

            annotations, adjustments = (
                build_annotations(
                    sequence,
                    method,
                    events,
                    last_frame,
                )
            )

            if events.get(
                "grasp_flaps"
            ) is None:
                counts[
                    method
                ][
                    "missing_grasp"
                ] += 1

            if events.get(
                "peel_apart"
            ) is None:
                counts[
                    method
                ][
                    "missing_peel"
                ] += 1

            if events.get(
                "drop_contents"
            ) is None:
                counts[
                    method
                ][
                    "missing_release"
                ] += 1

            counts[
                method
            ][
                "frame_adjustments"
            ] += len(
                adjustments
            )

            output = {
                "sequence": sequence,
                "method": method,
                "object_name": object_name,

                # Pickup is outside the three-event grounding
                # ablation. The BT therefore starts from the
                # package-held execution precondition.
                "pickup_observed": False,

                "annotations": annotations,

                # Preserve the unmodified predictions used in
                # the grounding ablation.
                "grounding_events": {
                    event: events.get(event)
                    for event in EVENT_ORDER
                },

                # Representation-only adjustments required by
                # the interval-based BT compiler.
                "compiler_frame_adjustments":
                    adjustments,

                "adapter_notes": {
                    "semantic_events_not_invented":
                        True,

                    "original_grounding_frames_preserved":
                        True,

                    "same_frame_adjustment_is_bt_representation_only":
                        True,

                    "robot_coordinates_from_live_perception":
                        True,

                    "bt_execution_policy_shared_across_methods":
                        True,
                },
            }

            path = (
                out_root
                / method
                / f"{sequence}.json"
            )

            path.write_text(
                json.dumps(
                    output,
                    indent=2,
                )
                + "\n"
            )

            counts[
                method
            ][
                "written"
            ] += 1

            skills = [
                item["action"]
                for item in annotations
            ]

            adjustment_text = ""

            if adjustments:
                adjustment_text = (
                    "  [BT frame adjustment: "
                    + ", ".join(
                        f"{x['event']} "
                        f"{x['original_frame']}"
                        f"->{x['compiler_frame']}"
                        for x in adjustments
                    )
                    + "]"
                )

            print(
                f"{sequence:18s} "
                f"{method:16s} "
                f"{str(object_name or 'unknown'):20s} "
                f"{' -> '.join(skills)}"
                f"{adjustment_text}"
            )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for method in METHODS:
        c = counts[method]

        print(
            f"{method:16s} "
            f"written={c['written']:2d} "
            f"missing_grasp={c['missing_grasp']:2d} "
            f"missing_peel={c['missing_peel']:2d} "
            f"missing_release={c['missing_release']:2d} "
            f"adjustments={c['frame_adjustments']:2d}"
        )

    summary = {
        "cohort": str(cohort_path),
        "n_sequences": len(sequences),
        "methods": METHODS,
        "counts": counts,
        "notes": {
            "same_frame_grasp_peel":
                (
                    "When grasp_flaps and peel_apart share "
                    "the same predicted onset, grasp_flaps "
                    "is moved one frame earlier for BT "
                    "interval construction only. Original "
                    "grounding predictions are preserved."
                ),
            "common_compiler":
                (
                    "All methods are intended for compilation "
                    "with the same reactive BT compiler."
                ),
        },
    }

    summary_path = (
        out_root
        / "bt_input_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Summary saved:",
        summary_path,
    )


if __name__ == "__main__":
    main()
