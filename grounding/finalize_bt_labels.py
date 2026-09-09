#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


ALLOWED_ACTIONS = {
    "pick_up_package",
    "locate_flaps",
    "grasp_flaps",
    "peel_apart",
    "drop_contents",
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--refined",
        required=True,
    )

    parser.add_argument(
        "--start_phase",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    refined = json.loads(
        Path(args.refined).read_text()
    )

    start = json.loads(
        Path(args.start_phase).read_text()
    )

    sequence = refined["sequence"]

    if start["sequence"] != sequence:
        raise RuntimeError(
            "Sequence mismatch between refined and start-phase files."
        )

    # ============================================================
    # FRAME RANGE
    # ============================================================

    if "frame_range" in refined:

        first_frame = int(
            refined["frame_range"][0]
        )

        last_frame = int(
            refined["frame_range"][1]
        )

    else:

        first_frame = int(
            start["first_frame"]
        )

        candidate_file = (
            Path("event_candidates")
            / f"{sequence}.json"
        )

        if not candidate_file.exists():

            raise RuntimeError(
                f"Cannot determine last frame for {sequence}"
            )

        candidate_data = json.loads(
            candidate_file.read_text()
        )

        last_frame = int(
            candidate_data["frame_range"][1]
        )

    pickup_observed = bool(
        start["pickup_observed"]
    )

    locate_start = int(
        start["locate_flaps_start"]
    )

    # ============================================================
    # SEMANTIC EVENTS
    # ============================================================

    raw_events = refined.get(
        "semantic_events",
        [],
    )

    if not raw_events:
        raise RuntimeError(
            f"{sequence}: refined file has no semantic_events"
        )

    semantic_events = []

    for event in raw_events:

        action = str(
            event["action"]
        )

        frame = int(
            event["start_frame"]
        )

        if action not in ALLOWED_ACTIONS:

            raise RuntimeError(
                f"{sequence}: unknown action {action}"
            )

        semantic_events.append(
            {
                **event,
                "action": action,
                "start_frame": frame,
            }
        )

    semantic_events = sorted(
        semantic_events,
        key=lambda x:
            x["start_frame"],
    )


    # ============================================================
    # SAME-FRAME GRASP -> PEEL RESOLUTION
    #
    # Pixtral can occasionally place secure acquisition and the
    # first active peel on the same frame.
    #
    # Keep the peel onset unchanged and assign the immediately
    # preceding available frame to grasp_flaps so every BT action
    # has a non-zero temporal interval.
    # ============================================================

    same_frame_adjustments = []

    for i in range(
        len(semantic_events) - 1
    ):

        current = semantic_events[i]
        nxt = semantic_events[i + 1]

        if (
            current["action"] == "grasp_flaps"
            and nxt["action"] == "peel_apart"
            and int(current["start_frame"])
            == int(nxt["start_frame"])
        ):

            peel_frame = int(
                nxt["start_frame"]
            )

            if i == 0:

                lower_bound = (
                    locate_start + 1
                )

            else:

                lower_bound = (
                    int(
                        semantic_events[
                            i - 1
                        ]["start_frame"]
                    )
                    + 1
                )

            adjusted_grasp = (
                peel_frame - 1
            )

            if adjusted_grasp < lower_bound:

                raise RuntimeError(
                    f"{sequence}: same-frame grasp/peel "
                    f"at {peel_frame}, but no preceding "
                    f"frame is available for grasp_flaps"
                )

            old_frame = int(
                current["start_frame"]
            )

            current[
                "start_frame"
            ] = adjusted_grasp

            current[
                "source"
            ] = (
                "same_frame_grasp_adjustment"
            )

            same_frame_adjustments.append(
                {
                    "old_grasp_frame":
                        old_frame,

                    "new_grasp_frame":
                        adjusted_grasp,

                    "peel_frame":
                        peel_frame,
                }
            )

    if same_frame_adjustments:

        print()
        print(
            "SAME-FRAME GRASP/PEEL ADJUSTMENT"
        )

        for item in same_frame_adjustments:

            print(
                f"  grasp "
                f"{item['old_grasp_frame']} -> "
                f"{item['new_grasp_frame']}; "
                f"peel remains "
                f"{item['peel_frame']}"
            )

    # ============================================================
    # REMOVE EXACT DUPLICATES / COLLAPSE ADJACENT SAME ACTIONS
    # ============================================================

    cleaned = []

    for event in semantic_events:

        if cleaned:

            previous = cleaned[-1]

            if (
                event["action"]
                == previous["action"]
            ):

                # A later feature event that resolves to the same
                # semantic action does not create a new BT phase.
                continue

        cleaned.append(
            event
        )

    semantic_events = cleaned

    # ============================================================
    # TERMINAL DROP
    # ============================================================

    drop_events = [
        x
        for x in semantic_events
        if x["action"] == "drop_contents"
    ]

    if len(drop_events) != 1:

        raise RuntimeError(
            f"{sequence}: expected exactly one terminal "
            f"drop_contents event, found {len(drop_events)}"
        )

    drop_frame = int(
        drop_events[0]["start_frame"]
    )

    # Nothing after terminal drop is a valid BT event.
    semantic_events = [
        x
        for x in semantic_events
        if x["start_frame"] <= drop_frame
    ]

    # ============================================================
    # BASIC TEMPORAL VALIDATION
    # ============================================================

    previous_frame = None

    for event in semantic_events:

        frame = event["start_frame"]

        if (
            previous_frame is not None
            and frame <= previous_frame
        ):

            raise RuntimeError(
                f"{sequence}: non-increasing semantic event order"
            )

        previous_frame = frame

    if semantic_events[0]["action"] != "grasp_flaps":

        raise RuntimeError(
            f"{sequence}: first post-locate semantic event "
            f"must be grasp_flaps, got "
            f"{semantic_events[0]['action']}"
        )

    if semantic_events[-1]["action"] != "drop_contents":

        raise RuntimeError(
            f"{sequence}: final semantic event must be drop_contents"
        )

    if semantic_events[0]["start_frame"] <= locate_start:

        raise RuntimeError(
            f"{sequence}: grasp must occur after locate_flaps begins"
        )

    # ============================================================
    # VALID BT TRANSITIONS
    # ============================================================

    allowed_transitions = {
        ("grasp_flaps", "peel_apart"),
        ("peel_apart", "grasp_flaps"),
        ("peel_apart", "drop_contents"),
    }

    for a, b in zip(
        semantic_events[:-1],
        semantic_events[1:],
    ):

        transition = (
            a["action"],
            b["action"],
        )

        if transition not in allowed_transitions:

            raise RuntimeError(
                f"{sequence}: invalid BT transition "
                f"{transition[0]} -> {transition[1]}"
            )

    # ============================================================
    # BUILD FULL TEMPORAL EVENT LIST
    # ============================================================

    timeline_events = []

    if pickup_observed:

        if locate_start <= first_frame:

            raise RuntimeError(
                f"{sequence}: pickup marked present but "
                f"locate_flaps begins at first frame"
            )

        timeline_events.append(
            {
                "action":
                    "pick_up_package",

                "start_frame":
                    first_frame,

                "source":
                    "pixtral_start_phase",
            }
        )

    else:

        if locate_start != first_frame:

            raise RuntimeError(
                f"{sequence}: pickup absent but locate_flaps "
                f"does not begin at first frame"
            )

    timeline_events.append(
        {
            "action":
                "locate_flaps",

            "start_frame":
                locate_start,

            "source":
                "start_phase",
        }
    )

    timeline_events.extend(
        semantic_events
    )

    timeline_events = sorted(
        timeline_events,
        key=lambda x:
            x["start_frame"],
    )

    # ============================================================
    # BUILD CONTIGUOUS INTERVALS
    # ============================================================

    annotations = []

    for i, event in enumerate(
        timeline_events
    ):

        start_frame = int(
            event["start_frame"]
        )

        if i + 1 < len(
            timeline_events
        ):

            end_frame = int(
                timeline_events[
                    i + 1
                ]["start_frame"]
            ) - 1

        else:

            end_frame = (
                last_frame
            )

        if end_frame < start_frame:

            raise RuntimeError(
                f"{sequence}: invalid interval for "
                f"{event['action']}: "
                f"{start_frame}-{end_frame}"
            )

        annotations.append(
            {
                "action":
                    event["action"],

                "start_frame":
                    start_frame,

                "end_frame":
                    end_frame,

                "source":
                    event.get(
                        "source",
                        "unknown",
                    ),
            }
        )

    # ============================================================
    # PER-FRAME LABELS
    # ============================================================

    frame_labels = []

    for annotation in annotations:

        for frame in range(
            annotation["start_frame"],
            annotation["end_frame"] + 1,
        ):

            frame_labels.append(
                {
                    "frame":
                        frame,

                    "action":
                        annotation[
                            "action"
                        ],
                }
            )

    expected_count = (
        last_frame
        - first_frame
        + 1
    )

    if len(frame_labels) != expected_count:

        raise RuntimeError(
            f"{sequence}: labelled "
            f"{len(frame_labels)} frames, "
            f"expected {expected_count}"
        )

    if frame_labels[0]["frame"] != first_frame:

        raise RuntimeError(
            f"{sequence}: first frame not labelled"
        )

    if frame_labels[-1]["frame"] != last_frame:

        raise RuntimeError(
            f"{sequence}: last frame not labelled"
        )

    # ============================================================
    # OUTPUT
    # ============================================================

    output = {
        "sequence":
            sequence,

        "method":
            refined.get(
                "method",
                "automatic_bt_pipeline",
            ),

        "frame_range": [
            first_frame,
            last_frame,
        ],

        "pickup_observed":
            pickup_observed,

        "annotations":
            annotations,

        "frame_labels":
            frame_labels,

        "semantic_events":
            semantic_events,

        "review_required":
            bool(
                refined.get(
                    "review_required",
                    False,
                )
            ),

        "provenance": {
            "start_phase":
                start,

            "event_resolution":
                refined,
        },
    }

    out_file = Path(
        args.out
    )

    out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_file.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n"
    )

    # ============================================================
    # PRINT
    # ============================================================

    print()
    print("=" * 90)
    print(
        f"FINAL BT LABELS: {sequence}"
    )
    print("=" * 90)

    for annotation in annotations:

        print(
            f"{annotation['start_frame']:4d}-"
            f"{annotation['end_frame']:4d}  "
            f"{annotation['action']}"
        )

    print()
    print(
        f"Total labelled frames: "
        f"{len(frame_labels)}"
    )

    print(
        f"Review required: "
        f"{output['review_required']}"
    )

    print(
        f"Wrote {out_file}"
    )


if __name__ == "__main__":
    main()
