#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUNDING = REPO_ROOT / "grounding"

# Configured in main() from command-line arguments.
ROOT = REPO_ROOT
DATASET = None

DIRS = {}
STATUS_FILE = None
FAILURE_LOG = None


def configure_paths(dataset, workdir):
    global ROOT, DATASET, DIRS, STATUS_FILE, FAILURE_LOG

    DATASET = Path(dataset).expanduser().resolve()
    ROOT = Path(workdir).expanduser().resolve()

    DIRS = {
        "motion": ROOT / "bt_features",
        "hands": ROOT / "hand_features",
        "events": ROOT / "event_candidates",
        "refined": ROOT / "refined_boundaries",
        "start": ROOT / "start_phase",
        "final": ROOT / "final_labels",
    }

    STATUS_FILE = ROOT / "pipeline_status.json"
    FAILURE_LOG = ROOT / "pipeline_failures.log"


def needs_run(output, inputs, force=False):

    output = Path(output)

    if force or not output.exists():
        return True

    out_time = output.stat().st_mtime

    for item in inputs:

        item = Path(item)

        if item.exists() and item.stat().st_mtime > out_time:
            return True

    return False


def run_stage(
    sequence,
    stage,
    command,
    output,
    inputs,
    force=False,
):

    if not needs_run(
        output,
        inputs,
        force=force,
    ):

        print(
            f"[SKIP] {sequence} :: {stage}"
        )

        return True

    print()
    print("=" * 100)
    print(
        f"{sequence} :: {stage}"
    )
    print("=" * 100)

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    recent = []

    assert process.stdout is not None

    for line in process.stdout:

        print(
            line,
            end="",
        )

        recent.append(
            line
        )

        if len(recent) > 250:
            recent.pop(0)

    return_code = process.wait()

    if return_code != 0:

        with FAILURE_LOG.open(
            "a"
        ) as f:

            f.write(
                "\n"
                + "=" * 100
                + "\n"
            )

            f.write(
                f"{datetime.now().isoformat()} "
                f"{sequence} :: {stage}\n"
            )

            f.write(
                "COMMAND:\n"
                + " ".join(command)
                + "\n\n"
            )

            f.writelines(
                recent
            )

        print()
        print(
            f"[FAILED] {sequence} :: {stage}"
        )

        print(
            f"See {FAILURE_LOG}"
        )

        return False

    if not Path(output).exists():

        with FAILURE_LOG.open(
            "a"
        ) as f:

            f.write(
                f"\n{sequence} :: {stage}: "
                f"command succeeded but output missing: "
                f"{output}\n"
            )

        return False

    print(
        f"[OK] {sequence} :: {stage}"
    )

    return True


def load_status():

    if not STATUS_FILE.exists():
        return {}

    try:
        return json.loads(
            STATUS_FILE.read_text()
        )
    except Exception:
        return {}


def save_status(status):

    STATUS_FILE.write_text(
        json.dumps(
            status,
            indent=2,
        )
        + "\n"
    )


def final_summary(final_file):

    data = json.loads(
        final_file.read_text()
    )

    annotations = []

    for ann in data.get(
        "annotations",
        []
    ):

        annotations.append(
            {
                "action":
                    ann["action"],

                "start":
                    int(
                        ann["start_frame"]
                    ),

                "end":
                    int(
                        ann["end_frame"]
                    ),
            }
        )

    return {
        "frame_range":
            data.get(
                "frame_range"
            ),

        "pickup_observed":
            data.get(
                "pickup_observed"
            ),

        "review_required":
            data.get(
                "review_required",
                False,
            ),

        "annotations":
            annotations,
    }


def process_sequence(
    sequence,
    force=False,
):

    seq_dir = (
        DATASET
        / sequence
    )

    if not (
        seq_dir
        / "color"
    ).is_dir():

        return (
            False,
            "missing color directory",
        )

    motion = (
        DIRS["motion"]
        / f"{sequence}.json"
    )

    hand_csv = (
        DIRS["hands"]
        / f"{sequence}.csv"
    )

    hand_json = (
        DIRS["hands"]
        / f"{sequence}.json"
    )

    events = (
        DIRS["events"]
        / f"{sequence}.json"
    )

    refined = (
        DIRS["refined"]
        / f"{sequence}.json"
    )

    start = (
        DIRS["start"]
        / f"{sequence}.json"
    )

    final = (
        DIRS["final"]
        / f"{sequence}.json"
    )

    py = sys.executable

    stages = [
        (
            "motion features",

            [
                py,
                "extract_bt_features.py",
                "--seq_dir",
                str(seq_dir),
                "--out",
                str(motion),
            ],

            motion,

            [
                GROUNDING
                / "extract_bt_features.py",
            ],
        ),

        (
            "hand geometry",

            [
                py,
                "extract_hand_geometry.py",
                "--seq_dir",
                str(seq_dir),
                "--out_csv",
                str(hand_csv),
                "--out_json",
                str(hand_json),
            ],

            hand_csv,

            [
                GROUNDING
                / "extract_hand_geometry.py",
                ROOT
                / "models"
                / "hand_landmarker.task",
            ],
        ),

        (
            "event candidates",

            [
                py,
                "detect_bt_events.py",
                "--hand_csv",
                str(hand_csv),
                "--motion_json",
                str(motion),
                "--out",
                str(events),
            ],

            events,

            [
                GROUNDING
                / "detect_bt_events.py",
                hand_csv,
                motion,
            ],
        ),

        (
            "Pixtral event refinement",

            [
                py,
                "-u",
                "pixtral_refine_boundaries.py",
                "--events",
                str(events),
                "--hand_csv",
                str(hand_csv),
                "--motion_json",
                str(motion),
                "--out",
                str(refined),
            ],

            refined,

            [
                GROUNDING
                / "pixtral_refine_boundaries.py",
                events,
                hand_csv,
                motion,
            ],
        ),

        (
            "start phase",

            [
                py,
                "-u",
                "pixtral_detect_start_phase.py",
                "--hand_csv",
                str(hand_csv),
                "--refined",
                str(refined),
                "--out",
                str(start),
            ],

            start,

            [
                GROUNDING
                / "pixtral_detect_start_phase.py",
                hand_csv,
                refined,
            ],
        ),

        (
            "final labels",

            [
                py,
                "finalize_bt_labels.py",
                "--refined",
                str(refined),
                "--start_phase",
                str(start),
                "--out",
                str(final),
            ],

            final,

            [
                GROUNDING
                / "finalize_bt_labels.py",
                refined,
                start,
            ],
        ),
    ]

    for (
        stage,
        command,
        output,
        inputs,
    ) in stages:

        ok = run_stage(
            sequence,
            stage,
            command,
            output,
            inputs,
            force=force,
        )

        if not ok:

            return (
                False,
                stage,
            )

    return (
        True,
        final_summary(
            final
        ),
    )


def get_sequences():

    return sorted(
        p.name
        for p in DATASET.iterdir()
        if (
            p.is_dir()
            and (
                p
                / "color"
            ).is_dir()
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset root containing sequence directories.",
    )

    parser.add_argument(
        "--workdir",
        default=str(REPO_ROOT),
        help=(
            "Directory for generated features, candidates, labels, "
            "status and logs. Defaults to the repository root."
        ),
    )

    parser.add_argument(
        "--sequence",
        action="append",
        help=(
            "Process only this sequence. "
            "May be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun all stages even when outputs exist.",
    )

    args = parser.parse_args()

    configure_paths(
        dataset=args.dataset,
        workdir=args.workdir,
    )

    if not DATASET.is_dir():
        raise RuntimeError(
            f"Dataset directory does not exist: {DATASET}"
        )

    for directory in DIRS.values():

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    all_sequences = get_sequences()

    if args.sequence:

        requested = []

        for sequence in args.sequence:

            if sequence not in all_sequences:

                raise RuntimeError(
                    f"Unknown sequence: {sequence}"
                )

            requested.append(
                sequence
            )

        sequences = requested

    else:

        sequences = (
            all_sequences
        )

    print()
    print("=" * 100)
    print("AUTOMATIC BT LABELLING PIPELINE")
    print("=" * 100)

    print(
        f"Dataset:   {DATASET}"
    )

    print(
        f"Sequences: {len(sequences)}"
    )

    print(
        f"Python:    {sys.executable}"
    )

    print(
        f"Resume:    "
        f"{'disabled (--force)' if args.force else 'enabled'}"
    )

    status = load_status()

    successful = 0
    failed = 0

    for index, sequence in enumerate(
        sequences,
        start=1,
    ):

        print()
        print("#" * 100)

        print(
            f"[{index}/{len(sequences)}] "
            f"{sequence}"
        )

        print("#" * 100)

        try:

            ok, result = (
                process_sequence(
                    sequence,
                    force=args.force,
                )
            )

        except Exception as exc:

            ok = False
            result = (
                f"driver exception: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        if ok:

            successful += 1

            status[
                sequence
            ] = {
                "status":
                    "complete",

                "updated":
                    datetime.now().isoformat(),

                **result,
            }

            print()
            print(
                f"[COMPLETE] {sequence}"
            )

            for ann in result[
                "annotations"
            ]:

                print(
                    f"  "
                    f"{ann['start']:4d}-"
                    f"{ann['end']:4d}  "
                    f"{ann['action']}"
                )

        else:

            failed += 1

            status[
                sequence
            ] = {
                "status":
                    "failed",

                "updated":
                    datetime.now().isoformat(),

                "failure":
                    str(result),
            }

            print()
            print(
                f"[FAILED SEQUENCE] "
                f"{sequence}: {result}"
            )

        save_status(
            status
        )

    print()
    print("=" * 100)
    print("PIPELINE FINISHED")
    print("=" * 100)

    print(
        f"Complete: {successful}"
    )

    print(
        f"Failed:   {failed}"
    )

    print(
        f"Status:   {STATUS_FILE}"
    )

    if failed:

        print(
            f"Failures: {FAILURE_LOG}"
        )


if __name__ == "__main__":
    main()
