#!/usr/bin/env python3
"""
Run the RGB-only Pixtral event-grounding ablation while loading Pixtral once.

The bt_features JSON files are used ONLY as a source of:
    - frame
    - frame_path

No geometry, motion, event-candidate, or feature-table values are passed to
Pixtral.

Outputs are written after every sequence, so the run is resumable.
"""

import argparse
import json
import sys
from pathlib import Path

# Make grounding/ importable from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
GROUNDING_DIR = REPO_ROOT / "grounding"
sys.path.insert(0, str(GROUNDING_DIR))

from pixtral_labeler import load_model
from pixtral_only_event_grounder import (
    EVENTS,
    collect_coarse_candidates,
    representative_candidate,
    refine_event,
    enforce_order,
)


def make_rgb_manifest(feature_path):
    data = json.loads(feature_path.read_text())
    features = data.get("features", [])

    manifest = []

    for item in features:
        if (
            isinstance(item, dict)
            and "frame" in item
            and "frame_path" in item
        ):
            manifest.append(
                {
                    "frame": int(item["frame"]),
                    "frame_path": str(item["frame_path"]),
                }
            )

    manifest.sort(
        key=lambda x: int(x["frame"])
    )

    return manifest


def process_sequence(
    model,
    processor,
    seq,
    manifest,
    out_path,
    args,
):
    print()
    print("=" * 72)
    print(f"{seq}: RGB-ONLY EVENT GROUNDING")
    print("=" * 72)
    print("Frames:", len(manifest))
    print("First :", manifest[0]["frame"])
    print("Last  :", manifest[-1]["frame"])

    candidates, coarse_diagnostics = collect_coarse_candidates(
        model,
        processor,
        manifest,
        window_size=args.window_size,
        stride=args.stride,
        max_images=args.coarse_images,
    )

    coarse_consensus = {
        event: representative_candidate(
            candidates[event]
        )
        for event in EVENTS
    }

    refined = {}
    refinement_diagnostics = {}

    for event in EVENTS:
        boundary, diagnostic = refine_event(
            model,
            processor,
            manifest,
            event,
            coarse_consensus[event],
            radius=args.refine_radius,
            max_images=args.refine_images,
        )

        refined[event] = boundary
        refinement_diagnostics[event] = diagnostic

    final = enforce_order(refined)

    output = {
        "sequence": seq,
        "method": "pixtral_only_rgb_event_grounding",
        "model": "mistral-community/pixtral-12b",
        "inputs": "rgb_only",
        "uses_hand_geometry": False,
        "uses_optical_flow": False,
        "uses_event_candidates": False,
        "uses_feature_table": False,
        "first_frame": int(manifest[0]["frame"]),
        "last_frame": int(manifest[-1]["frame"]),
        "boundaries": final,
        "coarse_consensus": coarse_consensus,
        "coarse_candidates": candidates,
        "diagnostics": {
            "coarse_windows": coarse_diagnostics,
            "refinement": refinement_diagnostics,
        },
        "parameters": {
            "window_size": args.window_size,
            "stride": args.stride,
            "coarse_images": args.coarse_images,
            "refine_radius": args.refine_radius,
            "refine_images": args.refine_images,
            "do_sample": False,
        },
    }

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print("FINAL:", final)
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cras_dir",
        default="cras_annotations",
    )

    parser.add_argument(
        "--features_dir",
        default="bt_features",
    )

    parser.add_argument(
        "--out_dir",
        default="pixtral_only_events",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=28,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--coarse_images",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--refine_radius",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--refine_images",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--load_4bit",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--sequence",
        action="append",
        help=(
            "Optional sequence to process. "
            "May be supplied multiple times."
        ),
    )

    args = parser.parse_args()

    cras_dir = Path(args.cras_dir)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)

    sequences = sorted(
        p.stem.replace(
            "_annotations",
            "",
        )
        for p in cras_dir.rglob(
            "*_annotations.json"
        )
    )

    if args.sequence:
        requested = set(args.sequence)
        sequences = [
            seq
            for seq in sequences
            if seq in requested
        ]

    available = [
        seq
        for seq in sequences
        if (
            features_dir
            / f"{seq}.json"
        ).exists()
    ]

    missing = [
        seq
        for seq in sequences
        if not (
            features_dir
            / f"{seq}.json"
        ).exists()
    ]

    pending = [
        seq
        for seq in available
        if (
            args.overwrite
            or not (
                out_dir
                / f"{seq}.json"
            ).exists()
        )
    ]

    print("=" * 72)
    print("PIXTRAL-ONLY RGB EVENT ABLATION")
    print("=" * 72)
    print("Manual sequences :", len(sequences))
    print("Available        :", len(available))
    print("Missing          :", len(missing), missing)
    print("Already complete :", len(available) - len(pending))
    print("Pending          :", len(pending))

    if not pending:
        print("Nothing to run.")
        return

    print()
    print("Loading Pixtral ONCE...")
    model, processor = load_model(
        load_4bit=args.load_4bit
    )

    print()
    print("Model loaded. Processing sequences.")

    for index, seq in enumerate(
        pending,
        start=1,
    ):
        feature_path = (
            features_dir
            / f"{seq}.json"
        )

        out_path = (
            out_dir
            / f"{seq}.json"
        )

        manifest = make_rgb_manifest(
            feature_path
        )

        if not manifest:
            print(
                f"[{index}/{len(pending)}] "
                f"SKIP {seq}: empty RGB manifest"
            )
            continue

        print(
            f"\n[{index}/{len(pending)}] "
            f"{seq}"
        )

        try:
            process_sequence(
                model,
                processor,
                seq,
                manifest,
                out_path,
                args,
            )

        except KeyboardInterrupt:
            print("\nInterrupted. Completed outputs are preserved.")
            raise

        except Exception as exc:
            print(
                f"ERROR {seq}: "
                f"{type(exc).__name__}: {exc}"
            )
            # Continue to the next sequence so one failure does not
            # waste the entire GPU allocation.
            continue

    print()
    print("=" * 72)
    print("PIXTRAL-ONLY RGB EVENT RUN COMPLETE")
    print("=" * 72)
    print("Outputs:", out_dir)


if __name__ == "__main__":
    main()
