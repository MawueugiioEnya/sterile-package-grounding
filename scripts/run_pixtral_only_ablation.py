#!/usr/bin/env python3

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


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
        default="pixtral_only_labels",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--temporal_window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--load_4bit",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    cras_dir = Path(args.cras_dir)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequences = sorted(
        p.stem.replace(
            "_annotations",
            "",
        )
        for p in cras_dir.rglob(
            "*_annotations.json"
        )
    )

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

    print("=" * 72)
    print("PIXTRAL-ONLY ABLATION")
    print("=" * 72)
    print(
        "Manual sequences:",
        len(sequences),
    )
    print(
        "Available:",
        len(available),
    )
    print(
        "Missing:",
        len(missing),
        missing,
    )

    for index, seq in enumerate(
        available,
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

        if (
            out_path.exists()
            and not args.overwrite
        ):
            print(
                f"[{index}/{len(available)}] "
                f"SKIP {seq}: output exists"
            )
            continue

        data = json.loads(
            feature_path.read_text()
        )

        features = data.get(
            "features",
            [],
        )

        if not features:
            print(
                f"[{index}/{len(available)}] "
                f"SKIP {seq}: no features"
            )
            continue

        manifest = []

        for item in features:
            if (
                "frame" not in item
                or "frame_path" not in item
            ):
                continue

            manifest.append(
                {
                    "frame":
                        int(item["frame"]),
                    "frame_path":
                        str(item["frame_path"]),
                }
            )

        if not manifest:
            print(
                f"[{index}/{len(available)}] "
                f"SKIP {seq}: empty RGB manifest"
            )
            continue

        print()
        print("=" * 72)
        print(
            f"[{index}/{len(available)}] "
            f"{seq}"
        )
        print(
            f"frames: {len(manifest)}"
        )
        print("=" * 72)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{seq}_",
            delete=False,
        ) as tmp:
            json.dump(
                manifest,
                tmp,
                indent=2,
            )

            manifest_path = Path(
                tmp.name
            )

        cmd = [
            "python",
            "pixtral_labeler.py",
            "--manifest",
            str(manifest_path),
            "--out",
            str(out_path),
            "--batch_size",
            str(args.batch_size),
            "--temporal_window",
            str(args.temporal_window),
        ]

        if args.load_4bit:
            cmd.append(
                "--load_4bit"
            )

        try:
            subprocess.run(
                cmd,
                check=True,
            )

        finally:
            try:
                manifest_path.unlink()
            except FileNotFoundError:
                pass

    print()
    print("=" * 72)
    print("PIXTRAL-ONLY RUN COMPLETE")
    print("=" * 72)
    print(
        "Outputs:",
        out_dir,
    )


if __name__ == "__main__":
    main()
