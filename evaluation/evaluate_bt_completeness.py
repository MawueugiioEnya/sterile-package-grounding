#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


METHODS = [
    "geometry_motion",
    "pixtral_only",
    "hybrid",
]

REQUIRED_NODES = {
    "grasp": "EnsureFlapsSecured",
    "open": "EnsurePackageOpen",
    "release": "EnsureContentsReleased",
}


def collect_names(node, names):
    if isinstance(node, dict):
        name = node.get("name")
        if name:
            names.add(name)

        for value in node.values():
            collect_names(value, names)

    elif isinstance(node, list):
        for value in node:
            collect_names(value, names)


def inspect_tree(path):
    data = json.loads(path.read_text())

    names = set()
    collect_names(data, names)

    grasp = REQUIRED_NODES["grasp"] in names
    open_ = REQUIRED_NODES["open"] in names
    release = REQUIRED_NODES["release"] in names

    return {
        "grasp": grasp,
        "open": open_,
        "release": release,
        "complete": grasp and open_ and release,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="generated_bts_ablation",
        help="Root directory containing method-specific BT folders",
    )

    parser.add_argument(
        "--out-json",
        default="evaluation/bt_completeness_summary.json",
    )

    parser.add_argument(
        "--out-csv",
        default="evaluation/bt_completeness_per_sequence.csv",
    )

    args = parser.parse_args()

    root = Path(args.root)

    rows = []
    summary = {}

    for method in METHODS:
        method_dir = root / method

        files = sorted(
            method_dir.glob("P*_SEQ_*.json")
        )

        if not files:
            raise SystemExit(
                f"No BT files found for {method}: {method_dir}"
            )

        method_rows = []

        for path in files:
            result = inspect_tree(path)

            row = {
                "sequence": path.stem,
                "method": method,
                **result,
            }

            rows.append(row)
            method_rows.append(row)

        n = len(method_rows)

        grasp_n = sum(
            int(x["grasp"])
            for x in method_rows
        )

        open_n = sum(
            int(x["open"])
            for x in method_rows
        )

        release_n = sum(
            int(x["release"])
            for x in method_rows
        )

        complete_n = sum(
            int(x["complete"])
            for x in method_rows
        )

        incomplete = [
            x["sequence"]
            for x in method_rows
            if not x["complete"]
        ]

        summary[method] = {
            "n_sequences": n,

            "grasp_subtree": {
                "count": grasp_n,
                "percent": 100.0 * grasp_n / n,
            },

            "open_subtree": {
                "count": open_n,
                "percent": 100.0 * open_n / n,
            },

            "release_subtree": {
                "count": release_n,
                "percent": 100.0 * release_n / n,
            },

            "complete_bt": {
                "count": complete_n,
                "percent": 100.0 * complete_n / n,
            },

            "incomplete_sequences": incomplete,
        }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    out_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_json.write_text(
        json.dumps(
            {
                "methods": METHODS,
                "required_nodes": REQUIRED_NODES,
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )

    with out_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "method",
                "grasp",
                "open",
                "release",
                "complete",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 86)
    print("BT COMPLETENESS SUMMARY")
    print("=" * 86)

    print(
        f"{'METHOD':18s} "
        f"{'GRASP':12s} "
        f"{'OPEN':12s} "
        f"{'RELEASE':12s} "
        f"{'COMPLETE':12s}"
    )

    print("-" * 86)

    for method in METHODS:
        s = summary[method]
        n = s["n_sequences"]

        print(
            f"{method:18s} "
            f"{s['grasp_subtree']['count']:2d}/{n:<2d} "
            f"({s['grasp_subtree']['percent']:5.1f}%)  "
            f"{s['open_subtree']['count']:2d}/{n:<2d} "
            f"({s['open_subtree']['percent']:5.1f}%)  "
            f"{s['release_subtree']['count']:2d}/{n:<2d} "
            f"({s['release_subtree']['percent']:5.1f}%)  "
            f"{s['complete_bt']['count']:2d}/{n:<2d} "
            f"({s['complete_bt']['percent']:5.1f}%)"
        )

    print()

    for method in METHODS:
        incomplete = summary[
            method
        ]["incomplete_sequences"]

        print(
            f"{method} incomplete:"
        )

        if incomplete:
            for seq in incomplete:
                print(
                    f"  {seq}"
                )
        else:
            print("  none")

        print()

    print("Saved:")
    print(" ", out_json)
    print(" ", out_csv)


if __name__ == "__main__":
    main()
