#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAST = 8
FUTURE = 8
GRASP_LOOKBACK = 12


def percentile(series):
    return (
        series.rank(
            pct=True,
            method="average",
        )
        * 100.0
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hand_csv",
        required=True,
    )

    parser.add_argument(
        "--motion_json",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    hands = pd.read_csv(
        args.hand_csv
    )

    motion_data = json.loads(
        Path(args.motion_json).read_text()
    )

    sequence = motion_data["sequence"]

    motion = pd.DataFrame(
        motion_data["features"]
    )

    df = hands.merge(
        motion,
        on="frame",
        how="inner",
        suffixes=(
            "_hand",
            "_flow",
        ),
    )

    df = df.sort_values(
        "frame"
    ).reset_index(drop=True)

    if len(df) < 8:
        raise RuntimeError(
            f"{sequence}: too few usable frames "
            f"({len(df)}; minimum 8)"
        )

    # Adaptive temporal windows.
    #
    # Long clips preserve the original 8/8 context.
    # Short clips use smaller valid windows.
    PAST = min(
        8,
        max(
            3,
            len(df) // 4,
        ),
    )

    FUTURE = min(
        8,
        max(
            3,
            len(df) // 4,
        ),
    )

    while (
        PAST + FUTURE + 2 > len(df)
        and (
            PAST > 3
            or FUTURE > 3
        )
    ):
        if PAST >= FUTURE and PAST > 3:
            PAST -= 1
        elif FUTURE > 3:
            FUTURE -= 1

    min_valid_side = max(
        2,
        min(
            5,
            PAST,
            FUTURE,
        ),
    )

    grip_coverage = float(
        np.mean(
            df["grip_distance"].notna()
        )
    )

    two_hand_coverage_local = float(
        np.mean(
            df["detected_hands"] >= 2
        )
    )

    use_optical_fallback = (
        grip_coverage < 0.20
        or two_hand_coverage_local < 0.15
    )

    candidate_mode = (
        "optical_flow_fallback"
        if use_optical_fallback
        else "hand_geometry+optical_flow"
    )

    def build_optical_fallback():
        """
        Build opening-change candidates from stabilised optical flow
        when bimanual hand geometry is unavailable.
        """

        rows = []

        opp = (
            df["opposing_horizontal"]
            .fillna(0.0)
        )

        div = (
            df["positive_divergence"]
            .fillna(0.0)
        )

        # Sequence-relative evidence avoids fixed pixel-motion thresholds.
        optical_signal = (
            0.55
            * opp.rank(
                pct=True,
                method="average",
            )
            +
            0.45
            * div.rank(
                pct=True,
                method="average",
            )
        )

        for j in range(
            PAST - 1,
            len(df) - FUTURE - 1,
        ):

            past_opt = optical_signal.iloc[
                j - PAST + 1:
                j + 1
            ]

            future_opt = optical_signal.iloc[
                j + 1:
                j + 1 + FUTURE
            ]

            if (
                len(past_opt) < 2
                or len(future_opt) < 2
            ):
                continue

            before_opt = float(
                np.median(
                    past_opt
                )
            )

            after_opt = float(
                np.median(
                    future_opt
                )
            )

            persistence = float(
                np.mean(
                    future_opt > before_opt
                )
            )

            rows.append(
                {
                    "candidate_frame":
                        int(
                            df.iloc[
                                j + 1
                            ]["frame"]
                        ),

                    # Hand-distance geometry is unavailable.
                    # These retain the existing event schema.
                    "before_distance":
                        0.0,

                    "after_distance":
                        0.0,

                    # Relative optical change used for candidate ranking.
                    "level_shift":
                        float(
                            after_opt - before_opt
                        ),

                    "persistence":
                        persistence,

                    "separation_support":
                        0.0,

                    "optical_support":
                        float(
                            np.median(
                                future_opt.iloc[
                                    :min(
                                        5,
                                        len(future_opt),
                                    )
                                ]
                            )
                        ),
                }
            )

        return pd.DataFrame(rows)
    first_frame = int(
        df["frame"].min()
    )

    last_frame = int(
        df["frame"].max()
    )

    # ============================================================
    # Bimanual pinch feature
    # ============================================================

    df["worst_pinch"] = df[
        [
            "h0_pinch_ratio",
            "h1_pinch_ratio",
        ]
    ].max(
        axis=1
    )

    # ============================================================
    # Compute every possible sustained opening change point
    # ============================================================

    splits = []

    for i in range(
        PAST - 1,
        len(df) - FUTURE - 1,
    ):

        past = df.iloc[
            i - PAST + 1:
            i + 1
        ]

        future = df.iloc[
            i + 1:
            i + 1 + FUTURE
        ]

        past_dist = past[
            "grip_distance"
        ].dropna()

        future_dist = future[
            "grip_distance"
        ].dropna()

        if (
            len(past_dist) < min_valid_side
            or len(future_dist) < min_valid_side
        ):
            continue

        before = float(
            np.median(
                past_dist
            )
        )

        after = float(
            np.median(
                future_dist
            )
        )

        shift = (
            after - before
        )

        persistence = float(
            np.mean(
                future_dist > before
            )
        )

        sep = future[
            "separating_speed_mean5"
        ].dropna()

        separation_support = (
            float(
                np.median(
                    sep.iloc[:5]
                )
            )
            if len(sep)
            else 0.0
        )

        opp = future[
            "opposing_horizontal"
        ].dropna()

        div = future[
            "positive_divergence"
        ].dropna()

        if len(opp) and len(div):

            optical_support = (
                float(
                    np.median(
                        opp.iloc[:5]
                    )
                )
                +
                float(
                    np.median(
                        div.iloc[:5]
                    )
                )
            ) / 2.0

        else:
            optical_support = 0.0

        splits.append(
            {
                "candidate_frame":
                    int(
                        df.iloc[
                            i + 1
                        ]["frame"]
                    ),

                "before_distance":
                    before,

                "after_distance":
                    after,

                "level_shift":
                    shift,

                "persistence":
                    persistence,

                "separation_support":
                    separation_support,

                "optical_support":
                    optical_support,
            }
        )

    split_df = pd.DataFrame(
        splits
    )

    if (
        use_optical_fallback
        or len(split_df) == 0
    ):
        candidate_mode = (
            "optical_flow_fallback"
        )

        split_df = (
            build_optical_fallback()
        )

    if len(split_df) == 0:
        raise RuntimeError(
            f"{sequence}: no usable hand or optical opening candidates"
        )
    # ============================================================
    # Score change points
    # ============================================================

    positive_shift = np.maximum(
        split_df["level_shift"],
        0.0,
    )

    split_df["shift_pct"] = percentile(
        positive_shift
    )

    split_df["sep_pct"] = percentile(
        split_df[
            "separation_support"
        ]
    )

    split_df["optical_pct"] = percentile(
        split_df[
            "optical_support"
        ]
    )

    split_df["opening_score"] = (
        0.60
        * split_df["shift_pct"]
        +
        0.20
        * split_df["sep_pct"]
        +
        0.10
        * split_df["optical_pct"]
        +
        0.10
        * (
            split_df["persistence"]
            * 100.0
        )
    )

    positive = split_df[
        (
            split_df["level_shift"] > 0
        )
        &
        (
            split_df["persistence"] >= 0.75
        )
    ].copy()

    if len(positive) == 0:
        candidate_mode = (
            "optical_flow_fallback"
        )

        split_df = (
            build_optical_fallback()
        )

        if len(split_df) == 0:
            raise RuntimeError(
                f"{sequence}: no persistent hand or optical "
                f"opening events"
            )

        positive_shift = np.maximum(
            split_df[
                "level_shift"
            ],
            0.0,
        )

        split_df["shift_pct"] = (
            percentile(
                pd.Series(
                    positive_shift,
                    index=split_df.index,
                )
            )
        )

        split_df["sep_pct"] = (
            percentile(
                split_df[
                    "separation_support"
                ]
            )
        )

        split_df["optical_pct"] = (
            percentile(
                split_df[
                    "optical_support"
                ]
            )
        )

        split_df[
            "opening_score"
        ] = (
            0.60
            * split_df[
                "shift_pct"
            ]
            +
            0.20
            * split_df[
                "sep_pct"
            ]
            +
            0.10
            * split_df[
                "optical_pct"
            ]
            +
            0.10
            * (
                split_df[
                    "persistence"
                ]
                * 100.0
            )
        )

        positive = split_df[
            (
                split_df[
                    "level_shift"
                ] > 0
            )
            &
            (
                split_df[
                    "persistence"
                ] >= 0.60
            )
        ].copy()

        # Last graceful optical fallback.
        if len(positive) == 0:
            positive = (
                split_df[
                    split_df[
                        "level_shift"
                    ] > 0
                ]
                .nlargest(
                    min(
                        8,
                        len(split_df),
                    ),
                    "opening_score",
                )
                .copy()
            )

        if len(positive) == 0:
            raise RuntimeError(
                f"{sequence}: no positive optical opening evidence"
            )
    # ============================================================
    # Keep strong candidates.
    #
    # Unlike the old detector, this is NOT relative only to the
    # single strongest opening event. We want weaker later pulls too.
    # ============================================================

    score_cut = max(
        75.0,
        float(
            np.percentile(
                positive[
                    "opening_score"
                ],
                65,
            )
        ),
    )

    strong = positive[
        positive[
            "opening_score"
        ] >= score_cut
    ].copy()

    strong = strong.sort_values(
        "candidate_frame"
    ).reset_index(drop=True)

    if len(strong) == 0:

        strong = positive.nlargest(
            min(
                5,
                len(positive),
            ),
            "opening_score",
        ).sort_values(
            "candidate_frame"
        ).reset_index(drop=True)

    # ============================================================
    # Cluster neighbouring candidate frames.
    #
    # Each cluster is an OPENING BURST CANDIDATE.
    #
    # It is not automatically a separate peel label.
    # Pixtral will later determine whether a later burst is:
    #
    #   - continuation of current peel
    #   - regrasp then another peel
    #   - false feature candidate
    # ============================================================

    clusters = []
    current = []

    for _, row in strong.iterrows():

        frame = int(
            row["candidate_frame"]
        )

        if not current:

            current = [
                row
            ]

            continue

        previous_frame = int(
            current[-1][
                "candidate_frame"
            ]
        )

        if (
            frame - previous_frame
            <= 3
        ):

            current.append(
                row
            )

        else:

            clusters.append(
                current
            )

            current = [
                row
            ]

    if current:
        clusters.append(
            current
        )

    # ============================================================
    # Turn clusters into opening-burst proposals
    # ============================================================

    bursts = []

    for cluster_id, cluster in enumerate(
        clusters,
        start=1,
    ):

        cluster_df = pd.DataFrame(
            cluster
        )

        cluster_df = (
            cluster_df.sort_values(
                "candidate_frame"
            )
            .reset_index(
                drop=True
            )
        )

        cluster_max = float(
            cluster_df[
                "opening_score"
            ].max()
        )

        # Find earliest frame sufficiently close to
        # the cluster's strongest evidence.
        onset_threshold = (
            0.90 * cluster_max
        )

        onset_rows = cluster_df[
            cluster_df[
                "opening_score"
            ] >= onset_threshold
        ]

        if len(onset_rows):

            onset_row = (
                onset_rows.iloc[0]
            )

        else:

            onset_row = (
                cluster_df.iloc[0]
            )

        onset = int(
            onset_row[
                "candidate_frame"
            ]
        )

        cluster_start = int(
            cluster_df[
                "candidate_frame"
            ].min()
        )

        cluster_end = int(
            cluster_df[
                "candidate_frame"
            ].max()
        )

        # --------------------------------------------------------
        # Candidate grasp/regrasp immediately before this burst
        # --------------------------------------------------------

        grasp_search_start = max(
            first_frame,
            onset - GRASP_LOOKBACK,
        )

        grasp_search_end = (
            onset - 1
        )

        grasp_region = df[
            (
                df["frame"]
                >= grasp_search_start
            )
            &
            (
                df["frame"]
                <= grasp_search_end
            )
        ].copy()

        valid_grasp = grasp_region[
            grasp_region[
                "detected_hands"
            ] >= 2
        ].dropna(
            subset=[
                "worst_pinch"
            ]
        )

        if len(valid_grasp):

            idx = valid_grasp[
                "worst_pinch"
            ].idxmin()

            grasp_candidate = int(
                valid_grasp.loc[
                    idx,
                    "frame",
                ]
            )

            grasp_pinch = float(
                valid_grasp.loc[
                    idx,
                    "worst_pinch",
                ]
            )

        else:

            grasp_candidate = max(
                first_frame,
                onset - 3,
            )

            grasp_pinch = None

        burst = {
            "burst_id":
                cluster_id,

            "opening_candidate":
                onset,

            "opening_window": [
                max(
                    first_frame,
                    onset - 3,
                ),
                min(
                    last_frame,
                    onset + 3,
                ),
            ],

            "feature_cluster": [
                cluster_start,
                cluster_end,
            ],

            "cluster_max_score":
                cluster_max,

            "before_distance":
                float(
                    onset_row[
                        "before_distance"
                    ]
                ),

            "after_distance":
                float(
                    onset_row[
                        "after_distance"
                    ]
                ),

            "level_shift":
                float(
                    onset_row[
                        "level_shift"
                    ]
                ),

            "persistence":
                float(
                    onset_row[
                        "persistence"
                    ]
                ),

            "preceding_grasp_candidate":
                grasp_candidate,

            "preceding_grasp_window": [
                max(
                    first_frame,
                    grasp_candidate - 3,
                ),
                min(
                    onset - 1,
                    grasp_candidate + 3,
                ),
            ],

            "preceding_worst_pinch":
                grasp_pinch,
        }

        bursts.append(
            burst
        )

    if len(bursts) == 0:
        raise RuntimeError(
            f"{sequence}: no opening burst clusters"
        )

    # ============================================================
    # Build inter-burst intervals.
    #
    # These are the regions Pixtral must inspect to decide:
    #
    #   CONTINUOUS_PEEL
    # or
    #   REGRASP_THEN_PEEL
    # ============================================================

    inter_burst = []

    for i in range(
        1,
        len(bursts),
    ):

        previous = bursts[
            i - 1
        ]

        current = bursts[
            i
        ]

        previous_onset = int(
            previous[
                "opening_candidate"
            ]
        )

        current_onset = int(
            current[
                "opening_candidate"
            ]
        )

        start = min(
            current_onset - 1,
            previous_onset + 3,
        )

        end = (
            current_onset - 1
        )

        candidate_grasp = int(
            current[
                "preceding_grasp_candidate"
            ]
        )

        inter_burst.append(
            {
                "from_burst":
                    previous[
                        "burst_id"
                    ],

                "to_burst":
                    current[
                        "burst_id"
                    ],

                "interval": [
                    start,
                    end,
                ],

                "regrasp_candidate":
                    candidate_grasp,

                "regrasp_window": (
                    current[
                        "preceding_grasp_window"
                    ]
                ),
            }
        )

    # ============================================================
    # TERMINAL DROP PROPOSAL
    #
    # Crucial change:
    # Search for drop only AFTER THE LAST OPENING BURST.
    #
    # This prevents sequence 2's first pull ending at ~62 from
    # becoming a false terminal drop when another opening episode
    # occurs around ~95.
    # ============================================================

    final_burst = bursts[-1]

    final_opening = int(
        final_burst[
            "opening_candidate"
        ]
    )

    post_final = df[
        df["frame"]
        >= final_opening + 5
    ].copy()

    valid_peak = post_final.dropna(
        subset=[
            "separating_speed_mean5"
        ]
    )

    if len(valid_peak):

        idx = valid_peak[
            "separating_speed_mean5"
        ].idxmax()

        final_sep_peak = int(
            df.loc[
                idx,
                "frame",
            ]
        )

    else:

        final_sep_peak = (
            final_opening
        )

    drop_search = df[
        df["frame"]
        > final_sep_peak
    ].copy()

    drop_candidate = None

    for _, row in drop_search.iterrows():

        hand_decline = (
            pd.notna(
                row[
                    "separation_trend"
                ]
            )
            and
            row[
                "separation_trend"
            ] < 0
        )

        flow_decline = (
            row[
                "opposing_horizontal_trend"
            ] < 0
            and
            row[
                "positive_divergence_trend"
            ] < 0
        )

        if (
            hand_decline
            and flow_decline
        ):

            drop_candidate = int(
                row["frame"]
            )

            break

    if drop_candidate is None:

        declining = drop_search[
            drop_search[
                "separation_trend"
            ] < 0
        ]

        if len(declining):

            drop_candidate = int(
                declining.iloc[
                    0
                ]["frame"]
            )

        else:

            drop_candidate = (
                last_frame
            )

    # ============================================================
    # OUTPUT
    # ============================================================

    output = {
        "sequence":
            sequence,

        "frame_range": [
            first_frame,
            last_frame,
        ],

        "coverage": {
            "one_or_more_hands":
                float(
                    np.mean(
                        df[
                            "detected_hands"
                        ] >= 1
                    )
                ),

            "two_hands":
                float(
                    np.mean(
                        df[
                            "detected_hands"
                        ] >= 2
                    )
                ),
        },

        "opening_burst_candidates":
            bursts,

        "inter_burst_candidates":
            inter_burst,

        "terminal_release_candidate": {
            "last_opening_burst":
                int(
                    final_burst[
                        "burst_id"
                    ]
                ),

            "last_opening_onset":
                final_opening,

            "final_separation_peak":
                final_sep_peak,

            "drop_candidate":
                drop_candidate,

            "drop_window": [
                max(
                    first_frame,
                    drop_candidate - 4,
                ),
                min(
                    last_frame,
                    drop_candidate + 4,
                ),
            ],
        },

        "notes": {
            "opening_bursts_are_proposals":
                True,

            "later_burst_requires_pixtral_semantic_verification":
                True,

            "drop_is_searched_only_after_last_opening_burst":
                True,
        },
    }

    out_path = Path(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
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
        f"BT EVENT CANDIDATES: {sequence}"
    )
    print("=" * 90)

    print(
        f"frames: "
        f"{first_frame}-{last_frame}"
    )
    print(
        f"candidate mode: "
        f"{candidate_mode}"
    )

    print(
        f"temporal context: "
        f"past={PAST}, "
        f"future={FUTURE}, "
        f"min_valid={min_valid_side}"
    )


    print(
        f"one+ hand coverage: "
        f"{100*output['coverage']['one_or_more_hands']:.1f}%"
    )

    print(
        f"two-hand coverage: "
        f"{100*output['coverage']['two_hands']:.1f}%"
    )

    print()
    print(
        f"opening burst candidates: "
        f"{len(bursts)}"
    )

    for burst in bursts:

        print()
        print(
            f"BURST {burst['burst_id']}"
        )

        print(
            f"  grasp/regrasp candidate: "
            f"{burst['preceding_grasp_candidate']} "
            f"window="
            f"{burst['preceding_grasp_window']}"
        )

        print(
            f"  opening onset candidate: "
            f"{burst['opening_candidate']} "
            f"window="
            f"{burst['opening_window']}"
        )

        print(
            f"  feature cluster: "
            f"{burst['feature_cluster']}"
        )

        print(
            f"  distance: "
            f"{burst['before_distance']:.4f}"
            f" -> "
            f"{burst['after_distance']:.4f}"
        )

        print(
            f"  shift="
            f"{burst['level_shift']:+.4f}, "
            f"persistence="
            f"{100*burst['persistence']:.1f}%"
        )

    if inter_burst:

        print()
        print(
            "INTER-BURST REGIONS REQUIRING "
            "PIXTRAL SEMANTIC CHECK"
        )

        for item in inter_burst:

            print(
                f"  burst "
                f"{item['from_burst']} -> "
                f"{item['to_burst']}: "
                f"interval={item['interval']} "
                f"regrasp≈"
                f"{item['regrasp_candidate']}"
            )

    release = output[
        "terminal_release_candidate"
    ]

    print()
    print(
        "TERMINAL RELEASE PROPOSAL"
    )

    print(
        f"  after final opening burst: "
        f"{release['last_opening_onset']}"
    )

    print(
        f"  final separation peak: "
        f"{release['final_separation_peak']}"
    )

    print(
        f"  drop candidate: "
        f"{release['drop_candidate']} "
        f"window="
        f"{release['drop_window']}"
    )

    print()
    print(
        f"Wrote {out_path}"
    )


if __name__ == "__main__":
    main()
