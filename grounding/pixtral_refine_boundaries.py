#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from pixtral_labeler import load_model


# ================================================================
# PROMPTS
# ================================================================

GRASP_PROMPT = """
You are analysing a small chronological window from a sterile-package
opening demonstration.

Requested transition:

locate_flaps -> grasp_flaps

grasp_flaps begins at the FIRST frame where BOTH required package
sides/lips are securely controlled for the forthcoming peel.

Searching, touching, one-sided holding, failed acquisition and finger
repositioning remain locate_flaps.

Return ONLY valid JSON:

{
  "boundary_frame": integer_or_null,
  "visual_evidence": "short phrase",
  "feature_evidence": "short phrase"
}

Return null if this transition is not actually visible in the window.
No Markdown fences.
""".strip()


PEEL_PROMPT = """
You are analysing a small chronological window from a sterile-package
opening demonstration.

Requested transition:

grasp_flaps -> peel_apart

peel_apart begins at the FIRST frame where sustained wrapper separation
starts.

A small finger adjustment while the material is held remains grasp_flaps.

Use:
- visible pulling apart,
- increasing hand distance,
- separation velocity,
- opposing horizontal motion,
- divergence.

Choose the ONSET, not maximum opening.

Return ONLY valid JSON:

{
  "boundary_frame": integer_or_null,
  "visual_evidence": "short phrase",
  "feature_evidence": "short phrase"
}

Return null if sustained peeling does not actually begin in the window.
No Markdown fences.
""".strip()


INTER_BURST_PROMPT = """
You are analysing a sterile-package-opening demonstration after peeling has
already begun.

An automatic feature detector has detected another later opening-motion
burst. This feature burst is ONLY A PROPOSAL and may simply be continued
peeling or post-release hand motion.

You must determine what actually happens between the previous opening burst
and the later proposed burst.

Choose exactly ONE semantic decision:

CONTINUOUS_PEEL

The manipulation remains within peel_apart.

This includes:
- temporary slowing,
- changing pulling direction,
- tightening the grip,
- sliding fingers along the package,
- small grip corrections,
- repositioning while still controlling the material,
- brief reductions in hand separation.

IMPORTANT:
A regrasp-like finger adjustment while package control is maintained is
still CONTINUOUS_PEEL.

DISTINCT_REGRASP

Use this ONLY when ALL of the following are visibly supported:

1. the active peel clearly stops,
2. at least one hand visibly releases/disengages from its previous package
   contact or substantially opens its grip,
3. the hand deliberately reacquires a new section of package material,
4. a new sustained peel then begins.

This creates:

peel_apart -> grasp_flaps -> peel_apart

Do NOT choose DISTINCT_REGRASP merely because:
- pinch strength changes,
- hand distance decreases,
- the hands move closer together,
- the nurse slides fingers,
- the peel temporarily slows.

TERMINAL_DROP

Use this ONLY when:

1. the package-opening manipulation has ended,
2. behaviour has changed into terminal release/drop,
3. the later feature burst is NOT genuine renewed peeling.

drop_contents is terminal.

Large packages may contain multiple genuine pull cycles, so a temporary
pause must NOT be confused with drop.

Transparent contents do not need to be visibly falling.

Return ONLY valid JSON:

{
  "decision": "CONTINUOUS_PEEL or DISTINCT_REGRASP or TERMINAL_DROP",
  "release_from_previous_peel_observed": true_or_false,
  "reacquisition_observed": true_or_false,
  "later_sustained_peel_observed": true_or_false,
  "terminal_release_observed": true_or_false,
  "approx_regrasp_frame": integer_or_null,
  "approx_peel_frame": integer_or_null,
  "approx_drop_frame": integer_or_null,
  "visual_evidence": "short phrase",
  "feature_evidence": "short phrase"
}

Grounding requirements:

later_sustained_peel_observed=true means that genuine package
separation/opening is visibly still occurring at or after the later
feature-burst region. It does NOT have to be a new peel cycle.

CONTINUOUS_PEEL is valid ONLY if genuine peeling is visibly still
occurring at or after the later burst.

DISTINCT_REGRASP is valid ONLY if:
release_from_previous_peel_observed=true
AND reacquisition_observed=true
AND later_sustained_peel_observed=true.

TERMINAL_DROP is valid ONLY if:
terminal_release_observed=true
AND later_sustained_peel_observed=false.

If those requirements are not met, choose CONTINUOUS_PEEL.

No Markdown fences.
""".strip()


DROP_BEFORE_BURST_PROMPT = """
You are checking whether a possible terminal release/drop occurs BEFORE a
later automatic motion burst in a sterile-package-opening video.

The automatic feature detector may produce opening-like motion even AFTER
the package has already been released. Therefore the later feature burst is
NOT proof that peeling continues.

A candidate decline frame has been supplied.

Decide whether the behaviour around that candidate is:

TERMINAL DROP:
- active opening/peeling ends,
- the nurse changes into release/drop behaviour,
- package control used for peeling is relinquished,
- there is NO genuine renewed peel afterwards.

or

NOT TERMINAL:
- the candidate is only a pause, direction change, grip adjustment,
  regrasp, or temporary reduction in separation,
- genuine package peeling subsequently continues.

Important:
- transparent contents do not need to be visibly falling,
- hand motion after release is not automatically another peel,
- a later feature burst counts as renewed peeling ONLY if the RGB evidence
  shows genuine package separation/opening,
- if genuine peeling resumes later, this candidate CANNOT be drop_contents.

Return ONLY valid JSON:

{
  "terminal_drop_observed": true_or_false,
  "boundary_frame": integer_or_null,
  "later_genuine_peel_observed": true_or_false,
  "visual_evidence": "short phrase",
  "feature_evidence": "short phrase"
}

If uncertain, set terminal_drop_observed=false.

No Markdown fences.
""".strip()


GLOBAL_TERMINAL_DROP_PROMPT = """
You are deciding whether a candidate frame is the TERMINAL transition:

peel_apart -> drop_contents

in a sterile-package-opening demonstration.

This decision is independent of automatic opening-motion bursts.

A motion detector may produce false opening-like bursts after the package
has already been released, because the hands, wrapper, camera, or loose
material may continue moving.

TERMINAL DROP means:

- active package opening/peeling has ended,
- the manipulation changes into release/drop behaviour,
- the package is no longer being actively opened afterwards.

The contents may be transparent and do NOT need to be visibly falling.

CRITICAL RULE:

If genuine package peeling/opening visibly resumes anywhere AFTER the
candidate frame, then this candidate is NOT terminal drop.

Genuine later peeling means actual renewed package separation while the
package sides are being controlled.

The following do NOT count as later peeling by themselves:

- hand movement,
- wrapper movement after release,
- changing hand distance,
- optical-flow divergence,
- fingers moving apart,
- a feature detector reporting another opening burst.

Judge actual package-opening behaviour from the RGB images.

Return ONLY valid JSON:

{
  "terminal_drop_observed": true_or_false,
  "boundary_frame": integer_or_null,
  "genuine_peeling_after_candidate": true_or_false,
  "visual_evidence": "short phrase",
  "future_evidence": "short phrase"
}

terminal_drop_observed=true is valid ONLY when
genuine_peeling_after_candidate=false.

If uncertain, return terminal_drop_observed=false.

No Markdown fences.
""".strip()


DROP_REFINE_PROMPT = """
You are refining the terminal Behaviour Tree transition:

peel_apart -> drop_contents

Choose the FIRST frame where active package peeling/opening has changed into
the terminal release/drop behaviour.

Important:

- A temporary pause or regrasp is NOT drop_contents.
- drop_contents is terminal.
- The contents may be transparent.
- Do not require seeing the object fall.
- Use the manipulation change, hand behaviour, package behaviour and motion
  evidence together.

Return ONLY valid JSON:

{
  "boundary_frame": integer_or_null,
  "visual_evidence": "short phrase",
  "feature_evidence": "short phrase"
}

No Markdown fences.
""".strip()


# ================================================================
# HELPERS
# ================================================================

def parse_json(text):

    cleaned = (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        return json.loads(cleaned)
    except Exception:
        return None


def crop_image(image, fraction=0.85):

    h, w = image.shape[:2]

    rh = int(h * fraction)
    rw = int(w * fraction)

    y0 = (h - rh) // 2
    x0 = (w - rw) // 2

    return image[
        y0:y0 + rh,
        x0:x0 + rw
    ]


def fmt(value, digits=4):

    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass

    return f"{float(value):.{digits}f}"


def choose_display_frames(
    start,
    end,
    important=None,
    max_frames=12,
):

    important = important or []

    start = int(start)
    end = int(end)

    if end < start:
        return []

    all_frames = list(
        range(start, end + 1)
    )

    if len(all_frames) <= max_frames:
        return all_frames

    base_count = max(
        4,
        max_frames - len(important),
    )

    indices = np.linspace(
        0,
        len(all_frames) - 1,
        base_count,
    ).round().astype(int)

    selected = {
        all_frames[int(i)]
        for i in indices
    }

    for frame in important:

        frame = int(frame)

        if start <= frame <= end:
            selected.add(frame)

    selected = sorted(selected)

    # If important frames pushed us slightly over limit,
    # preserve important frames and thin ordinary samples.
    if len(selected) > max_frames:

        required = {
            int(x)
            for x in important
            if start <= int(x) <= end
        }

        ordinary = [
            x for x in selected
            if x not in required
        ]

        remaining = max(
            0,
            max_frames - len(required),
        )

        if len(ordinary) > remaining and remaining > 0:

            indices = np.linspace(
                0,
                len(ordinary) - 1,
                remaining,
            ).round().astype(int)

            ordinary = [
                ordinary[int(i)]
                for i in indices
            ]

        elif remaining == 0:
            ordinary = []

        selected = sorted(
            required.union(ordinary)
        )

    return selected


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--events",
        required=True,
    )

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

    events_file = Path(args.events)
    hand_file = Path(args.hand_csv)
    motion_file = Path(args.motion_json)
    out_file = Path(args.out)

    events = json.loads(
        events_file.read_text()
    )

    sequence = events["sequence"]

    hands = pd.read_csv(
        hand_file
    )

    motion_data = json.loads(
        motion_file.read_text()
    )

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

    first_frame = int(
        df["frame"].min()
    )

    last_frame = int(
        df["frame"].max()
    )

    # ------------------------------------------------------------
    # Relative contextual features
    # ------------------------------------------------------------

    df["worst_pinch"] = df[
        [
            "h0_pinch_ratio",
            "h1_pinch_ratio",
        ]
    ].max(
        axis=1
    )

    df["pinch_strength_pct"] = (
        df["worst_pinch"]
        .rank(
            ascending=False,
            pct=True,
        )
        * 100.0
    )

    df["sep_pct"] = (
        df["separating_speed_mean5"]
        .rank(
            pct=True
        )
        * 100.0
    )

    df["opp_pct"] = (
        df["opposing_horizontal"]
        .rank(
            pct=True
        )
        * 100.0
    )

    df["div_pct"] = (
        df["positive_divergence"]
        .rank(
            pct=True
        )
        * 100.0
    )

    by_frame = {
        int(row["frame"]): row
        for _, row in df.iterrows()
    }

    crop_dir = (
        Path("pixtral_transition_crops")
        / sequence
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Source image
    # ============================================================

    def get_path(frame):

        row = by_frame[
            int(frame)
        ]

        for key in [
            "frame_path_hand",
            "frame_path",
            "frame_path_flow",
        ]:

            if key in row.index:

                value = row[key]

                if isinstance(value, str):
                    return Path(value)

        raise RuntimeError(
            f"No source image path for frame {frame}"
        )

    def make_crop(frame):

        frame = int(frame)

        path = get_path(frame)

        image = cv2.imread(
            str(path)
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read {path}"
            )

        crop = crop_image(
            image,
            0.85,
        )

        out = (
            crop_dir
            / f"frame_{frame:06d}.png"
        )

        cv2.imwrite(
            str(out),
            crop,
        )

        return out

    # ============================================================
    # Feature table
    # ============================================================

    def feature_table(start, end):

        lines = []

        lines.append(
            "frame | hands | pinch | grip-distance | "
            "sep-speed | sep-trend | opposing | opp-trend | "
            "divergence | div-trend"
        )

        for frame in range(
            int(start),
            int(end) + 1,
        ):

            if frame not in by_frame:
                continue

            r = by_frame[frame]

            lines.append(
                f"{frame:3d} | "
                f"{int(r['detected_hands'])} | "
                f"{fmt(r['pinch_strength_pct'], 1)} | "
                f"{fmt(r['grip_distance'])} | "
                f"{fmt(r['separating_speed_mean5'])} | "
                f"{fmt(r['separation_trend'])} | "
                f"{fmt(r['opposing_horizontal'], 3)} | "
                f"{fmt(r['opposing_horizontal_trend'], 3)} | "
                f"{fmt(r['positive_divergence'], 3)} | "
                f"{fmt(r['positive_divergence_trend'], 3)}"
            )

        return "\n".join(lines)

    # ============================================================
    # Model
    # ============================================================

    model, processor = load_model(
        load_4bit=False
    )

    def run_pixtral(
        prompt,
        start,
        end,
        important=None,
        max_images=12,
        max_tokens=180,
    ):

        start = max(
            first_frame,
            int(start),
        )

        end = min(
            last_frame,
            int(end),
        )

        display_frames = choose_display_frames(
            start,
            end,
            important=important,
            max_frames=max_images,
        )

        content = [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "text",
                "text": (
                    f"SEQUENCE: {sequence}\n"
                    f"WINDOW: {start}-{end}\n\n"
                    f"AUTOMATIC FEATURES:\n"
                    f"{feature_table(start, end)}"
                ),
            },
            {
                "type": "text",
                "text": (
                    "Chronological RGB crops follow. "
                    "Source-frame numbers are shown explicitly."
                ),
            },
        ]

        for frame in display_frames:

            if frame not in by_frame:
                continue

            content.append(
                {
                    "type": "text",
                    "text": (
                        f"[SOURCE FRAME {frame}]"
                    ),
                }
            )

            content.append(
                {
                    "type": "image",
                    "url": str(
                        make_crop(frame)
                    ),
                }
            )

        chat = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(
            model.device
        )

        with torch.inference_mode():

            generated = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        input_length = (
            inputs["input_ids"].shape[1]
        )

        raw = processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        parsed = parse_json(
            raw
        )

        del inputs
        del generated

        torch.cuda.empty_cache()

        return {
            "parsed": parsed,
            "raw": raw,
            "displayed_frames":
                display_frames,
        }

    # ============================================================
    # Boundary refinement helper
    # ============================================================

    def refine_boundary(
        prompt,
        window,
        candidate,
    ):

        start, end = [
            int(x)
            for x in window
        ]

        result = run_pixtral(
            prompt,
            start,
            end,
            important=[
                candidate
            ],
            max_images=12,
        )

        boundary = None

        parsed = result[
            "parsed"
        ]

        if parsed is not None:

            value = parsed.get(
                "boundary_frame"
            )

            try:

                if value is not None:

                    value = int(value)

                    if (
                        start
                        <= value
                        <= end
                    ):
                        boundary = value

            except Exception:
                pass

        result[
            "boundary_frame"
        ] = boundary

        return result

    # ============================================================
    # POTENTIAL TERMINAL RELEASE INSIDE AN INTER-BURST GAP
    #
    # A later optical/hand burst may be post-release motion, so
    # terminal release must be allowed BEFORE the last feature burst.
    # ============================================================

    def find_local_drop_candidate(
        start,
        end,
        after_frame,
    ):

        start = max(
            int(start),
            int(after_frame) + 3,
            first_frame,
        )

        end = min(
            int(end),
            last_frame,
        )

        if end < start:
            return None

        region = df[
            (
                df["frame"] >= start
            )
            &
            (
                df["frame"] <= end
            )
        ].copy()

        if len(region) == 0:
            return None

        # Require decline in hand separation AND both optical
        # separation cues.
        candidates = region[
            (
                region[
                    "separation_trend"
                ] < 0
            )
            &
            (
                region[
                    "opposing_horizontal_trend"
                ] < 0
            )
            &
            (
                region[
                    "positive_divergence_trend"
                ] < 0
            )
        ]

        if len(candidates) == 0:
            return None

        # Prefer the earliest decline that persists rather than a
        # single-frame fluctuation.
        for _, row in candidates.iterrows():

            frame = int(
                row["frame"]
            )

            future = region[
                (
                    region["frame"] >= frame
                )
                &
                (
                    region["frame"] <= frame + 2
                )
            ]

            if len(future) == 0:
                continue

            if optical_only:

                persistent = (
                    (
                        future[
                            "opposing_horizontal_trend"
                        ] < 0
                    )
                    &
                    (
                        future[
                            "positive_divergence_trend"
                        ] < 0
                    )
                )

            else:

                persistent = (
                    (
                        future[
                            "separation_trend"
                        ] < 0
                    )
                    &
                    (
                        future[
                            "opposing_horizontal_trend"
                        ] < 0
                    )
                    &
                    (
                        future[
                            "positive_divergence_trend"
                        ] < 0
                    )
                )

            if int(
                persistent.sum()
            ) >= min(
                2,
                len(future),
            ):
                return frame

        # Graceful fallback to first cross-modal decline.
        return int(
            candidates.iloc[0][
                "frame"
            ]
        )


    def validate_drop_before_burst(
        candidate,
        gap_start,
        gap_end,
        later_burst_onset,
    ):

        candidate = int(
            candidate
        )

        validation_start = max(
            first_frame,
            candidate - 4,
        )

        # Deliberately show frames AFTER the later feature burst.
        # This lets Pixtral distinguish genuine renewed peeling
        # from post-release hand/package motion.
        validation_end = min(
            last_frame,
            int(later_burst_onset) + 6,
        )

        prompt = (
            DROP_BEFORE_BURST_PROMPT
            + "\n\n"
            + (
                f"AUTOMATIC DECLINE CANDIDATE: "
                f"{candidate}\n"
                f"INTER-BURST GAP: "
                f"{int(gap_start)}-{int(gap_end)}\n"
                f"LATER FEATURE-BURST ONSET: "
                f"{int(later_burst_onset)}\n\n"
                "The later feature burst is only a computer-vision "
                "proposal. Decide from the RGB sequence whether genuine "
                "package peeling actually resumes."
            )
        )

        result = run_pixtral(
            prompt,
            validation_start,
            validation_end,
            important=[
                candidate,
                int(gap_end),
                int(later_burst_onset),
                min(
                    last_frame,
                    int(later_burst_onset) + 3,
                ),
            ],
            max_images=18,
            max_tokens=200,
        )

        parsed = result[
            "parsed"
        ]

        accepted = False
        boundary = None
        later_peel = False

        if parsed is not None:

            def read_bool(key):

                value = parsed.get(
                    key,
                    False,
                )

                if isinstance(
                    value,
                    bool,
                ):
                    return value

                return str(
                    value
                ).strip().lower() in {
                    "true",
                    "yes",
                    "1",
                }

            terminal = read_bool(
                "terminal_drop_observed"
            )

            later_peel = read_bool(
                "later_genuine_peel_observed"
            )

            value = parsed.get(
                "boundary_frame"
            )

            try:

                if value is not None:

                    value = int(
                        value
                    )

                    # A terminal boundary must occur before the
                    # proposed later opening burst.
                    if (
                        max(
                            first_frame,
                            candidate - 5,
                        )
                        <= value
                        <= int(gap_end)
                    ):
                        boundary = value

            except Exception:
                boundary = None

            accepted = (
                terminal
                and not later_peel
                and boundary is not None
            )

        result[
            "accepted_terminal_drop"
        ] = bool(
            accepted
        )

        result[
            "boundary_frame"
        ] = boundary

        result[
            "later_genuine_peel_observed"
        ] = bool(
            later_peel
        )

        return result


    # ============================================================
    # INDEPENDENT TERMINAL DROP RESOLVER
    #
    # Drop is deliberately NOT tied to the final feature burst.
    # ============================================================

    def get_terminal_decline_candidates(
        first_peel_frame,
        max_candidates=10,
    ):

        region = df[
            df["frame"]
            >= int(first_peel_frame) + 5
        ].copy()

        if len(region) == 0:
            return []

        mask = (
            (
                region[
                    "separation_trend"
                ] < 0
            )
            &
            (
                region[
                    "opposing_horizontal_trend"
                ] < 0
            )
            &
            (
                region[
                    "positive_divergence_trend"
                ] < 0
            )
        )

        frames = [
            int(x)
            for x in region.loc[
                mask,
                "frame",
            ].tolist()
        ]

        if not frames:
            return []

        # --------------------------------------------------------
        # Cluster neighbouring decline frames.
        # A sustained decline produces one proposal rather than
        # one proposal for every frame.
        # --------------------------------------------------------

        clusters = []
        current = [frames[0]]

        for frame in frames[1:]:

            if frame - current[-1] <= 2:

                current.append(
                    frame
                )

            else:

                clusters.append(
                    current
                )

                current = [
                    frame
                ]

        clusters.append(
            current
        )

        candidates = []

        for cluster in clusters:

            # Prefer persistent clusters.
            if len(cluster) >= 2:

                candidates.append(
                    int(cluster[0])
                )

        # If everything was isolated, retain the isolated
        # candidates rather than failing completely.
        if not candidates:

            candidates = [
                int(cluster[0])
                for cluster in clusters
            ]

        # Also retain the existing feature release candidate as
        # an additional fallback proposal.
        try:

            fallback = int(
                events[
                    "terminal_release_candidate"
                ][
                    "drop_candidate"
                ]
            )

            candidates.append(
                fallback
            )

        except Exception:
            pass

        candidates = sorted(
            set(
                x
                for x in candidates
                if (
                    int(first_peel_frame) + 4
                    < x
                    <= last_frame
                )
            )
        )

        # During development preserve candidates across the whole
        # sequence rather than only the earliest fluctuations.
        if len(candidates) > max_candidates:

            indices = np.linspace(
                0,
                len(candidates) - 1,
                max_candidates,
            ).round().astype(int)

            candidates = [
                candidates[int(i)]
                for i in indices
            ]

            candidates = sorted(
                set(candidates)
            )

        return candidates


    def validate_global_terminal_drop(
        candidate,
        opening_bursts,
    ):

        candidate = int(
            candidate
        )

        local_start = max(
            first_frame,
            candidate - 5,
        )

        local_end = min(
            last_frame,
            candidate + 5,
        )

        # --------------------------------------------------------
        # Show every local frame around the proposed transition.
        # --------------------------------------------------------

        local_frames = [
            frame
            for frame in range(
                local_start,
                local_end + 1,
            )
            if frame in by_frame
        ]

        # --------------------------------------------------------
        # Then show a FUTURE OVERVIEW extending to sequence end.
        #
        # Feature burst centres are intentionally included as
        # important frames, but the prompt tells Pixtral that they
        # are only proposals.
        # --------------------------------------------------------

        future_important = []

        for burst in opening_bursts:

            onset = int(
                burst[
                    "opening_candidate"
                ]
            )

            if onset > candidate:

                future_important.extend(
                    [
                        onset,
                        min(
                            last_frame,
                            onset + 3,
                        ),
                    ]
                )

        future_start = min(
            last_frame,
            candidate + 6,
        )

        if future_start <= last_frame:

            future_frames = (
                choose_display_frames(
                    future_start,
                    last_frame,
                    important=
                        future_important,
                    max_frames=10,
                )
            )

        else:

            future_frames = []

        display_frames = sorted(
            set(
                local_frames
                + future_frames
            )
        )

        later_bursts = [
            int(
                burst[
                    "opening_candidate"
                ]
            )
            for burst
            in opening_bursts
            if int(
                burst[
                    "opening_candidate"
                ]
            ) > candidate
        ]

        content = [
            {
                "type":
                    "text",

                "text":
                    GLOBAL_TERMINAL_DROP_PROMPT,
            },

            {
                "type":
                    "text",

                "text": (
                    f"SEQUENCE: {sequence}\n"
                    f"TERMINAL-DROP CANDIDATE: "
                    f"{candidate}\n"
                    f"LOCAL WINDOW: "
                    f"{local_start}-{local_end}\n"
                    f"LAST SOURCE FRAME: "
                    f"{last_frame}\n"
                    f"LATER AUTOMATIC FEATURE-BURST PROPOSALS: "
                    f"{later_bursts}\n\n"
                    "Those later burst numbers are NOT semantic labels. "
                    "They are shown only so that you inspect those "
                    "regions particularly carefully."
                ),
            },

            {
                "type":
                    "text",

                "text": (
                    "LOCAL MOTION/HAND FEATURES AROUND "
                    "THE DROP CANDIDATE:\n"
                    + feature_table(
                        local_start,
                        local_end,
                    )
                ),
            },

            {
                "type":
                    "text",

                "text": (
                    "Chronological images follow. "
                    "The first group is around the candidate. "
                    "Later images provide evidence about whether "
                    "genuine package peeling ever resumes."
                ),
            },
        ]

        for frame in display_frames:

            if frame not in by_frame:
                continue

            if frame <= local_end:

                label = (
                    f"[LOCAL SOURCE FRAME {frame}]"
                )

            else:

                label = (
                    f"[FUTURE SOURCE FRAME {frame}]"
                )

            content.append(
                {
                    "type":
                        "text",

                    "text":
                        label,
                }
            )

            content.append(
                {
                    "type":
                        "image",

                    "url":
                        str(
                            make_crop(
                                frame
                            )
                        ),
                }
            )

        chat = [
            {
                "role":
                    "user",

                "content":
                    content,
            }
        ]

        inputs = (
            processor.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        inputs = inputs.to(
            model.device
        )

        with torch.inference_mode():

            generated = (
                model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=False,
                )
            )

        input_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )

        raw = (
            processor.batch_decode(
                generated[
                    :,
                    input_length:
                ],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )

        parsed = parse_json(
            raw
        )

        accepted = False
        boundary = None
        later_peel = None

        if parsed is not None:

            def read_bool(
                key,
                default=False,
            ):

                value = parsed.get(
                    key,
                    default,
                )

                if isinstance(
                    value,
                    bool,
                ):
                    return value

                return str(
                    value
                ).strip().lower() in {
                    "true",
                    "yes",
                    "1",
                }

            terminal = read_bool(
                "terminal_drop_observed"
            )

            later_peel = read_bool(
                "genuine_peeling_after_candidate"
            )

            value = parsed.get(
                "boundary_frame"
            )

            try:

                if value is not None:

                    value = int(
                        value
                    )

                    if (
                        local_start
                        <= value
                        <= local_end
                    ):

                        boundary = value

            except Exception:

                boundary = None

            accepted = (
                terminal
                and not later_peel
                and boundary is not None
            )

        del inputs
        del generated

        torch.cuda.empty_cache()

        return {
            "candidate":
                candidate,

            "local_window": [
                local_start,
                local_end,
            ],

            "displayed_frames":
                display_frames,

            "later_feature_bursts":
                later_bursts,

            "accepted":
                bool(
                    accepted
                ),

            "boundary_frame":
                boundary,

            "genuine_peeling_after_candidate":
                later_peel,

            "response":
                parsed,

            "raw":
                raw,
        }


    def resolve_terminal_drop_globally(
        first_peel_frame,
        opening_bursts,
    ):

        candidates = (
            get_terminal_decline_candidates(
                first_peel_frame
            )
        )

        validations = []

        print()
        print(
            "GLOBAL TERMINAL-DROP CANDIDATES:",
            candidates,
        )

        # --------------------------------------------------------
        # Earliest validated terminal release wins.
        # --------------------------------------------------------

        for candidate in candidates:

            result = (
                validate_global_terminal_drop(
                    candidate,
                    opening_bursts,
                )
            )

            validations.append(
                result
            )

            print(
                f"  candidate {candidate}: "
                f"accepted={result['accepted']} "
                f"boundary="
                f"{result['boundary_frame']} "
                f"later_peel="
                f"{result['genuine_peeling_after_candidate']}"
            )

            if result[
                "accepted"
            ]:

                return {
                    "boundary_frame":
                        int(
                            result[
                                "boundary_frame"
                            ]
                        ),

                    "source":
                        "pixtral_global_terminal_validation",

                    "candidate":
                        candidate,

                    "validations":
                        validations,
                }

        # --------------------------------------------------------
        # Fallback:
        # retain the existing late feature candidate and perform
        # local refinement. This is used only if no global candidate
        # survives semantic validation.
        # --------------------------------------------------------

        terminal = events[
            "terminal_release_candidate"
        ]

        fallback_candidate = int(
            terminal[
                "drop_candidate"
            ]
        )

        fallback_window = [
            int(x)
            for x in terminal[
                "drop_window"
            ]
        ]

        fallback = refine_boundary(
            DROP_REFINE_PROMPT,
            fallback_window,
            fallback_candidate,
        )

        boundary = fallback[
            "boundary_frame"
        ]

        if boundary is None:

            boundary = (
                fallback_candidate
            )

            source = (
                "feature_terminal_fallback"
            )

        else:

            source = (
                "pixtral_terminal_fallback"
            )

        return {
            "boundary_frame":
                int(boundary),

            "source":
                source,

            "candidate":
                fallback_candidate,

            "validations":
                validations,

            "fallback":
                fallback,
        }


    # ============================================================
    # OPENING BURSTS
    # ============================================================

    bursts = events[
        "opening_burst_candidates"
    ]

    if not bursts:
        raise RuntimeError(
            "No opening burst candidates."
        )

    semantic_events = []
    resolver_steps = []

    # ============================================================
    # Find first semantically confirmed peel burst
    # ============================================================

    first_confirmed_index = None
    current_opening_frame = None

    for burst_index, burst in enumerate(
        bursts
    ):

        grasp_candidate = int(
            burst[
                "preceding_grasp_candidate"
            ]
        )

        peel_candidate = int(
            burst[
                "opening_candidate"
            ]
        )

        grasp_result = refine_boundary(
            GRASP_PROMPT,
            burst[
                "preceding_grasp_window"
            ],
            grasp_candidate,
        )

        peel_result = refine_boundary(
            PEEL_PROMPT,
            burst[
                "opening_window"
            ],
            peel_candidate,
        )

        peel_frame = peel_result[
            "boundary_frame"
        ]

        resolver_steps.append(
            {
                "type":
                    "initial_burst_test",

                "burst_id":
                    burst["burst_id"],

                "grasp_result":
                    grasp_result,

                "peel_result":
                    peel_result,
            }
        )

        # If Pixtral does not see actual peeling here,
        # this feature burst is rejected.
        if peel_frame is None:
            continue

        grasp_frame = grasp_result[
            "boundary_frame"
        ]

        if grasp_frame is None:
            grasp_frame = (
                grasp_candidate
            )

            grasp_source = (
                "feature_fallback"
            )

        else:
            grasp_source = (
                "pixtral"
            )

        semantic_events.append(
            {
                "action":
                    "grasp_flaps",

                "start_frame":
                    int(grasp_frame),

                "source":
                    grasp_source,

                "burst_id":
                    burst["burst_id"],
            }
        )

        semantic_events.append(
            {
                "action":
                    "peel_apart",

                "start_frame":
                    int(peel_frame),

                "source":
                    "pixtral",

                "burst_id":
                    burst["burst_id"],
            }
        )

        first_confirmed_index = (
            burst_index
        )

        current_opening_frame = int(
            peel_frame
        )

        break

    if first_confirmed_index is None:

        raise RuntimeError(
            f"{sequence}: Pixtral rejected all opening bursts."
        )

    # ============================================================
    # Resolve every later feature burst
    # ============================================================

    drop_frame = None

    # ------------------------------------------------------------
    # Candidate-generation route.
    #
    # Optical fallback candidates deliberately carry zero
    # before/after grip distances because reliable bimanual hand
    # geometry was unavailable.
    # ------------------------------------------------------------

    candidate_mode = str(
        events.get(
            "candidate_mode",
            "",
        )
    ).strip()

    zero_distance_bursts = (
        len(bursts) > 0
        and all(
            abs(
                float(
                    burst.get(
                        "before_distance",
                        0.0,
                    )
                )
            ) < 1e-12
            and abs(
                float(
                    burst.get(
                        "after_distance",
                        0.0,
                    )
                )
            ) < 1e-12
            for burst in bursts
        )
    )

    optical_only = (
        candidate_mode
        == "optical_flow_fallback"
        or zero_distance_bursts
    )

    if not candidate_mode:
        candidate_mode = (
            "optical_flow_fallback"
            if optical_only
            else "hand_geometry+optical_flow"
        )

    # Optical-only labels are valid automatic proposals, but remain
    # lower-confidence because bimanual hand geometry was unavailable.
    review_required = bool(
        optical_only
    )

    print()
    print(
        f"CANDIDATE ROUTE: {candidate_mode}"
    )

    previous_burst = bursts[
        first_confirmed_index
    ]

    for burst_index in range(
        first_confirmed_index + 1,
        len(bursts),
    ):

        current_burst = bursts[
            burst_index
        ]

        previous_feature_onset = int(
            previous_burst[
                "opening_candidate"
            ]
        )

        current_feature_onset = int(
            current_burst[
                "opening_candidate"
            ]
        )

        gap_start = min(
            current_feature_onset - 1,
            previous_feature_onset + 3,
        )

        gap_start = max(
            first_frame,
            gap_start,
        )

        gap_end = min(
            last_frame,
            current_feature_onset - 1,
        )

        regrasp_candidate = int(
            current_burst[
                "preceding_grasp_candidate"
            ]
        )

        important = [
            previous_feature_onset,
            gap_start,
            regrasp_candidate,
            gap_end,
            current_feature_onset,
        ]

        decision_start = max(
            first_frame,
            previous_feature_onset,
        )

        decision_end = min(
            last_frame,
            current_feature_onset + 5,
        )

        result = run_pixtral(
            INTER_BURST_PROMPT,
            decision_start,
            decision_end,
            important=important,
            max_images=18,
            max_tokens=240,
        )

        parsed = result[
            "parsed"
        ]

        if parsed is None:

            decision = (
                "UNRESOLVED"
            )

            release_observed = False
            reacquisition_observed = False
            later_peel_observed = False
            terminal_release_observed = False

        else:

            decision = str(
                parsed.get(
                    "decision",
                    "UNRESOLVED",
                )
            ).strip().upper()

            # Backwards-compatible alias in case Pixtral happens to
            # use the previous wording.
            if decision == "DISTINCT_REGRASP":
                decision = "DISTINCT_REGRASP"

            def read_bool(key):

                value = parsed.get(
                    key,
                    False,
                )

                if isinstance(
                    value,
                    bool,
                ):
                    return value

                return str(
                    value
                ).strip().lower() in {
                    "true",
                    "yes",
                    "1",
                }

            release_observed = read_bool(
                "release_from_previous_peel_observed"
            )

            reacquisition_observed = read_bool(
                "reacquisition_observed"
            )

            later_peel_observed = read_bool(
                "later_sustained_peel_observed"
            )

            terminal_release_observed = read_bool(
                "terminal_release_observed"
            )

            # ----------------------------------------------------
            # Conservative semantic gating.
            #
            # Every semantic decision must agree with Pixtral's
            # explicit visual observations.
            # ----------------------------------------------------

            if decision == "DISTINCT_REGRASP":

                valid_regrasp = (
                    release_observed
                    and reacquisition_observed
                    and later_peel_observed
                )

                if not valid_regrasp:

                    if (
                        terminal_release_observed
                        and not later_peel_observed
                    ):
                        decision = "TERMINAL_DROP"

                    elif later_peel_observed:
                        decision = "CONTINUOUS_PEEL"

                    else:
                        decision = "UNRESOLVED"

            elif decision == "TERMINAL_DROP":

                valid_drop = (
                    terminal_release_observed
                    and not later_peel_observed
                )

                if not valid_drop:

                    if later_peel_observed:
                        decision = "CONTINUOUS_PEEL"
                    else:
                        decision = "UNRESOLVED"

            elif decision == "CONTINUOUS_PEEL":

                # This is the important correction:
                #
                # A feature burst cannot be called continued peeling
                # unless Pixtral explicitly sees genuine peeling at
                # or after that later burst.
                if later_peel_observed:

                    decision = "CONTINUOUS_PEEL"

                elif terminal_release_observed:

                    decision = "TERMINAL_DROP"

                else:

                    decision = "UNRESOLVED"

            else:

                decision = "UNRESOLVED"

        step = {
            "type":
                "inter_burst_resolution",

            "from_burst":
                previous_burst[
                    "burst_id"
                ],

            "to_burst":
                current_burst[
                    "burst_id"
                ],

            "decision":
                decision,

            "grounding": {
                "release_from_previous_peel_observed":
                    release_observed,

                "reacquisition_observed":
                    reacquisition_observed,

                "later_sustained_peel_observed":
                    later_peel_observed,

                "terminal_release_observed":
                    terminal_release_observed,
            },

            "result":
                result,
        }

        # --------------------------------------------------------
        # Same peel continues.
        # --------------------------------------------------------

        if decision == "CONTINUOUS_PEEL":

            step[
                "semantic_action"
            ] = (
                "remain_peel_apart"
            )

            resolver_steps.append(
                step
            )

            # The later burst has now been semantically confirmed
            # as belonging to real ongoing peeling.
            previous_burst = (
                current_burst
            )

            continue

        # --------------------------------------------------------
        # Genuine regrasp and another peel.
        # --------------------------------------------------------

        if decision == "REGRASP_THEN_PEEL":

            grasp_result = (
                refine_boundary(
                    GRASP_PROMPT,
                    current_burst[
                        "preceding_grasp_window"
                    ],
                    regrasp_candidate,
                )
            )

            peel_result = (
                refine_boundary(
                    PEEL_PROMPT,
                    current_burst[
                        "opening_window"
                    ],
                    current_feature_onset,
                )
            )

            regrasp_frame = (
                grasp_result[
                    "boundary_frame"
                ]
            )

            new_peel_frame = (
                peel_result[
                    "boundary_frame"
                ]
            )

            if regrasp_frame is None:

                regrasp_frame = (
                    regrasp_candidate
                )

                grasp_source = (
                    "feature_fallback"
                )

            else:

                grasp_source = (
                    "pixtral"
                )

            if new_peel_frame is None:

                # Semantic decision said regrasp+peel, but
                # exact peel refinement failed. Keep feature
                # centre and flag provenance.
                new_peel_frame = (
                    current_feature_onset
                )

                peel_source = (
                    "feature_fallback"
                )

            else:

                peel_source = (
                    "pixtral"
                )

            if (
                regrasp_frame
                >= new_peel_frame
            ):

                review_required = True

                step[
                    "warning"
                ] = (
                    "invalid regrasp/peel ordering"
                )

            semantic_events.append(
                {
                    "action":
                        "grasp_flaps",

                    "start_frame":
                        int(
                            regrasp_frame
                        ),

                    "source":
                        grasp_source,

                    "burst_id":
                        current_burst[
                            "burst_id"
                        ],
                }
            )

            semantic_events.append(
                {
                    "action":
                        "peel_apart",

                    "start_frame":
                        int(
                            new_peel_frame
                        ),

                    "source":
                        peel_source,

                    "burst_id":
                        current_burst[
                            "burst_id"
                        ],
                }
            )

            step[
                "refined_grasp"
            ] = grasp_result

            step[
                "refined_peel"
            ] = peel_result

            resolver_steps.append(
                step
            )

            current_opening_frame = int(
                new_peel_frame
            )

            previous_burst = (
                current_burst
            )

            continue

        # --------------------------------------------------------
        # Terminal drop occurs before proposed later burst.
        # --------------------------------------------------------

        if decision == "TERMINAL_DROP":

            approx = None

            if parsed is not None:

                value = parsed.get(
                    "approx_drop_frame"
                )

                try:
                    if value is not None:
                        approx = int(value)
                except Exception:
                    pass

            if (
                approx is None
                or approx < gap_start
                or approx > gap_end
            ):

                # Feature fallback for the refinement centre:
                # first cross-modal decline inside the interval.
                gap_df = df[
                    (
                        df["frame"]
                        >= gap_start
                    )
                    &
                    (
                        df["frame"]
                        <= gap_end
                    )
                ]

                decline = gap_df[
                    (
                        gap_df[
                            "separation_trend"
                        ] < 0
                    )
                    &
                    (
                        gap_df[
                            "opposing_horizontal_trend"
                        ] < 0
                    )
                    &
                    (
                        gap_df[
                            "positive_divergence_trend"
                        ] < 0
                    )
                ]

                if len(decline):

                    approx = int(
                        decline.iloc[0][
                            "frame"
                        ]
                    )

                else:

                    approx = int(
                        (
                            gap_start
                            + gap_end
                        )
                        // 2
                    )

            refine_start = max(
                gap_start,
                approx - 4,
            )

            refine_end = min(
                gap_end,
                approx + 4,
            )

            drop_result = (
                refine_boundary(
                    DROP_REFINE_PROMPT,
                    [
                        refine_start,
                        refine_end,
                    ],
                    approx,
                )
            )

            drop_frame = drop_result[
                "boundary_frame"
            ]

            if drop_frame is None:

                drop_frame = approx
                drop_source = (
                    "feature_fallback"
                )

            else:

                drop_source = (
                    "pixtral"
                )

            semantic_events.append(
                {
                    "action":
                        "drop_contents",

                    "start_frame":
                        int(drop_frame),

                    "source":
                        drop_source,
                }
            )

            step[
                "drop_refinement"
            ] = drop_result

            resolver_steps.append(
                step
            )

            # Terminal. Ignore every later feature burst.
            break

        # --------------------------------------------------------
        # Unresolved semantic interval.
        # --------------------------------------------------------

        review_required = True

        step[
            "warning"
        ] = (
            "Pixtral did not return a valid "
            "inter-burst semantic decision"
        )

        resolver_steps.append(
            step
        )

        previous_burst = (
            current_burst
        )

    # ============================================================
    # DETERMINISTIC TERMINAL RELEASE
    #
    # Pixtral is deliberately NOT used for drop_contents.
    #
    # Only semantically confirmed peel_apart events reset the
    # opening cycle. CONTINUOUS_PEEL feature bursts do not.
    #
    # After the last supported opening evidence:
    #
    #   1. find the strongest BIMANUAL separation peak
    #   2. search forward for persistent cross-modal decline
    #   3. use the first persistent decline as drop_contents
    #
    # This allows:
    #
    #   small package:
    #       grasp -> peel -> drop
    #
    #   large package:
    #       grasp -> peel -> regrasp -> peel -> drop
    #
    # without allowing temporary pauses before a confirmed later
    # peel cycle to become terminal release.
    # ============================================================

    # Remove any provisional drop decisions generated by the
    # semantic inter-burst resolver. Drop is handled deterministically.
    semantic_events = [
        event
        for event
        in semantic_events
        if event[
            "action"
        ] != "drop_contents"
    ]

    # ============================================================
    # OPENING HORIZON
    #
    # Feature bursts are proposals, not automatically new actions.
    #
    # A later burst extends the active opening horizon if:
    #
    #   1. genuine peeling is still visible and no persistent
    #      terminal-like decline occurred before that burst,
    #
    # OR
    #
    #   2. a persistent decline occurred, BUT the package/hand
    #      geometry substantially reset to a closed configuration,
    #      two-hand grasp evidence exists, and genuine peeling is
    #      visible afterwards.
    #
    # This lets large packages perform:
    #
    #   peel -> reposition/reacquire -> peel -> ...
    #
    # while rejecting opening-like hand motion after release.
    # ============================================================

    semantic_peels = [
        event
        for event
        in semantic_events
        if event[
            "action"
        ] == "peel_apart"
    ]

    if not semantic_peels:

        raise RuntimeError(
            f"{sequence}: no confirmed peel event "
            f"for deterministic drop detection"
        )

    first_peel_frame = int(
        semantic_peels[0][
            "start_frame"
        ]
    )

    # ------------------------------------------------------------
    # Map Pixtral inter-burst decisions by destination burst.
    # ------------------------------------------------------------

    inter_steps = {}

    for step in resolver_steps:

        if (
            step.get(
                "type"
            )
            == "inter_burst_resolution"
        ):

            inter_steps[
                int(
                    step[
                        "to_burst"
                    ]
                )
            ] = step

    # ------------------------------------------------------------
    # Exact semantic peel frame, when a DISTINCT_REGRASP generated
    # a real new peel_apart event.
    # ------------------------------------------------------------

    semantic_peel_by_burst = {}

    for event in semantic_events:

        if (
            event[
                "action"
            ] == "peel_apart"
            and event.get(
                "burst_id"
            ) is not None
        ):

            semantic_peel_by_burst[
                int(
                    event[
                        "burst_id"
                    ]
                )
            ] = int(
                event[
                    "start_frame"
                ]
            )

    if first_confirmed_index is None:

        raise RuntimeError(
            f"{sequence}: no first confirmed opening burst"
        )

    supported_burst = bursts[
        first_confirmed_index
    ]

    last_opening_evidence = (
        first_peel_frame
    )

    opening_horizon = [
        {
            "burst_id":
                int(
                    supported_burst[
                        "burst_id"
                    ]
                ),

            "opening_frame":
                first_peel_frame,

            "supported":
                True,

            "reason":
                "initial_confirmed_peel",
        }
    ]

    # Relative reset criterion:
    #
    # new pre-pull hand distance <= 70% of previous opened distance
    # indicates that the manipulation has physically returned toward
    # a closed/reacquired configuration.
    RESET_RATIO_THRESHOLD = 0.70


    def persistent_decline_between(
        start_frame,
        end_frame,
    ):

        region = df[
            (
                df[
                    "frame"
                ] > int(
                    start_frame
                ) + 3
            )
            &
            (
                df[
                    "frame"
                ] < int(
                    end_frame
                )
            )
        ].copy()

        if len(region) == 0:

            return (
                False,
                None,
            )

        if optical_only:

            region[
                "decline"
            ] = (
                (
                    region[
                        "opposing_horizontal_trend"
                    ] < 0
                )
                &
                (
                    region[
                        "positive_divergence_trend"
                    ] < 0
                )
            )

        else:

            region[
                "decline"
            ] = (
                (
                    region[
                        "separation_trend"
                    ] < 0
                )
                &
                (
                    region[
                        "opposing_horizontal_trend"
                    ] < 0
                )
                &
                (
                    region[
                        "positive_divergence_trend"
                    ] < 0
                )
            )

        for _, row in region.iterrows():

            if not bool(
                row[
                    "decline"
                ]
            ):
                continue

            frame = int(
                row[
                    "frame"
                ]
            )

            future = region[
                (
                    region[
                        "frame"
                    ] >= frame
                )
                &
                (
                    region[
                        "frame"
                    ] <= frame + 4
                )
            ]

            if len(future) == 0:
                continue

            required = min(
                3,
                len(future),
            )

            if int(
                future[
                    "decline"
                ].sum()
            ) >= required:

                return (
                    True,
                    frame,
                )

        return (
            False,
            None,
        )


    # ------------------------------------------------------------
    # Walk through every later feature burst chronologically.
    # ------------------------------------------------------------

    for burst_index in range(
        first_confirmed_index + 1,
        len(bursts),
    ):

        current = bursts[
            burst_index
        ]

        burst_id = int(
            current[
                "burst_id"
            ]
        )

        current_onset = int(
            current[
                "opening_candidate"
            ]
        )

        previous_onset = int(
            supported_burst[
                "opening_candidate"
            ]
        )

        step = inter_steps.get(
            burst_id,
            {},
        )

        decision = str(
            step.get(
                "decision",
                "UNRESOLVED",
            )
        ).upper()

        grounding = step.get(
            "grounding",
            {},
        )

        later_peel = bool(
            grounding.get(
                "later_sustained_peel_observed",
                False,
            )
        )

        has_decline, decline_frame = (
            persistent_decline_between(
                previous_onset,
                current_onset,
            )
        )

        previous_open_distance = float(
            supported_burst.get(
                "after_distance",
                0.0,
            )
        )

        current_closed_distance = float(
            current.get(
                "before_distance",
                0.0,
            )
        )

        if previous_open_distance > 1e-8:

            reset_ratio = (
                current_closed_distance
                / previous_open_distance
            )

        else:

            reset_ratio = float(
                "inf"
            )

        strong_reset = (
            reset_ratio
            <= RESET_RATIO_THRESHOLD
        )

        # Candidate detector only provides preceding_worst_pinch
        # when an actual two-hand grasp observation exists.
        bimanual_reacquisition = (
            current.get(
                "preceding_worst_pinch"
            )
            is not None
        )

        explicitly_regrasped = (
            decision
            == "DISTINCT_REGRASP"
        )

        # --------------------------------------------------------
        # Decide whether this later burst represents genuine
        # package opening.
        #
        # Normal route:
        #     use hand geometry + optical decline + Pixtral.
        #
        # Optical-only route:
        #     grip-distance reset is unavailable by design.
        #     A later burst extends the opening horizon only when
        #     Pixtral explicitly confirms sustained genuine peeling.
        # --------------------------------------------------------

        if optical_only:

            if (
                later_peel
                and not has_decline
            ):

                supported = True

                reason = (
                    "optical_semantic_continuous_peel"
                )

            elif (
                later_peel
                and has_decline
                and explicitly_regrasped
            ):

                supported = True

                reason = (
                    "optical_semantic_regrasp_then_peel"
                )

            else:

                supported = False

                if (
                    later_peel
                    and has_decline
                    and not explicitly_regrasped
                ):

                    reason = (
                        "optical_decline_without_confirmed_regrasp"
                    )

                elif has_decline:

                    reason = (
                        "optical_decline_without_semantic_later_peel"
                    )

                else:

                    reason = (
                        "no_semantic_later_peel_support"
                    )

        else:

            if explicitly_regrasped:

                supported = True

                reason = (
                    "semantic_distinct_regrasp"
                )

            elif (
                later_peel
                and not has_decline
            ):

                supported = True

                reason = (
                    "continuous_peel_no_persistent_decline"
                )

            elif (
                later_peel
                and has_decline
                and strong_reset
                and bimanual_reacquisition
            ):

                supported = True

                reason = (
                    "renewed_peel_after_reacquisition"
                )

            else:

                supported = False

                if (
                    has_decline
                    and not strong_reset
                ):

                    reason = (
                        "persistent_decline_without_geometry_reset"
                    )

                elif (
                    has_decline
                    and not bimanual_reacquisition
                ):

                    reason = (
                        "persistent_decline_without_bimanual_reacquisition"
                    )

                elif not later_peel:

                    reason = (
                        "no_semantic_later_peel_support"
                    )

                else:

                    reason = (
                        "insufficient_opening_support"
                    )

        opening_frame = (
            semantic_peel_by_burst.get(
                burst_id,
                current_onset,
            )
        )

        opening_horizon.append(
            {
                "burst_id":
                    burst_id,

                "opening_frame":
                    int(
                        opening_frame
                    ),

                "feature_onset":
                    current_onset,

                "supported":
                    bool(
                        supported
                    ),

                "reason":
                    reason,

                "semantic_decision":
                    decision,

                "later_peel":
                    later_peel,

                "persistent_decline_before_burst":
                    bool(
                        has_decline
                    ),

                "decline_frame":
                    decline_frame,

                "previous_after_distance":
                    previous_open_distance,

                "current_before_distance":
                    current_closed_distance,

                "reset_ratio":
                    (
                        None
                        if not np.isfinite(
                            reset_ratio
                        )
                        else float(
                            reset_ratio
                        )
                    ),

                "reset_threshold":
                    RESET_RATIO_THRESHOLD,

                "strong_reset":
                    bool(
                        strong_reset
                    ),

                "bimanual_reacquisition":
                    bool(
                        bimanual_reacquisition
                    ),
            }
        )

        if supported:

            last_opening_evidence = int(
                opening_frame
            )

            supported_burst = (
                current
            )

        elif has_decline:

            # Once there is a persistent collapse and the next
            # opening proposal fails the reset/reacquisition test,
            # later bursts are treated as post-release motion.
            break

    last_confirmed_peel = int(
        last_opening_evidence
    )

    print()
    print(
        "OPENING HORIZON"
    )

    for item in opening_horizon:

        if (
            item[
                "reason"
            ]
            == "initial_confirmed_peel"
        ):

            print(
                f"  burst "
                f"{item['burst_id']}: "
                f"SUPPORTED "
                f"opening="
                f"{item['opening_frame']} "
                f"[initial peel]"
            )

            continue

        print(
            f"  burst "
            f"{item['burst_id']}: "
            f"{'SUPPORTED' if item['supported'] else 'REJECTED '} "
            f"opening="
            f"{item['opening_frame']} "
            f"decline="
            f"{item['persistent_decline_before_burst']} "
            f"reset_ratio="
            f"{item['reset_ratio']} "
            f"two_hand="
            f"{item['bimanual_reacquisition']} "
            f"later_peel="
            f"{item['later_peel']} "
            f"[{item['reason']}]"
        )

    print(
        f"  last supported opening evidence: "
        f"{last_confirmed_peel}"
    )

    # ------------------------------------------------------------
    # Search sufficiently after the peel onset so that the onset
    # itself cannot become the separation maximum trivially.
    # ------------------------------------------------------------

    peak_search = df[
        df[
            "frame"
        ] >= last_confirmed_peel + 3
    ].copy()

    if len(
        peak_search
    ) == 0:

        raise RuntimeError(
            f"{sequence}: no frames after last confirmed peel"
        )

    # Optical candidate generation means hand separation was not
    # reliable enough to define the opening events. Do not reintroduce
    # sparse hand separation at the terminal stage.
    if optical_only:
        peak_search[
            "separating_speed_mean5"
        ] = np.nan

    # ------------------------------------------------------------
    # Prefer a peak while BOTH hands are detected.
    #
    # Post-release one-hand movement can otherwise create a large
    # false separation velocity.
    # ------------------------------------------------------------

    bimanual_peak_search = (
        peak_search[
            (
                peak_search[
                    "detected_hands"
                ] >= 2
            )
            &
            (
                peak_search[
                    "separating_speed_mean5"
                ].notna()
            )
        ]
    )

    use_optical_terminal = False

    if len(
        bimanual_peak_search
    ):

        peak_pool = (
            bimanual_peak_search
        )

        peak_source = (
            "bimanual_separation"
        )

        peak_metric = (
            "separating_speed_mean5"
        )

    else:

        separation_pool = (
            peak_search.dropna(
                subset=[
                    "separating_speed_mean5"
                ]
            )
        )

        if len(
            separation_pool
        ):

            peak_pool = (
                separation_pool
            )

            peak_source = (
                "separation_fallback"
            )

            peak_metric = (
                "separating_speed_mean5"
            )

        else:

            # ----------------------------------------------------
            # No hand separation signal at all.
            #
            # Use sequence-relative optical opening strength.
            # ----------------------------------------------------

            peak_pool = (
                peak_search.dropna(
                    subset=[
                        "opposing_horizontal",
                        "positive_divergence",
                    ]
                )
                .copy()
            )

            if len(
                peak_pool
            ) == 0:

                raise RuntimeError(
                    f"{sequence}: no hand or optical opening "
                    f"signal after peel"
                )

            peak_pool[
                "_optical_opening_score"
            ] = (
                0.55
                * peak_pool[
                    "opposing_horizontal"
                ].rank(
                    pct=True
                )
                +
                0.45
                * peak_pool[
                    "positive_divergence"
                ].rank(
                    pct=True
                )
            )

            peak_source = (
                "optical_opening_fallback"
            )

            peak_metric = (
                "_optical_opening_score"
            )

            use_optical_terminal = True

            # Lower-confidence route: retain provenance and make
            # this sequence easy to identify later.
            review_required = True

    peak_idx = (
        peak_pool[
            peak_metric
        ].idxmax()
    )

    separation_peak_frame = int(
        df.loc[
            peak_idx,
            "frame",
        ]
    )

    separation_peak_value = float(
        peak_pool.loc[
            peak_idx,
            peak_metric,
        ]
    )

    # ------------------------------------------------------------
    # Search for the first PERSISTENT cross-modal decline after
    # the strongest active-opening peak.
    #
    # A single negative frame is not sufficient.
    # ------------------------------------------------------------

    decline_search = df[
        df[
            "frame"
        ] > separation_peak_frame
    ].copy()

    if use_optical_terminal:

        decline_search[
            "cross_modal_decline"
        ] = (
            (
                decline_search[
                    "opposing_horizontal_trend"
                ] < 0
            )
            &
            (
                decline_search[
                    "positive_divergence_trend"
                ] < 0
            )
        )

    else:

        decline_search[
            "cross_modal_decline"
        ] = (
            (
                decline_search[
                    "separation_trend"
                ] < 0
            )
            &
            (
                decline_search[
                    "opposing_horizontal_trend"
                ] < 0
            )
            &
            (
                decline_search[
                    "positive_divergence_trend"
                ] < 0
            )
        )

    drop_candidate = None

    persistence_details = None

    # Require at least 3 declining frames in a 5-frame
    # forward window. This suppresses brief pauses.
    for idx, row in decline_search.iterrows():

        if not bool(
            row[
                "cross_modal_decline"
            ]
        ):
            continue

        frame = int(
            row[
                "frame"
            ]
        )

        future = decline_search[
            (
                decline_search[
                    "frame"
                ] >= frame
            )
            &
            (
                decline_search[
                    "frame"
                ] <= frame + 4
            )
        ]

        if len(
            future
        ) == 0:
            continue

        decline_count = int(
            future[
                "cross_modal_decline"
            ].sum()
        )

        required = min(
            3,
            len(
                future
            ),
        )

        if (
            decline_count
            >= required
        ):

            drop_candidate = (
                frame
            )

            persistence_details = {
                "window": [
                    frame,
                    int(
                        future[
                            "frame"
                        ].max()
                    ),
                ],

                "declining_frames":
                    decline_count,

                "required":
                    required,
            }

            break

    # ------------------------------------------------------------
    # Graceful fallback:
    # hand-separation decline alone, but still persistent.
    # ------------------------------------------------------------

    if (
        drop_candidate is None
        and not use_optical_terminal
    ):

        for idx, row in decline_search.iterrows():

            frame = int(
                row[
                    "frame"
                ]
            )

            future = decline_search[
                (
                    decline_search[
                        "frame"
                    ] >= frame
                )
                &
                (
                    decline_search[
                        "frame"
                    ] <= frame + 4
                )
            ]

            if len(
                future
            ) == 0:
                continue

            hand_declines = (
                future[
                    "separation_trend"
                ] < 0
            )

            required = min(
                3,
                len(
                    future
                ),
            )

            if int(
                hand_declines.sum()
            ) >= required:

                drop_candidate = (
                    frame
                )

                persistence_details = {
                    "window": [
                        frame,
                        int(
                            future[
                                "frame"
                            ].max()
                        ),
                    ],

                    "declining_frames":
                        int(
                            hand_declines.sum()
                        ),

                    "required":
                        required,

                    "fallback":
                        "hand_separation_only",
                }

                break

    # ------------------------------------------------------------
    # Last-resort fallback.
    # ------------------------------------------------------------

    if drop_candidate is None:

        drop_candidate = int(
            events[
                "terminal_release_candidate"
            ][
                "drop_candidate"
            ]
        )

        persistence_details = {
            "fallback":
                "event_detector_terminal_candidate"
        }

        drop_source = (
            "event_detector_fallback"
        )

    else:

        drop_source = (
            "deterministic_persistent_decline"
        )

    drop_frame = int(
        drop_candidate
    )

    # ------------------------------------------------------------
    # Terminal means terminal.
    #
    # Remove any semantically proposed regrasp/peel event that
    # happens AFTER the deterministic release boundary.
    # ------------------------------------------------------------

    semantic_events = [
        event
        for event
        in semantic_events
        if int(
            event[
                "start_frame"
            ]
        ) < drop_frame
    ]

    semantic_events.append(
        {
            "action":
                "drop_contents",

            "start_frame":
                drop_frame,

            "source":
                drop_source,
        }
    )

    resolver_steps.append(
        {
            "type":
                "deterministic_terminal_drop",

            "last_confirmed_peel":
                last_confirmed_peel,

            "opening_horizon":
                opening_horizon,

            "separation_peak_frame":
                separation_peak_frame,

            "separation_peak_value":
                separation_peak_value,

            "peak_source":
                peak_source,

            "drop_frame":
                drop_frame,

            "drop_source":
                drop_source,

            "persistence":
                persistence_details,
        }
    )

    print()
    print(
        "DETERMINISTIC TERMINAL DROP"
    )

    print(
        f"  last supported opening evidence: "
        f"{last_confirmed_peel}"
    )

    print(
        f"  active separation peak:    "
        f"{separation_peak_frame} "
        f"({separation_peak_value:.6f}) "
        f"[{peak_source}]"
    )

    print(
        f"  terminal decline onset:    "
        f"{drop_frame}"
    )

    print(
        f"  source:                    "
        f"{drop_source}"
    )

    print(
        f"  persistence:               "
        f"{persistence_details}"
    )

    # ============================================================
    # Validate semantic chronology
    # ============================================================

    semantic_events = sorted(
        semantic_events,
        key=lambda x:
            x["start_frame"],
    )

    terminal_seen = False

    previous_frame = None

    for event in semantic_events:

        frame = int(
            event[
                "start_frame"
            ]
        )

        if (
            previous_frame is not None
            and frame <= previous_frame
        ):
            review_required = True

        if terminal_seen:
            review_required = True

        if (
            event["action"]
            == "drop_contents"
        ):
            terminal_seen = True

        previous_frame = frame

    # ============================================================
    # Convenience boundaries
    # ============================================================

    first_grasp = next(
        (
            x["start_frame"]
            for x in semantic_events
            if x["action"]
            == "grasp_flaps"
        ),
        None,
    )

    first_peel = next(
        (
            x["start_frame"]
            for x in semantic_events
            if x["action"]
            == "peel_apart"
        ),
        None,
    )

    # ============================================================
    # OUTPUT
    # ============================================================

    output = {
        "sequence":
            sequence,

        "method":
            (
                "mediapipe_hand_geometry"
                "+stabilised_optical_flow"
                "+pixtral_event_resolution"
            ),

        "frame_range": [
            first_frame,
            last_frame,
        ],

        "coverage":
            events.get(
                "coverage",
                {},
            ),

        "boundaries": {
            "grasp_flaps":
                first_grasp,

            "peel_apart":
                first_peel,

            "drop_contents":
                int(drop_frame),
        },

        "semantic_events":
            semantic_events,

        "resolver_steps":
            resolver_steps,

        "feature_burst_count":
            len(bursts),

        "review_required":
            bool(
                review_required
            ),
    }

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
    print("=" * 100)
    print(
        f"SEMANTIC BT EVENT RESOLUTION: {sequence}"
    )
    print("=" * 100)

    print(
        f"feature bursts proposed: "
        f"{len(bursts)}"
    )

    print()

    for event in semantic_events:

        print(
            f"{event['start_frame']:4d}  "
            f"{event['action']:16s} "
            f"source={event['source']}"
        )

    print()
    print(
        f"review_required: "
        f"{review_required}"
    )

    print()

    for step in resolver_steps:

        if (
            step["type"]
            == "inter_burst_resolution"
        ):

            grounding = step.get(
                "grounding",
                {},
            )

            print(
                f"burst "
                f"{step['from_burst']} -> "
                f"{step['to_burst']}: "
                f"{step['decision']}  "
                f"[release="
                f"{grounding.get('release_from_previous_peel_observed')} "
                f"reacquire="
                f"{grounding.get('reacquisition_observed')} "
                f"later_peel="
                f"{grounding.get('later_sustained_peel_observed')} "
                f"terminal="
                f"{grounding.get('terminal_release_observed')}]"
            )

    print()
    print(
        f"Wrote {out_file}"
    )


if __name__ == "__main__":
    main()
