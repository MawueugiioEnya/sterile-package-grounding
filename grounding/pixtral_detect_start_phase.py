#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from pixtral_labeler import load_model


COARSE_PROMPT = """
You are analysing the BEGINNING of a sterile-package-opening demonstration.

The Behaviour Tree may begin in one of two ways:

A) pick_up_package is visible:
   The package starts not yet securely held, and the nurse acquires/lifts it.
   After pickup is complete, the nurse begins locate_flaps.

B) pickup is NOT visible:
   The recording starts with the package already securely held.
   In that case the sequence begins directly in locate_flaps.

locate_flaps means the package is already held while the nurse searches,
touches, repositions, or acquires the peelable lips.

Your task:

1. Decide whether pick_up_package is actually visible at the beginning.
2. If pickup is visible, estimate the FIRST source frame where pickup has
   completed and locate_flaps has begun.
3. If the package is already securely held from the first frame, report
   pickup_observed=false and locate_start equal to the first source frame.

Do NOT confuse searching for package flaps with package pickup.

Return ONLY valid JSON:

{
  "pickup_observed": true_or_false,
  "approx_locate_start": integer,
  "visual_evidence": "short phrase"
}

No Markdown fences.
""".strip()


REFINE_PROMPT = """
You are refining the transition:

pick_up_package -> locate_flaps

pick_up_package:
The nurse is acquiring, lifting, or establishing secure control of the
whole sterile package.

locate_flaps:
The package is already securely held and the nurse has changed to searching
for, touching, repositioning around, or acquiring the peelable lips.

Choose the FIRST source frame where the package is securely held and the
behaviour has entered locate_flaps.

The images are chronological.

Return ONLY valid JSON:

{
  "locate_start": integer,
  "visual_evidence": "short phrase"
}

No Markdown fences.
""".strip()


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


def crop_image(image, fraction=0.90):
    h, w = image.shape[:2]

    rh = int(h * fraction)
    rw = int(w * fraction)

    y0 = (h - rh) // 2
    x0 = (w - rw) // 2

    return image[
        y0:y0 + rh,
        x0:x0 + rw
    ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hand_csv",
        required=True,
    )

    parser.add_argument(
        "--refined",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    hand_file = Path(args.hand_csv)
    refined_file = Path(args.refined)
    out_file = Path(args.out)

    hands = pd.read_csv(hand_file)

    refined = json.loads(
        refined_file.read_text()
    )

    sequence = refined["sequence"]

    hands = hands.sort_values(
        "frame"
    ).reset_index(drop=True)

    first_frame = int(
        hands["frame"].min()
    )

    last_frame = int(
        hands["frame"].max()
    )

    grasp_start = int(
        refined["boundaries"]["grasp_flaps"]
    )

    by_frame = {
        int(row["frame"]): row
        for _, row in hands.iterrows()
    }

    crop_dir = (
        Path("pixtral_start_crops")
        / sequence
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def get_path(frame):
        row = by_frame[frame]
        return Path(row["frame_path"])

    def make_crop(frame):
        path = get_path(frame)

        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"Cannot read {path}"
            )

        crop = crop_image(
            image,
            0.90,
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

    model, processor = load_model(
        load_4bit=False
    )

    # ============================================================
    # COARSE START INSPECTION
    #
    # Never inspect beyond grasp onset.
    # Sample at most 9 images from the beginning.
    # ============================================================

    coarse_end = min(
        grasp_start - 1,
        first_frame + 30,
    )

    available = [
        f for f in range(
            first_frame,
            coarse_end + 1
        )
        if f in by_frame
    ]

    if not available:
        raise RuntimeError(
            "No early frames available."
        )

    sample_count = min(
        9,
        len(available),
    )

    indices = np.linspace(
        0,
        len(available) - 1,
        sample_count,
    ).round().astype(int)

    sampled_frames = []

    for i in indices:
        frame = available[int(i)]

        if frame not in sampled_frames:
            sampled_frames.append(frame)

    content = [
        {
            "type": "text",
            "text": COARSE_PROMPT,
        },
        {
            "type": "text",
            "text": (
                f"SEQUENCE: {sequence}\n"
                f"FIRST SOURCE FRAME: {first_frame}\n"
                f"KNOWN LATER GRASP CANDIDATE: {grasp_start}\n\n"
                f"The later grasp number only limits the beginning region. "
                f"It does NOT tell you whether pickup is present."
            ),
        },
        {
            "type": "text",
            "text": (
                "Chronological images from the beginning follow."
            ),
        },
    ]

    for frame in sampled_frames:

        content.append(
            {
                "type": "text",
                "text": f"[SOURCE FRAME {frame}]",
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
            max_new_tokens=140,
            do_sample=False,
        )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    coarse_raw = processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    coarse = parse_json(
        coarse_raw
    )

    if coarse is None:
        raise RuntimeError(
            f"Could not parse coarse Pixtral result:\n{coarse_raw}"
        )

    pickup_observed = bool(
        coarse.get(
            "pickup_observed",
            False,
        )
    )

    # ============================================================
    # NO PICKUP
    # ============================================================

    if not pickup_observed:

        locate_start = first_frame

        refinement = None
        refinement_raw = None

    # ============================================================
    # PICKUP PRESENT: refine boundary
    # ============================================================

    else:

        approx = coarse.get(
            "approx_locate_start",
            first_frame,
        )

        try:
            approx = int(approx)
        except Exception:
            approx = first_frame

        approx = max(
            first_frame,
            min(
                approx,
                grasp_start - 1,
            ),
        )

        radius = 4

        refine_start = max(
            first_frame,
            approx - radius,
        )

        refine_end = min(
            grasp_start - 1,
            approx + radius,
        )

        refine_frames = [
            f for f in range(
                refine_start,
                refine_end + 1
            )
            if f in by_frame
        ]

        content = [
            {
                "type": "text",
                "text": REFINE_PROMPT,
            },
            {
                "type": "text",
                "text": (
                    f"SEQUENCE: {sequence}\n"
                    f"CANDIDATE WINDOW: "
                    f"{refine_start}-{refine_end}\n\n"
                    f"Choose the FIRST frame where pickup is complete "
                    f"and locate_flaps has begun."
                ),
            },
        ]

        for frame in refine_frames:

            content.append(
                {
                    "type": "text",
                    "text": f"[SOURCE FRAME {frame}]",
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

        inputs2 = processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs2 = inputs2.to(
            model.device
        )

        with torch.inference_mode():
            generated2 = model.generate(
                **inputs2,
                max_new_tokens=120,
                do_sample=False,
            )

        input_length2 = (
            inputs2["input_ids"].shape[1]
        )

        refinement_raw = processor.batch_decode(
            generated2[:, input_length2:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        refinement = parse_json(
            refinement_raw
        )

        locate_start = approx

        if refinement is not None:

            value = refinement.get(
                "locate_start"
            )

            try:
                value = int(value)

                if (
                    refine_start
                    <= value
                    <= refine_end
                ):
                    locate_start = value

            except Exception:
                pass

    # ============================================================
    # SANITY
    # ============================================================

    locate_start = max(
        first_frame,
        min(
            int(locate_start),
            grasp_start - 1,
        ),
    )

    output = {
        "sequence": sequence,

        "pickup_observed":
            pickup_observed,

        "first_frame":
            first_frame,

        "locate_flaps_start":
            locate_start,

        "grasp_flaps_start":
            grasp_start,

        "coarse_sampled_frames":
            sampled_frames,

        "coarse_response":
            coarse,

        "coarse_raw":
            coarse_raw,

        "refinement_response":
            refinement,

        "refinement_raw":
            refinement_raw,
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

    print()
    print("=" * 90)
    print(
        f"START-PHASE RESULT: {sequence}"
    )
    print("=" * 90)

    print(
        f"pickup observed:   "
        f"{pickup_observed}"
    )

    print(
        f"locate_flaps start:"
        f" {locate_start}"
    )

    print(
        f"grasp_flaps start: "
        f"{grasp_start}"
    )

    if pickup_observed:
        print(
            f"pick_up_package:    "
            f"{first_frame}-{locate_start - 1}"
        )
    else:
        print(
            "pick_up_package:    ABSENT"
        )

    print()
    print("COARSE RAW:")
    print(coarse_raw)

    if refinement_raw is not None:
        print()
        print("REFINEMENT RAW:")
        print(refinement_raw)

    print()
    print(
        f"Wrote {out_file}"
    )


if __name__ == "__main__":
    main()
