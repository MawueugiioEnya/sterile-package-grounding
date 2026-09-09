#!/usr/bin/env python3

"""
Pixtral visual-state labelling for sterile-package-opening videos.

The VLM does NOT classify actions and does NOT reason about whether an
event happened historically.

For each TARGET frame it assesses concrete CURRENT visual states.

Python later detects stable NO -> YES changes in these states.
"""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from vocabulary import CONDITIONS


MODEL_ID = "mistral-community/pixtral-12b"

STATES = [
    "package_held",
    "both_lips_grasped",
    "package_separated",
    "contents_leaving",
]

VALID_STATUS = {
    "yes",
    "no",
    "uncertain",
}


def check_cuda():

    print("=" * 70)
    print("PIXTRAL CUDA CHECK")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(
        "CUDA_VISIBLE_DEVICES: "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("FATAL: CUDA unavailable. Refusing CPU execution.")
        sys.exit(21)

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    free_mem, total_mem = torch.cuda.mem_get_info(0)

    print(f"GPU free memory: {free_mem / 1024**3:.2f} GB")
    print(f"GPU total memory: {total_mem / 1024**3:.2f} GB")
    print("=" * 70)


def build_prompt(object_name=None):

    object_context = (
        f"The sterile package contains a {object_name}."
        if object_name
        else ""
    )

    return f"""
You are analysing a short chronological sequence of images from ONE
sterile-package-opening procedure performed by a nurse.

{object_context}

Your task is NOT action recognition.
Your task is NOT to decide whether something happened earlier in the video.

Assess the CURRENT VISUAL STATE at the image marked [TARGET FRAME].

The earlier images are context only.
The TARGET FRAME is the frame that must be labelled.

The recording may start with the package already held.
Therefore do not assume pickup is visible.

Assess these FOUR visual states independently.

1. package_held

YES:
The package is visibly supported/held by the nurse's hand or hands at TARGET.

NO:
The package is clearly resting unsupported on the table or other surface.

UNCERTAIN:
The support relationship cannot be determined.

2. both_lips_grasped

This has a VERY STRICT definition.

YES only if, at TARGET:
- BOTH peelable/free package lips or opening edges are securely acquired,
- the two sides needed for pulling the package apart are simultaneously
  controlled,
- the configuration is ready for a two-sided pull/opening action.

NO if:
- only ONE edge/lip has been touched or grasped,
- one hand holds one edge while the other hand is still searching,
- fingers are approaching the second edge,
- the nurse is repositioning/searching for the free lip,
- the second peelable edge has not yet been securely acquired.

Contact with one side is NOT enough.

Searching for the opposite lip is NOT enough.

YES requires the final two-sided grasp configuration.

3. package_separated

YES only if, at TARGET:
- the package layers have genuinely separated,
- there is a real opening/gap created by pulling the package apart.

NO if:
- the wrapper is still closed,
- both lips are grasped but have not yet separated,
- the hands merely look ready to pull,
- a printed line, transparent seam, material edge, fold, or package border
  only LOOKS like a gap.

The natural boundary between package materials is NOT evidence of opening.

There must be actual visible separation of the package layers.

4. contents_leaving

YES only if, at TARGET:
- the sterile contents themselves are visibly moving out of the package,
  being dropped, sliding out, falling out, or being released from it.

NO if:
- the contents remain inside the wrapper,
- the contents are merely visible through an open wrapper,
- the package is open but contents have not started leaving.

For a syringe package, seeing the syringe inside the opened package does NOT
mean contents_leaving=yes.

Use UNCERTAIN rather than guessing.

IMPORTANT:

- Do not infer a later state from the normal procedural order.
- Do not mark package_separated=yes merely because both hands hold the wrapper.
- Do not mark both_lips_grasped=yes while one hand is still searching for
  the second free lip.
- Judge the TARGET FRAME itself.
- Earlier images are only supporting context.
- There are deliberately no example answers in this prompt.

Also assess direct hand/finger contact with the sterile contents:

contents_touched
contents_untouched
not_applicable

Return ONLY one JSON object.

It must contain exactly:

"states":
    "package_held"
    "both_lips_grasped"
    "package_separated"
    "contents_leaving"

Each state value must be exactly one of:
"yes"
"no"
"uncertain"

It must also contain:
"condition"

Condition must be exactly one of:
"contents_touched"
"contents_untouched"
"not_applicable"

Do not include explanations.
""".strip()


def load_model(load_4bit=False):

    check_cuda()

    print(f"Loading processor: {MODEL_ID}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    processor.tokenizer.padding_side = "left"

    print(f"Loading model: {MODEL_ID}", flush=True)

    if load_4bit:

        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
        )

    else:

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()

    print(f"[diagnostic] Model device: {model.device}", flush=True)
    print(
        f"[diagnostic] Active GPU: {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    print(
        "[diagnostic] GPU memory allocated: "
        f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
        flush=True,
    )

    return model, processor


def normalise_status(value):

    if isinstance(value, bool):
        return "yes" if value else "no"

    value = str(value).strip().lower()

    if value in VALID_STATUS:
        return value

    return "uncertain"


def parse_json_response(text):

    fallback = {
        "states": {
            state: "uncertain"
            for state in STATES
        },
        "condition": "not_applicable",
    }

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        return fallback

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback

    raw_states = parsed.get("states", {})

    states = {
        state: normalise_status(
            raw_states.get(
                state,
                "uncertain",
            )
        )
        for state in STATES
    }

    condition = str(
        parsed.get(
            "condition",
            "not_applicable",
        )
    ).strip().lower()

    if condition not in CONDITIONS:
        condition = "not_applicable"

    return {
        "states": states,
        "condition": condition,
    }


def make_temporal_window(
    manifest,
    target_idx,
    window_size=5,
):
    """
    Past-only context.

    For window_size=5:

        T-4 T-3 T-2 T-1 TARGET

    No future images are shown.
    """

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            "temporal_window must be an odd positive integer"
        )

    n = len(manifest)

    if not 0 <= target_idx < n:
        raise IndexError(target_idx)

    start = max(
        0,
        target_idx - window_size + 1,
    )

    window = manifest[
        start:
        target_idx + 1
    ]

    target_position = len(window) - 1

    return window, target_position


def frame_marker(position, target_position):

    if position == target_position:
        return "[TARGET FRAME]"

    offset = target_position - position

    return (
        f"[PAST CONTEXT: "
        f"{offset} sampled frame(s) before TARGET]"
    )


def label_window_batch(
    model,
    processor,
    windows,
    target_positions,
    prompt_text,
):

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable before inference"
        )

    chats = []

    for window, target_position in zip(
        windows,
        target_positions,
    ):

        content = [
            {
                "type": "text",
                "text": prompt_text,
            }
        ]

        for position, entry in enumerate(window):

            content.append(
                {
                    "type": "text",
                    "text": frame_marker(
                        position,
                        target_position,
                    ),
                }
            )

            content.append(
                {
                    "type": "image",
                    "url": entry["frame_path"],
                }
            )

        chats.append(
            [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        )

    inputs = processor.apply_chat_template(
        chats,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_only = generated[
        :,
        input_length:
    ]

    output_texts = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return [
        parse_json_response(text)
        for text in output_texts
    ]


def label_manifest(
    manifest_path,
    out_path,
    load_4bit=False,
    batch_size=1,
    temporal_window=5,
):

    check_cuda()

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if not manifest:
        raise ValueError(
            f"Empty manifest: {manifest_path}"
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1"
        )

    if temporal_window < 1 or temporal_window % 2 == 0:
        raise ValueError(
            "temporal_window must be odd"
        )

    model, processor = load_model(
        load_4bit
    )

    object_name = manifest[0].get(
        "object_name"
    )

    prompt_text = build_prompt(
        object_name
    )

    total_frames = len(manifest)

    print("=" * 70)
    print("VISUAL-STATE LABELLING")
    print("=" * 70)
    print(f"Frames: {total_frames}")
    print(f"Temporal window: {temporal_window}")
    print("Window mode: PAST-ONLY")
    print(f"GPU batch size: {batch_size}")
    print(f"Object: {object_name}")
    print("=" * 70)

    labelled = []

    for start in range(
        0,
        total_frames,
        batch_size,
    ):

        stop = min(
            start + batch_size,
            total_frames,
        )

        target_indices = list(
            range(start, stop)
        )

        windows = []
        target_positions = []

        for target_idx in target_indices:

            window, target_position = make_temporal_window(
                manifest,
                target_idx,
                temporal_window,
            )

            windows.append(window)
            target_positions.append(
                target_position
            )

        results = label_window_batch(
            model=model,
            processor=processor,
            windows=windows,
            target_positions=target_positions,
            prompt_text=prompt_text,
        )

        for target_idx, result in zip(
            target_indices,
            results,
        ):

            entry = manifest[
                target_idx
            ]

            labelled.append(
                {
                    **entry,
                    **result,
                    "temporal_window": temporal_window,
                    "window_mode": "past_only_visual_state",
                }
            )

            states = result[
                "states"
            ]

            print(
                f"[{len(labelled)}/{total_frames}] "
                f"src={os.path.basename(entry['frame_path'])} | "
                f"held={states['package_held'][:3]} "
                f"both={states['both_lips_grasped'][:3]} "
                f"open={states['package_separated'][:3]} "
                f"leave={states['contents_leaving'][:3]}",
                flush=True,
            )

    print()
    print("=" * 70)
    print("VISUAL-STATE SUMMARY")
    print("=" * 70)

    for state in STATES:

        counts = {
            value: sum(
                x["states"][state] == value
                for x in labelled
            )
            for value in VALID_STATUS
        }

        print(
            f"{state:24s} "
            f"YES={counts['yes']:3d} "
            f"NO={counts['no']:3d} "
            f"UNCERTAIN={counts['uncertain']:3d}"
        )

    print("=" * 70)

    os.makedirs(
        os.path.dirname(out_path) or ".",
        exist_ok=True,
    )

    with open(out_path, "w") as f:
        json.dump(
            labelled,
            f,
            indent=2,
        )

    print(
        f"Wrote {len(labelled)} labelled frames to {out_path}"
    )

    return out_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--load_4bit",
        action="store_true",
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

    args = parser.parse_args()

    label_manifest(
        manifest_path=args.manifest,
        out_path=args.out,
        load_4bit=args.load_4bit,
        batch_size=args.batch_size,
        temporal_window=args.temporal_window,
    )
