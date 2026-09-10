#!/usr/bin/env python3
"""
RGB-only Pixtral event grounder for the RA-L ablation.

This is intentionally NOT the deprecated framewise visual-state labeller.
It uses the same robot-relevant event semantics as the hybrid method:

    grasp_flaps   = secure two-sided package control established
    peel_apart    = sustained package separation begins
    drop_contents = active opening transitions to release/content delivery

Inputs:
    JSON manifest containing only:
        frame
        frame_path

No hand geometry, optical flow, event candidates, feature tables, or
geometry-derived temporal windows are used.

Method:
    1. RGB-only overlapping coarse temporal scans.
    2. Event-specific RGB-only local refinement.
    3. Deterministic procedural ordering: grasp < peel < release.

The model is loaded through pixtral_labeler.load_model so the model family
and BF16 loading path match the hybrid implementation.
"""

import argparse
import json
import math
import re
from pathlib import Path

import torch

from pixtral_labeler import load_model


EVENTS = ("grasp_flaps", "peel_apart", "drop_contents")

EVENT_DEFINITIONS = {
    "grasp_flaps": (
        "SECURE TWO-SIDED FLAP CONTROL. Select the earliest frame at which "
        "both required package sides/layers are visibly under stable hand "
        "control so peeling can proceed. Do NOT use first touch, exploratory "
        "contact, searching, or a transient pinch."
    ),
    "peel_apart": (
        "SUSTAINED PEEL ONSET. Select the earliest frame at which genuine "
        "relative package separation begins and continues as opening. Do NOT "
        "use exploratory wrapper motion, repositioning, a brief tug, or "
        "movement that does not develop into sustained separation."
    ),
    "drop_contents": (
        "RELEASE ONSET. Select the earliest frame at which manipulation "
        "changes from active opening/peeling to release or content-delivery "
        "behaviour. Do NOT wait for the later instant when the content has "
        "fully fallen out. Do NOT label a temporary pause or regrasp as "
        "release."
    ),
}


def parse_json(text):
    """Extract one JSON object from a deterministic model response."""
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def source_frame(entry):
    if "frame" in entry:
        return int(entry["frame"])

    path = Path(str(entry["frame_path"])).stem
    nums = re.findall(r"\d+", path)
    if not nums:
        raise RuntimeError(f"Cannot recover frame number from {entry}")
    return int(nums[-1])


def choose_indices(start, end, max_images):
    """Uniform chronological image selection, always retaining endpoints."""
    if end < start:
        return []

    count = end - start + 1
    if count <= max_images:
        return list(range(start, end + 1))

    if max_images <= 1:
        return [start]

    vals = []
    for i in range(max_images):
        x = start + (end - start) * i / (max_images - 1)
        vals.append(int(round(x)))

    # stable unique order
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def build_content(prompt, entries, indices):
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "text",
            "text": (
                "Chronological RGB observations follow. "
                "Each image is preceded by its exact source-frame number. "
                "Use only visible RGB evidence."
            ),
        },
    ]

    for idx in indices:
        entry = entries[idx]
        frame = source_frame(entry)
        content.append(
            {"type": "text", "text": f"[SOURCE FRAME {frame}]"}
        )
        content.append(
            {"type": "image", "url": str(entry["frame_path"])}
        )

    return content


def run_pixtral(model, processor, prompt, entries, indices, max_tokens=220):
    chat = [
        {
            "role": "user",
            "content": build_content(prompt, entries, indices),
        }
    ]

    inputs = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]

    raw = processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parsed = parse_json(raw)

    del inputs
    del generated
    torch.cuda.empty_cache()

    return parsed, raw


def coarse_prompt():
    return """You are temporally grounding a human sterile-package-opening demonstration.

Identify robot-relevant manipulation TRANSITIONS from RGB only.

Definitions:
1. grasp_flaps:
   secure two-sided package control is established; not first contact.
2. peel_apart:
   sustained genuine package separation begins; not exploratory movement.
3. drop_contents:
   active opening changes to release/content-delivery behaviour; do not wait
   for the final visible fall of the content.

A window may contain zero, one, or multiple transitions.

Return ONLY valid JSON:
{
  "events": [
    {
      "event": "grasp_flaps|peel_apart|drop_contents",
      "boundary_frame": <integer>,
      "confidence": "high|medium|low"
    }
  ]
}

Rules:
- boundary_frame MUST be one of the displayed SOURCE FRAME numbers.
- Use only visible RGB evidence.
- Do not infer an event merely from hand proximity.
- Do not classify temporary pauses/regrasps as release.
- Return {"events": []} if no event is visually supported.
"""


def refinement_prompt(event):
    definition = EVENT_DEFINITIONS[event]

    return f"""You are refining one temporal boundary in a sterile-package-opening demonstration.

TARGET EVENT:
{event}

DEFINITION:
{definition}

Select the EARLIEST displayed source frame that satisfies the definition.
The transition must be supported by the chronological RGB evidence, not by
hand proximity alone.

Return ONLY valid JSON:
{{
  "event": "{event}",
  "observed": true,
  "boundary_frame": <integer>,
  "confidence": "high|medium|low"
}}

If the event is not visually supported in this window, return:
{{
  "event": "{event}",
  "observed": false,
  "boundary_frame": null,
  "confidence": "low"
}}

boundary_frame MUST be one of the displayed SOURCE FRAME numbers.
"""


def confidence_weight(value):
    value = str(value).strip().lower()
    return {"high": 3.0, "medium": 2.0, "low": 1.0}.get(value, 1.0)


def collect_coarse_candidates(
    model,
    processor,
    entries,
    window_size=28,
    stride=14,
    max_images=12,
):
    n = len(entries)
    candidates = {event: [] for event in EVENTS}
    diagnostics = []

    starts = list(range(0, n, stride))
    if starts and starts[-1] + window_size < n:
        starts.append(max(0, n - window_size))

    seen_windows = set()

    for start in starts:
        end = min(n - 1, start + window_size - 1)
        key = (start, end)
        if key in seen_windows:
            continue
        seen_windows.add(key)

        indices = choose_indices(start, end, max_images)
        parsed, raw = run_pixtral(
            model,
            processor,
            coarse_prompt(),
            entries,
            indices,
            max_tokens=260,
        )

        accepted = []

        if isinstance(parsed, dict):
            events = parsed.get("events", [])
            if isinstance(events, list):
                displayed_frames = {
                    source_frame(entries[i])
                    for i in indices
                }

                for item in events:
                    if not isinstance(item, dict):
                        continue

                    event = str(item.get("event", "")).strip()
                    if event not in EVENTS:
                        continue

                    try:
                        frame = int(item.get("boundary_frame"))
                    except Exception:
                        continue

                    if frame not in displayed_frames:
                        continue

                    conf = str(item.get("confidence", "low")).lower()

                    candidates[event].append(
                        {
                            "frame": frame,
                            "confidence": conf,
                            "window": [
                                source_frame(entries[start]),
                                source_frame(entries[end]),
                            ],
                        }
                    )
                    accepted.append(
                        {
                            "event": event,
                            "frame": frame,
                            "confidence": conf,
                        }
                    )

        diagnostics.append(
            {
                "window": [
                    source_frame(entries[start]),
                    source_frame(entries[end]),
                ],
                "displayed_frames": [
                    source_frame(entries[i])
                    for i in indices
                ],
                "parsed": parsed,
                "accepted": accepted,
                "raw": raw,
            }
        )

    return candidates, diagnostics


def representative_candidate(items):
    """
    Robust RGB-only consensus candidate.

    Use a confidence-weighted median, rather than choosing the earliest raw
    VLM proposal, to reduce single-window hallucinations.
    """
    if not items:
        return None

    expanded = []
    for item in items:
        weight = int(round(confidence_weight(item.get("confidence"))))
        expanded.extend([int(item["frame"])] * max(1, weight))

    expanded.sort()
    return expanded[len(expanded) // 2]


def frame_to_index(entries):
    return {
        source_frame(entry): idx
        for idx, entry in enumerate(entries)
    }


def nearest_index(entries, frame):
    fmap = frame_to_index(entries)
    if frame in fmap:
        return fmap[frame]

    frames = sorted(fmap)
    nearest = min(frames, key=lambda x: abs(x - frame))
    return fmap[nearest]


def refine_event(
    model,
    processor,
    entries,
    event,
    coarse_frame,
    radius=12,
    max_images=16,
):
    if coarse_frame is None:
        return None, {
            "event": event,
            "coarse_frame": None,
            "observed": False,
            "reason": "no_coarse_candidate",
        }

    centre = nearest_index(entries, coarse_frame)
    start = max(0, centre - radius)
    end = min(len(entries) - 1, centre + radius)

    indices = choose_indices(
        start,
        end,
        max_images,
    )

    parsed, raw = run_pixtral(
        model,
        processor,
        refinement_prompt(event),
        entries,
        indices,
        max_tokens=180,
    )

    boundary = None
    observed = False

    displayed_frames = {
        source_frame(entries[i])
        for i in indices
    }

    if isinstance(parsed, dict):
        value = parsed.get("observed", False)
        if isinstance(value, bool):
            observed = value
        else:
            observed = str(value).strip().lower() in {
                "true", "yes", "1"
            }

        try:
            candidate = int(parsed.get("boundary_frame"))
        except Exception:
            candidate = None

        if observed and candidate in displayed_frames:
            boundary = candidate

    return boundary, {
        "event": event,
        "coarse_frame": coarse_frame,
        "window": [
            source_frame(entries[start]),
            source_frame(entries[end]),
        ],
        "displayed_frames": [
            source_frame(entries[i])
            for i in indices
        ],
        "parsed": parsed,
        "raw": raw,
        "observed": bool(boundary is not None),
        "boundary_frame": boundary,
    }


def enforce_order(boundaries):
    """
    Enforce only the task's procedural chronology.

    A later event that violates chronology is marked missing rather than moved
    to an artificial frame. No geometry/motion signal is used.
    """
    grasp = boundaries.get("grasp_flaps")
    peel = boundaries.get("peel_apart")
    drop = boundaries.get("drop_contents")

    if grasp is not None and peel is not None and peel <= grasp:
        peel = None

    if peel is None:
        drop = None
    elif drop is not None and drop <= peel:
        drop = None

    return {
        "grasp_flaps": grasp,
        "peel_apart": peel,
        "drop_contents": drop,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--window_size", type=int, default=28)
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--coarse_images", type=int, default=12)
    parser.add_argument("--refine_radius", type=int, default=12)
    parser.add_argument("--refine_images", type=int, default=16)

    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Optional quantised loading. Do not use for the BF16 paper ablation.",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)

    entries = json.loads(manifest_path.read_text())

    if not isinstance(entries, list) or not entries:
        raise RuntimeError("Manifest must be a non-empty JSON list")

    cleaned = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if "frame_path" not in item:
            continue
        cleaned.append(
            {
                "frame": source_frame(item),
                "frame_path": str(item["frame_path"]),
            }
        )

    cleaned.sort(key=lambda x: int(x["frame"]))

    if not cleaned:
        raise RuntimeError("No valid RGB frames in manifest")

    print("=" * 72)
    print("PIXTRAL-ONLY RGB EVENT GROUNDING")
    print("=" * 72)
    print("Frames:", len(cleaned))
    print("First:", cleaned[0]["frame"])
    print("Last :", cleaned[-1]["frame"])
    print("No geometry / motion / event proposals are used.")
    print("=" * 72)

    model, processor = load_model(
        load_4bit=args.load_4bit
    )

    candidates, coarse_diagnostics = collect_coarse_candidates(
        model,
        processor,
        cleaned,
        window_size=args.window_size,
        stride=args.stride,
        max_images=args.coarse_images,
    )

    coarse_consensus = {
        event: representative_candidate(candidates[event])
        for event in EVENTS
    }

    refined = {}
    refinement_diagnostics = {}

    for event in EVENTS:
        boundary, diagnostic = refine_event(
            model,
            processor,
            cleaned,
            event,
            coarse_consensus[event],
            radius=args.refine_radius,
            max_images=args.refine_images,
        )
        refined[event] = boundary
        refinement_diagnostics[event] = diagnostic

    final = enforce_order(refined)

    output = {
        "method": "pixtral_only_rgb_event_grounding",
        "model": "mistral-community/pixtral-12b",
        "inputs": "rgb_only",
        "uses_hand_geometry": False,
        "uses_optical_flow": False,
        "uses_event_candidates": False,
        "uses_feature_table": False,
        "first_frame": int(cleaned[0]["frame"]),
        "last_frame": int(cleaned[-1]["frame"]),
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
        json.dumps(output, indent=2)
    )

    print()
    print("FINAL RGB-ONLY BOUNDARIES")
    print(json.dumps(final, indent=2))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
