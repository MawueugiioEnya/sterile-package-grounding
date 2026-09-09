#!/usr/bin/env python3

"""
Vocabulary for one complete sterile-package-opening episode.

Instead of assigning one action label to each frame, the VLM predicts
whether key procedural events have happened BY the target frame.
"""

CONDITIONS = [
    "contents_untouched",
    "contents_touched",
    "not_applicable",
]

EVENTS = [
    "package_picked_up",
    "flaps_grasped",
    "peeling_started",
    "contents_release_started",
]

STATUS_VALUES = [
    "yes",
    "no",
    "uncertain",
]

EVENT_DESCRIPTIONS = {

    "package_picked_up": (
        "Has the package been picked up by the nurse by the target frame? "
        "YES means the package has clearly been lifted from its original "
        "supporting surface as part of this episode. Once YES, it remains "
        "YES later in the episode."
    ),

    "flaps_grasped": (
        "Have the two peelable package edges or flaps been securely grasped "
        "by the target frame? YES means the opening edges have been acquired "
        "and secured for opening. Once established, this remains YES even "
        "during later finger adjustments or regrasping."
    ),

    "peeling_started": (
        "Has sustained package peeling or opening begun by the target frame? "
        "YES means separation of the package sides has started. Once peeling "
        "has started, this remains YES for the rest of the episode, including "
        "during pauses, regrasping, holding the package open, or continued "
        "separation."
    ),

    "contents_release_started": (
        "Have the sterile contents begun to leave the package by the target "
        "frame? YES means the contents have visibly begun dropping, sliding, "
        "being tipped, or otherwise leaving the opened package. Once release "
        "starts, this remains YES later in the episode."
    ),
}
