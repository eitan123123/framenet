import nltk
nltk.download('framenet_v17')

import csv
import nltk
from nltk.corpus import framenet as fn

#################################
# 1) Define 20 (sentence, verb, object) items
#################################
sentences_data = [
    ("I open the door.", "open", "door"),
    ("I close the door.", "close", "door"),
    ("I knock on the door.", "knock", "door"),
    ("I lock the door.", "lock", "door"),

    ("I open the backpack.", "open", "backpack"),
    ("I close the backpack.", "close", "backpack"),
    ("I rummage through the backpack.", "rummage", "backpack"),
    ("I tear the backpack.", "tear", "backpack"),

    ("I open the laptop.", "open", "laptop"),
    ("I close the laptop.", "close", "laptop"),
    ("I fix the laptop.", "fix", "laptop"),
    ("I break the laptop.", "break", "laptop"),

    ("I write with the pencil.", "write", "pencil"),
    ("I drop the pencil.", "drop", "pencil"),
    ("I sharpen the pencil.", "sharpen", "pencil"),
    ("I snap the pencil.", "snap", "pencil"),

    ("I strum the guitar.", "strum", "guitar"),
    ("I tune the guitar.", "tune", "guitar"),
    ("I break the guitar.", "break", "guitar"),
    ("I smash the guitar.", "smash", "guitar"),
]

###############################################
# 2) Create CSV with frames for each verb
###############################################
out_filename = "my_framenet_output.csv"

with open(out_filename, mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header row
    header = [
        "ID",
        "Frame #",
        "Sentence",
        "Verb",
        "Object",
        "Envoked Frame",
        "",  # empty column
        "Ground Truth (Manually Checking)",
        "Frame Definition",
        "Notes",
        "Did the model Predicte the Frame"
    ]
    csv_writer.writerow(header)

    current_id = 1

    for (sentence_text, verb, obj) in sentences_data:
        # 2A) Find all frames for this specific verb, e.g. "open.v"
        matched_frames = []

        for lu in fn.lus():
            # lu.name is something like "open.v", "close.v", "open.n", etc.
            if lu.name == f"{verb}.v":
                matched_frames.append(lu.frame)

        # 2B) Deduplicate frames by their name (or by .ID)
        unique_frames_map = {frame_obj.name: frame_obj for frame_obj in matched_frames}
        matched_frames = list(unique_frames_map.values())

        # 2C) Limit to 12 frames max
        matched_frames = matched_frames[:12]

        # 2D) If no frames found, optionally write a row or skip
        if not matched_frames:
            # Uncomment below if you want a row indicating no frames found
            # row = [
            #     current_id,
            #     1,  # Frame #
            #     sentence_text,
            #     verb,
            #     obj,
            #     "NO_FRAMES_FOUND",
            #     "",
            #     "no",
            #     f"No frames found for verb: {verb}.v in FrameNet",
            #     "",
            #     "no"
            # ]
            # csv_writer.writerow(row)
            current_id += 1
            continue

        # 2E) Write CSV row for each matched Frame
        frame_number = 1
        for frame_obj in matched_frames:
            frame_name = frame_obj.name
            frame_def = frame_obj.definition or ""

            row = [
                current_id,  # ID
                frame_number,  # Frame # (1..12)
                sentence_text,  # Sentence
                verb,  # Verb
                obj,  # Object
                frame_name,  # Envoked Frame
                "",  # empty column
                "yes",  # Ground Truth (placeholder)
                frame_def,  # Frame Definition
                "",  # Notes (placeholder)
                "yes"  # Did the model Predict the Frame? (placeholder)
            ]
            csv_writer.writerow(row)
            frame_number += 1

        current_id += 1

print(f"CSV file '{out_filename}' created with frames for each verb (up to 12).")