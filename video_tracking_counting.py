"""
Video Tracking + Counting Pipeline
==================================

Pipeline: YOLOv8 detector -> BoTSORT tracker -> line-crossing counting + unique
people counting, with a reliability layer that filters out short-lived "ghost"
tracks caused by ID switches after occlusion.

Dataset : MOT17-04-FRCNN sequence (pedestrian-focused, ~1050 frames @ 30 FPS)
Detector: YOLOv8n (pretrained on COCO, class 0 = person)
Tracker : BoTSORT (appearance-based re-identification)

Reliability layer
-----------------
Object tracking suffers from track fragmentation: when a person is occluded for
longer than the tracker's buffer, they reappear with a NEW id. Naively counting
every unique track id therefore inflates the true head-count (164 raw ids vs.
~85 real people in MOT17-04).

The fix is a track-age filter. A real person is visible for many frames; ghost
tracks typically survive only a handful. We keep only tracks that appear for at
least MIN_TRACK_AGE frames (30 frames = 1 second at 30 FPS).

Because a track's full lifetime is only known after the whole video is seen, we
use a two-pass design:
  Pass 1 - measure the age (frame count) of every track id.
  Pass 2 - re-run tracking and count only ids whose age >= MIN_TRACK_AGE.
"""

from collections import Counter

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_PATH = "MOT17-04.mp4"
OUTPUT_PATH = "MOT17-04-reliable.mp4"

MODEL_WEIGHTS = "yolov8n.pt"   # pretrained COCO weights, downloaded automatically
TRACKER = "botsort.yaml"       # built-in BoTSORT config shipped with Ultralytics
PERSON_CLASS = 0               # COCO class index for "person"
CONF_THRESHOLD = 0.25          # detection confidence threshold

# Reliability threshold: a track must survive at least this many frames to count.
# 30 frames = 1 second at 30 FPS.
MIN_TRACK_AGE = 30


def measure_track_ages(video_path):
    """Pass 1: return a Counter mapping each track id to the number of frames
    it appears in (its 'track age')."""
    model = YOLO(MODEL_WEIGHTS)  # fresh model, tracker ids start from 1

    track_age = Counter()
    results = model.track(
        source=video_path,
        tracker=TRACKER,
        classes=[PERSON_CLASS],
        conf=CONF_THRESHOLD,
        persist=True,
        stream=True,
        verbose=False,
    )
    for r in results:
        if r.boxes.id is not None:
            for track_id in r.boxes.id.cpu().numpy().astype(int):
                track_age[track_id] += 1
    return track_age


def count_with_reliability(video_path, output_path, reliable_ids):
    """Pass 2: re-run tracking, draw only reliable tracks, and count both
    line crossings and unique reliable people. Returns (crossing_count,
    unique_count)."""
    # Fresh model so tracker ids reset to 1 and match the ids from Pass 1.
    model = YOLO(MODEL_WEIGHTS)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    line_y = height // 2  # horizontal counting line at the vertical middle

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_y = {}               # track_id -> previous centroid y
    crossed_ids = set()       # reliable ids that already crossed the line
    seen_reliable_ids = set() # reliable ids ever seen (unique count)
    crossing_count = 0

    results = model.track(
        source=video_path,
        tracker=TRACKER,
        classes=[PERSON_CLASS],
        conf=CONF_THRESHOLD,
        persist=True,
        stream=True,
        verbose=False,
    )

    for r in results:
        frame = r.orig_img.copy()  # clean frame; we draw reliable tracks ourselves

        if r.boxes.id is not None:
            boxes = r.boxes.xywh.cpu().numpy()          # center-x, center-y, w, h
            ids = r.boxes.id.cpu().numpy().astype(int)

            for (cx, cy, w, h), track_id in zip(boxes, ids):
                # Reliability filter: skip short-lived ghost tracks entirely.
                if track_id not in reliable_ids:
                    continue

                seen_reliable_ids.add(track_id)

                # Draw bounding box + id for reliable tracks only.
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Line crossing: count each reliable id once.
                if track_id not in crossed_ids and track_id in prev_y:
                    y_before = prev_y[track_id]
                    crossed = (y_before < line_y <= cy) or (y_before > line_y >= cy)
                    if crossed:
                        crossing_count += 1
                        crossed_ids.add(track_id)

                prev_y[track_id] = cy

        # Draw the counting line and both counters.
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Crossings: {crossing_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Unique people: {len(seen_reliable_ids)}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        writer.write(frame)

    writer.release()
    return crossing_count, len(seen_reliable_ids)


def main():
    # Pass 1: measure how long each track lives.
    track_age = measure_track_ages(VIDEO_PATH)
    reliable_ids = {tid for tid, age in track_age.items() if age >= MIN_TRACK_AGE}

    # Pass 2: count using only reliable tracks.
    crossing_count, unique_count = count_with_reliability(
        VIDEO_PATH, OUTPUT_PATH, reliable_ids
    )

    # Report raw vs. filtered results.
    raw_unique = len(track_age)
    print("=== Reliability Layer Results ===")
    print(f"MIN_TRACK_AGE threshold : {MIN_TRACK_AGE} frames")
    print(f"Raw unique IDs          : {raw_unique}")
    print(f"Filtered unique people  : {unique_count}")
    print(f"Ghost tracks removed    : {raw_unique - unique_count}")
    print(f"Filtered crossings      : {crossing_count}")
    print(f"Video saved to          : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
