# Video Tracking + Counting Pipeline

A computer-vision pipeline that tracks and counts pedestrians in video, with a
**reliability layer** that removes spurious tracks to produce an accurate
head-count.

**Pipeline:** YOLOv8 detector → BoTSORT tracker → line-crossing counting +
unique-people counting → reliability filter.

## Motivation

Most detection projects work on static frames. This one processes **temporal**
data: it has to keep a consistent identity for each person across frames, decide
when someone crosses a line, and count how many distinct people appear.

The hard part is not detection — it is that naive counting overcounts badly.

## The problem: track fragmentation

When a person is occluded (walks behind a pole or another person) for longer
than the tracker's memory buffer, they reappear with a **new track ID**. The
tracker sees two different objects where there is really one person.

Counting every unique track ID therefore inflates the result. On the MOT17-04
sequence the raw tracker produces **164 unique IDs**, while the true head-count
is roughly **80–90 people**.

## The reliability layer

A real person is visible for many frames; ghost tracks from ID switches usually
survive only a handful. So we keep only tracks whose lifetime (measured in
frames) is at least `MIN_TRACK_AGE`.

`MIN_TRACK_AGE = 30` (30 frames = 1 second at 30 FPS) was chosen from the track-age
distribution: it brings the filtered count closest to ground truth without
discarding genuine short appearances.

Because a track's full lifetime is only known after the whole video is seen, the
script uses a **two-pass** design:

1. **Pass 1** — measure the age (frame count) of every track ID.
2. **Pass 2** — re-run tracking and count only IDs whose age ≥ `MIN_TRACK_AGE`.

## Results (MOT17-04)

| Metric | Value |
|---|---|
| Raw unique track IDs | 164 |
| Filtered unique people | 85 |
| Ghost tracks removed | 79 |
| Line crossings | 13 |

The reliability layer removes 79 ghost tracks, bringing the unique-people count
from 164 down to 85 — in line with the ground-truth head-count.

Line crossings are unaffected by the filter, because people who actually cross
the mid-frame line are long-lived tracks that pass the reliability threshold
anyway.

## Usage

```bash
pip install ultralytics opencv-python
python video_tracking_counting.py
```

The script expects `MOT17-04.mp4` in the working directory and writes an
annotated `MOT17-04-reliable.mp4` showing bounding boxes, the counting line, and
both live counters.

## Dataset

MOT17-04-FRCNN sequence from the [MOT Challenge](https://motchallenge.net/)
benchmark: ~1050 frames at 30 FPS, 1920×1080, static camera, moderate pedestrian
density. Frames were assembled into an MP4 with OpenCV before running the
pipeline.

## Tech stack

- **YOLOv8n** (Ultralytics) — pretrained COCO detector, person class only
- **BoTSORT** — appearance-based multi-object tracker
- **OpenCV** — video I/O and annotation
- Google Colab (T4 GPU) for development

## Notes

- `yolov8n.pt` is downloaded automatically by Ultralytics on first run, so it is
  not committed to the repo.
- The output video is excluded from the repo (file-size limits); see the
  screenshots below for sample output.
