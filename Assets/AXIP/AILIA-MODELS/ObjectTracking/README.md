# Object Tracking (ailia SDK for Unity)

This folder demonstrates multi‑object tracking on top of YOLOX detection. The scene is `ObjectTracking/ObjectTrackingSample.unity`, controller `ObjectTracking/AiliaTrackingSample.cs`.

## Supported Trackers

- ByteTrack
  - Purpose: Robust MOT with simple association.
  - Sources: `./AiliaTrackingSample.cs`
  - Script behavior: Creates `AiliaTrackerModel` with `AILIA_TRACKER_ALGORITHM_BYTE_TRACK`, sets detector to YOLOX (nano/tiny/s), and fuses detections into tracks per frame.

## Model Files and Paths

- YOLOX models download to `Application.temporaryCachePath` under `yolox`.

## Reusing in Your Own Code

- Initialize tracker:
  - `ailia_tracker.Create(AILIA_TRACKER_ALGORITHM_BYTE_TRACK, settings);`
  - Open YOLOX via `AiliaDetectorModel.OpenFile(proto, onnx)` and detect each frame.
  - `var tracks = ailia_tracker.Compute(detections);` then draw track IDs/boxes.

## Source Links

- Controller: `./AiliaTrackingSample.cs`

