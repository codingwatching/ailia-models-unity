# Hand Detection (ailia SDK for Unity)

This folder provides MediaPipe‑style palm/hand detection and landmarks. The scene is `HandDetection/HandDetectionSample.unity`, controller `HandDetection/AiliaHandDetectorsSample.cs`.

## Supported AI Models

- BlazePalm + BlazeHand
  - Purpose: Two‑stage hand pipeline (palm detector → hand landmarks).
  - Sources: `./AiliaBlazehand.cs`, `./AiliaBlazehandAnchors.cs`, `./AiliaHandDetectorsSample.cs`
  - Script behavior: Runs palm detection (blazepalm), derives ROI, then hand landmarks (blazehand); renders keypoints and connections.

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under `blazepalm` and `blazehand`.

## Reusing in Your Own Code

- Open two models and call `blaze_hand.Main(palmModel, handModel, camera, w, h)` to get landmark results; draw with returned structures.

## Source Links

- Controller: `./AiliaHandDetectorsSample.cs`
- Palm/Hand: `./AiliaBlazehand.cs`, `./AiliaBlazehandAnchors.cs`

