# Pose Estimation (ailia SDK for Unity)

This folder provides multiple 2D human pose estimation samples. The scene is `PoseEstimation/PoseEstimationSample.unity` with controller `PoseEstimation/AiliaPoseEstimatorsSample.cs`.

## Supported AI Models

- Lightweight Human Pose Estimation
  - Purpose: Real‑time 2D keypoint estimation with a lightweight backbone.
  - Sources: `./AiliaPoseEstimatorsSample.cs`
  - Script behavior: Opens a single ONNX pose model (opt variant), runs inference on camera frames, and draws skeletons.

- BlazePose Full Body
  - Purpose: Google’s two‑stage pose detection and landmark estimation (33 keypoints).
  - Sources: `./AiliaBlazepose.cs`, `./AiliaPoseEstimatorsSample.cs`
  - Script behavior: Loads detection and landmark ONNX models, performs region‑of‑interest cropping between stages, and tracks pose across frames. Includes environment fallback to avoid FP16 issues on some GPUs.

- PoseResNet (+ detector)
  - Purpose: High‑accuracy top‑down pose estimation.
  - Sources: `./AiliaPoseResnet.cs`, `./AiliaPoseEstimatorsSample.cs`
  - Script behavior: Uses a detection model to get person boxes, then runs a PoseResNet on each crop and stitches the results back to the image.

- E2Pose (End‑to‑End)
  - Purpose: One‑shot end‑to‑end pose estimation (no separate detector required).
  - Sources: `./AiliaE2Pose.cs`, `./AiliaPoseEstimatorsSample.cs`
  - Script behavior: Opens a single ONNX, preprocesses input to the required resolution, runs inference, and decodes keypoints.

## Model Files and Paths

- Models are downloaded into `Application.temporaryCachePath` via `AiliaDownload` under the folders referenced in code (e.g., `blazepose_fullbody`, `pose_resnet`, `e2pose`).

## Reusing in Your Own Code

- Choose a pipeline:
  - Top‑down (detector + pose): use `AiliaPoseResnet` pattern.
  - Two‑stage (detection + landmark): use `AiliaBlazepose`.
  - Single‑stage: use `AiliaE2Pose`.
- Steps:
  - Open required model files with `AiliaModel.OpenFile(proto, onnx)` (for multi‑stage, open both models).
  - Prepare input using the provided `TexturePreprocessor` or the inline CPU paths from the sample.
  - Run inference and decode keypoints; draw lines/circles for visualization.

## Source Links

- Sample controller: `./AiliaPoseEstimatorsSample.cs`
- BlazePose: `./AiliaBlazepose.cs`
- PoseResNet: `./AiliaPoseResnet.cs`
- E2Pose: `./AiliaE2Pose.cs`

