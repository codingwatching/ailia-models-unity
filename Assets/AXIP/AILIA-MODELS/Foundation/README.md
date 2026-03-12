# Foundation (ailia SDK for Unity)

This folder includes Detic for open‑vocabulary detection/segmentation. The scene is `Foundation/FoundationSample.scene.unity`, controller `Foundation/AiliaFoundationSample.cs`.

## Supported AI Models

- Detic (Swin‑B 896, LVIS/COCO)
  - Purpose: Open‑vocabulary detection and segmentation with very large label sets (up to ~21K categories).
  - Source: `./AiliaFoundationSample.cs`
  - Script behavior: Loads ONNX, runs inference on resized frames, and draws boxes/masks with labels from `AiliaFoundationLabel.cs`.

## Model Files and Paths

- Downloads `Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis_op16.onnx` (+ `.prototxt`) to `Application.temporaryCachePath/detic`.

## Reusing in Your Own Code

- Open model and set inputs: image tensor (CHW float32) and `[h,w]` tensor; call `Update()`.
- Read `boxes/scores/classes` and per‑category `mask`; scale boxes and overlay masks; label with `AiliaFoundationLabel`.

## Source Links

- Controller: `./AiliaFoundationSample.cs`
- Labels: `./AiliaFoundationLabel.cs`
