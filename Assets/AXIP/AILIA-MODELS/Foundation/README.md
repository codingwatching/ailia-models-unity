# Foundation (ailia SDK for Unity)

This folder includes Detic for open‑vocabulary object detection. The scene is `Foundation/FoundationSample.scene.unity`, controller `Foundation/AiliaFoundationSample.cs`.

## Supported AI Models

- Detic (Swin‑B 896, LVIS/COCO)
  - Purpose: Open‑vocabulary detection with large label sets.
  - Source: `./AiliaFoundationSample.cs`
  - Script behavior: Loads ONNX (Detic) and runs detection on resized frames; draws class names from `AiliaFoundationLabel.cs`.

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under `detic`.

## Reusing in Your Own Code

- Open model: `ailia_model.OpenFile(proto, onnx)`; prepare a resized input shape (8‑pixel alignment in sample) and a separate H/W tensor; call `Update()` and fetch outputs.

## Source Links

- Controller: `./AiliaFoundationSample.cs`
- Labels: `./AiliaFoundationLabel.cs`

