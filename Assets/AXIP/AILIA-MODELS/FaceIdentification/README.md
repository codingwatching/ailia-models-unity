# Face Identification (ailia SDK for Unity)

This folder demonstrates face feature extraction and similarity search. The scene is `FaceIdentification/FaceIdentificationSample.unity`, controlled by `FaceIdentification/AiliaFeatureExtractorSample.cs`.

## Supported AI Models

- ArcFace (official)
  - Purpose: Face embedding extraction for verification/identification.
  - Source: `./AiliaFeatureExtractorSample.cs`
  - Script behavior: Detect face (YOLOv3‑face), align/crop, compute ArcFace embedding (including flipped augmentation), and cosine similarity against gallery.

- ArcFace‑M (AX retrained)
  - Purpose: Variant ArcFace with different threshold; often paired with a face‑mask detector for robust detection.
  - Source: `./AiliaFeatureExtractorSample.cs`

- VGGFace2 (ResNet50 scratch)
  - Purpose: Generic face features using Caffe weights.
  - Source: `./AiliaFeatureExtractorSample.cs`

- Person ReID Baseline
  - Purpose: Person re‑identification features (non‑face use‑case).
  - Source: `./AiliaFeatureExtractorSample.cs`
  - Script behavior: Detect with YOLOX‑tiny, crop person ROI, compute feature and compare.

## Model Files and Paths

- Detection: `yolov3-face` or `face-mask-detection` or `yolox` (tiny) depending on mode.
- Features: `arcface`, `arcface_mixed_90_82`, `resnet50_scratch` (Caffe), `ft_ResNet50` (ONNX).
- All download to `Application.temporaryCachePath`.

## Reusing in Your Own Code

- Detection → crop → embedding:
  - Use `AiliaDetectorModel` to get face/person ROI; preprocess to expected input size; call `AiliaModel.Predict` for feature.
  - Normalize/compare embeddings; ArcFace thresholds are in the sample (`threshold_arcface`, `threshold_arcfacem`).

## Source Links

- Controller: `./AiliaFeatureExtractorSample.cs`

