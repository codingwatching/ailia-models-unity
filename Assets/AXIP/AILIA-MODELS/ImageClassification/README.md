# Image Classification (ailia SDK for Unity)

This folder contains real‑time image classification using ailia SDK. The scene is `ImageClassification/ImageClassificationSample.unity`, driven by `ImageClassification/AiliaImageClassificationSample.cs`.

## Supported AI Models

- GoogLeNet
  - Purpose: ImageNet‑class image classification.
  - Source: `./AiliaImageClassificationSample.cs`
  - Script behavior: Loads `googlenet.onnx`, sets RGB/CHW/FP32 input format, and queries top‑k classes per frame.

- ResNet‑50 (INT8 opt variant selectable)
  - Purpose: ImageNet‑class classification with quantized or FP models.
  - Source: `./AiliaImageClassificationSample.cs`
  - Script behavior: Loads `resnet50*.onnx` according to the `resnet50model` field (e.g., `resnet50.opt.onnx`), sets RGB/CHW/INT8 range, and returns ranked logits.

- Inception v3
  - Purpose: ImageNet‑class classification.
  - Source: `./AiliaImageClassificationSample.cs`
  - Script behavior: Loads `inceptionv3.onnx`, sets RGB/CHW/FP32 input, runs per‑frame inference.

## Model Files and Paths

- Models download into `Application.temporaryCachePath` under `googlenet`, `resnet50`, or `inceptionv3` folders.
- Class labels come from `./AiliaClassifierLabel.cs` (English and Japanese names for ImageNet).

## Reusing in Your Own Code

- Create a classifier:
  - `var clf = new AiliaClassifierModel();`
  - Optional: `clf.Environment(AILIA_ENVIRONMENT_TYPE_GPU);`
  - `clf.Settings(format, channelOrder, valueRange);`
  - `clf.OpenFile(pathToPrototxt, pathToOnnx);`
- Run on a `Color32[]` image:
  - `var results = clf.ComputeFromImageB2T(camera, width, height, topK);`
  - Map `category` to labels in `AiliaClassifierLabel`.

## Source Links

- Sample controller: `./AiliaImageClassificationSample.cs`
- Labels: `./AiliaClassifierLabel.cs`

