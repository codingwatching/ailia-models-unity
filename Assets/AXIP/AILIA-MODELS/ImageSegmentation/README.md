# Image Segmentation (ailia SDK for Unity)

This folder provides semantic/instance segmentation, including Segment Anything. The scene is `ImageSegmentation/ImageSegmentationSample.unity`, controller `ImageSegmentation/AiliaImageSegmentationSample.cs`.

## Supported AI Models

- HRNetV2 (W18 Small v2 / W18 Small v1 / W48)
  - Purpose: General semantic segmentation.
  - Sources: `./SegmentationModel.cs`, `./AiliaImageSegmentationSample.cs`

- Hair Segmentation (lightweight) / PSPNet Hair Segmentation
  - Purpose: Foreground hair mask; PSPNet variant for improved accuracy.
  - Sources: `./SegmentationModel.cs`, `./AiliaImageSegmentationSample.cs`

- DeepLabV3
  - Purpose: General semantic segmentation.
  - Sources: `./SegmentationModel.cs`, `./AiliaImageSegmentationSample.cs`

- U^2‑Net / MODNet
  - Purpose: Portrait/background matting.
  - Sources: `./SegmentationModel.cs`, `./AiliaImageSegmentationSample.cs`

- Segment Anything (SAM v1, ViT‑B)
  - Purpose: Promptable instance segmentation using point/box prompts (sample uses center prompt).
  - Sources: `./SegmentAnythingModel.cs`, `./AiliaImageSegmentationSample.cs`
  - Script behavior: Loads encoder/decoder pairs (ViT‑B + SAM decoder), prepares prompt embedding, and decodes masks at input resolution.

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under the folders referenced in code: `hrnet`, `hair_segmentation`, `pspnet-hair-segmentation`, `deeplabv3`, `u2net`, `modnet`, `segment-anything`.

## Reusing in Your Own Code

- Use `SegmentationModel.GetModelURLs(...)` and `InitializeModels(...)` to open a model.
- Call `AllocateInputAndOutputTensor(...)`, then per frame: `ProcessFrame(...)` → `PostProcesss(...)` to obtain a colorized mask resized back to the input frame.
- For SAM, use `SegmentAnythingModel.InitializeModels(...)` and its encode/decode flow as in the sample.

## Source Links

- Controller: `./AiliaImageSegmentationSample.cs`
- Generic segmentation: `./SegmentationModel.cs`
- SAM: `./SegmentAnythingModel.cs`

