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

- Segment Anything 2 (SAM2, Hiera‑L)
  - Purpose: Promptable instance segmentation with improved accuracy using point/box prompts. Successor to SAM v1 with enhanced feature extraction via Hiera backbone and FPN.
  - Sources: `./SegmentAnything2Model.cs`, `./AiliaImageSegmentationSample.cs`
  - Script behavior: Loads three component models (image encoder, prompt encoder, mask decoder using Hiera‑L), encodes the image with backbone FPN to produce high-resolution feature maps, prepares prompt embedding from click points or box coordinates, and decodes masks at input resolution (1024×1024).

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under the folders referenced in code: `hrnet`, `hair_segmentation`, `pspnet-hair-segmentation`, `deeplabv3`, `u2net`, `modnet`, `segment-anything`, `segment-anything-2`.

## Reusing in Your Own Code

- Use `SegmentationModel.GetModelURLs(...)` and `InitializeModels(...)` to open a model.
- Call `AllocateInputAndOutputTensor(...)`, then per frame: `ProcessFrame(...)` → `PostProcesss(...)` to obtain a colorized mask resized back to the input frame.
- For SAM v1, use `SegmentAnythingModel.InitializeModels(...)` and its encode/decode flow as in the sample.
- For SAM2, use `SegmentAnything2Model.InitializeModels(...)` with its three-stage pipeline (image encode → prompt encode → mask decode).

## Source Links

- Controller: `./AiliaImageSegmentationSample.cs`
- Generic segmentation: `./SegmentationModel.cs`
- SAM v1: `./SegmentAnythingModel.cs`
- SAM2: `./SegmentAnything2Model.cs`

