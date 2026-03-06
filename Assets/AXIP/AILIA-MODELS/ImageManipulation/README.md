# Image Manipulation (ailia SDK for Unity)

This folder collects classic image enhancement models. The scene is `ImageManipulation/ImageManipulationSample.unity`, controller `ImageManipulation/AiliaImageManipulationSample.cs`.

## Supported AI Models

- Noise2Noise (Gaussian)
  - Purpose: Image denoising without clean targets.
  - Source: `./AiliaImageManipulationSample.cs`

- Illumination Correction (IllNet)
  - Purpose: Lighting/illumination normalization.
  - Source: `./AiliaImageManipulationSample.cs`

- Colorization
  - Purpose: Grayscale to color conversion.
  - Sources: `./AiliaImageManipulationSample.cs`, `./AiliaColorConv.cs`
  - Script behavior: Converts LAB/RGB as needed during postprocess.

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under `noise2noise_gaussian`, `illnet`, `colorizer`.

## Reusing in Your Own Code

- Create and open: `new AiliaModel()`, optional GPU environment, then `OpenFile(proto, onnx)`.
- Prepare input via CPU or compute shader paths provided; call `Predict` and postprocess to `Color32[]`.

## Source Links

- Controller: `./AiliaImageManipulationSample.cs`
- Color utilities: `./AiliaColorConv.cs`

