# Super Resolution (ailia SDK for Unity)

This folder demonstrates single‑image super‑resolution. The scene is `SuperResolution/SuperResolutionSample.unity`, controlled by `SuperResolution/AiliaSuperResolutionSample.cs`.

## Supported AI Models

- SRResNet
  - Purpose: 4× upscaling with a residual CNN.
  - Source: `./AiliaSuperResolutionSample.cs`
  - Script behavior: Loads `srresnet.opt.onnx`, runs once on a sample image, and displays the upscaled result alongside timing info.

- Real‑ESRGAN
  - Purpose: General photo restoration/upscaling.
  - Source: `./AiliaSuperResolutionSample.cs`
  - Script behavior: Loads `RealESRGAN.opt.onnx` with RGB/CHW input and applies postprocessing to `Color32[]`.

- Real‑ESRGAN (Anime)
  - Purpose: Anime/cartoon image upscaling with an anime‑tuned Real‑ESRGAN.
  - Source: `./AiliaSuperResolutionSample.cs`

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under `srresnet` or `real-esrgan`.

## Reusing in Your Own Code

- Create and open:
  - `var model = new AiliaModel();`
  - Optional: `model.Environment(AILIA_ENVIRONMENT_TYPE_GPU);`
  - `model.OpenFile(pathToPrototxt, pathToOnnx);`
- Prepare input and run:
  - Resize image to model tile size from `GetInputShape()`; pack to RGB/CHW and normalize per model.
  - `model.Predict(output, input);` then convert to `Color32[]` as in `OutputDataProcessingCPU`.

## Source Links

- Sample controller: `./AiliaSuperResolutionSample.cs`

