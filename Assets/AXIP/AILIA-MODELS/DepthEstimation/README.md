# Depth Estimation (ailia SDK for Unity)

This folder provides a monocular depth estimation sample using ailia SDK. The scene is `DepthEstimation/DepthEstimationSample.unity`, driven by `DepthEstimation/AiliaDepthEstimatorsSample.cs`.

## Supported AI Models

- MiDaS
  - Purpose: Predict dense relative depth from a single RGB image.
  - Source: `./AiliaDepthEstimatorsSample.cs`
  - Script behavior: Downloads `midas.onnx` (+ `.onnx.prototxt`), opens it via `AiliaModel`, and feeds a preprocessed camera/frame image. Uses a compute shader (optional) or CPU to prepare ImageNet‑style normalized input. Postprocess scales and normalizes output to a grayscale depth map and renders it to a `RawImage`.

## Model Files and Paths

- `midas.onnx` (+ `midas.onnx.prototxt`) — downloaded to `Application.temporaryCachePath` via `AiliaDownload`.

## Reusing in Your Own Code

- Create and open the model:
  - `var model = new AiliaModel();`
  - `model.OpenFile(pathToPrototxt, pathToOnnx);`
- Input preparation:
  - Resize your image to the model input shape from `model.GetInputShape()`.
  - Normalize as in `InputDataProcessingCPU`/`InputDataProcessingPSP` (ImageNet mean/std, channel‑first when required).
- Run inference and visualize:
  - `model.Predict(output, input);`
  - Map output to 0–255 and draw to a texture (see `LabelPaintMidas`).

## Source Links

- Sample controller: `./AiliaDepthEstimatorsSample.cs`

