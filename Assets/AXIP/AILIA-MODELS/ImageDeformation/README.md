# Image Deformation (ailia SDK for Unity)

This folder includes DewarpNet for document/image dewarping. The scene is `ImageDeformation/DewarpnetSample.unity`, controller `ImageDeformation/AiliaDewarpnetSample.cs`.

## Supported AI Models

- DewarpNet (WC/BM)
  - Purpose: Correct page warping via warping field estimation.
  - Source: `./AiliaDewarpnetSample.cs`
  - Script behavior: Loads two ONNX models (WC: warping cues, BM: backward mapping). Uses a compute shader to convert BM outputs to UV and applies via a custom material for dewarped rendering.

## Model Files and Paths

- `wc_model.onnx`, `bm_model.onnx` (+ `.onnx.prototxt`) under `dewarpnet` folder downloaded to `Application.temporaryCachePath`.

## Reusing in Your Own Code

- Open both models; feed resized input; run WC then BM; use the provided compute shader kernels (`WCtoBM_Resize`, `bmOutputToUVTexture`) to build a UV map and render through a material.

## Source Links

- Controller: `./AiliaDewarpnetSample.cs`
- Shader/compute: `./DewarpnetShader.shader`, `./DewarpCompute.compute`

