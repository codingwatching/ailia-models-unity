# Style Transfer (ailia SDK for Unity)

This folder implements AdaIN style transfer (content+style). The scene is `StyleTransfer/StyleTransferSample.unity`, controller `StyleTransfer/AiliaAdainSample.cs`.

## Supported Model

- AdaIN (VGG encoder + decoder)
  - Purpose: Arbitrary style transfer by aligning feature statistics.
  - Sources: `./AiliaAdainSample.cs`, `./AdainCompute.compute`
  - Script behavior: Loads `adain-vgg.onnx` + `adain-decoder.onnx`, extracts content/style features, computes AdaIN via compute shader, and decodes back to RGB. UI toggles between result/style/original.

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under `adain`.

## Reusing in Your Own Code

- Open both models; run feature extraction for content and style; compute AdaIN statistics (mean/std shift) on GPU via the provided compute kernels; run decoder to obtain stylized output.

## Source Links

- Controller: `./AiliaAdainSample.cs`
- Compute: `./AdainCompute.compute`

