# Generative Adversarial Networks (ailia SDK for Unity)

This folder contains GAN-based face processing samples. The scene is `GenerativeAdversarialNetworks/GenerativeAdversarialNetworksSample.unity`, controlled by `GenerativeAdversarialNetworks/AiliaGenerativeAdversarialNetworksSample.cs`.

Tutorial

- Open the scene above, select the controller object, and choose the model with `AiliaModelType`.
- Provide an input image to `image_lipgan` or `image_gfpgan`. If left null, the camera stream is used.

![gan_settings](../../../../Demo/gan_settings.png)

Supported AI Models

- LipGAN
  - Purpose: Lip synchronization of a face image driven by an audio clip.
  - Sources: `./AiliaLipGan.cs`, `./AiliaGenerativeAdversarialNetworksSample.cs`
  - Script behavior: Detects a face via BlazeFace, crops a 96×96 ROI, computes a mel‑spectrogram from the input audio using ailia.audio (resample to 16 kHz, pre‑emphasis, STFT, mel scaling, log/normalize), and feeds the audio features + ROI to the LipGAN ONNX to rewrite the mouth region frame‑by‑frame. Optionally plays back the audio in sync.

![lipgan](../../../../Demo/lipgan.png)

- GFPGAN
  - Purpose: High‑quality face restoration/enhancement.
  - Sources: `./AiliaGfpGan.cs`, `./AiliaGenerativeAdversarialNetworksSample.cs`
  - Script behavior: Detects a face with BlazeFace, crops the face ROI (restored at 512×512), runs the GFPGAN model, and composites the restored face back into the original frame. Runs one‑shot on stills or the first frame when enabled.

![gfpgan_input](../../../../Demo/gfpgan_input.png)

![gfpgan_output](../../../../Demo/gfpgan_output.png)

Model Files and Paths

- Detection: `blazeface.onnx` (+ `.onnx.prototxt`) under `blazeface`.
- GANs: `lipgan.onnx` (+ `.onnx.prototxt`) under `lipgan`, `GFPGANv1.4.onnx` (+ `.onnx.prototxt`) under `gfpgan`.
- Downloaded via `AiliaDownload` into `Application.temporaryCachePath` as referenced in `AiliaGenerativeAdversarialNetworksSample.cs`.

Reusing in Your Own Code

- Common (BlazeFace detection)
  - Open BlazeFace and call `blaze_face.Detection(model, cameraPixels, width, height)` to get face boxes/landmarks.
  - Compute ROI coordinates and crop for the GAN input.

- LipGAN pipeline
  - Audio features: use ailia.audio to resample to 16 kHz and compute an 80‑bin mel‑spectrogram with STFT hop/window sizes matching the sample (see `AiliaLipGan` for constants and normalization).
  - Prepare a 96×96 face crop, call the LipGAN ONNX via `AiliaModel.Predict`, and blend the generated mouth region back to the frame.

- GFPGAN pipeline
  - Prepare a face crop (512×512), run GFPGAN with `AiliaModel.Predict`, and composite the restored face region back to the original image.

Source Links

- Sample controller: `./AiliaGenerativeAdversarialNetworksSample.cs`
- LipGAN: `./AiliaLipGan.cs`
- GFPGAN: `./AiliaGfpGan.cs`
- Face detection (required): `../FaceDetection/AiliaBlazeface.cs`, `../FaceDetection/AiliaBlazefaceAnchors.cs`
