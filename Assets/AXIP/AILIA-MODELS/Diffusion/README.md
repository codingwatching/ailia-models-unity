# Diffusion (ailia SDK for Unity)

This folder provides diffusion-based image generation/manipulation. The scene is `Diffusion/DiffusionSample.unity`, controlled by `Diffusion/AiliaDiffusionSample.cs`.

## Supported AI Models

- Inpainting
  - Purpose: Fill masked regions based on surrounding context.
  - Sources: `./AiliaDiffusionInpainting.cs`, `./AiliaDiffusionDdim.cs`, `./AiliaDiffusionAlphasComprod.cs`
  - Script behavior: Loads diffusion + autoencoder + conditioning models, prepares image and mask (including a low‑res mask branch), then runs a DDIM loop to iteratively denoise and decode the final image.

- Super Resolution
  - Purpose: Single‑image upscaling via diffusion.
  - Sources: `./AiliaDiffusionSuperResolution.cs`, `./AiliaDiffusionDdim.cs`, `./AiliaDiffusionAlphasComprod.cs`
  - Script behavior: Opens a diffusion model and an autoencoder decoder for SR, performs multi‑step DDIM sampling, and decodes to RGB.

- Stable Diffusion (text‑to‑image)
  - Purpose: Generate 512×512 images from prompts.
  - Sources: `./AiliaDiffusionStableDiffusion.cs`, `./AiliaDiffusionDdim.cs`
  - Script behavior: Loads UNet (split as emb/mid/out), autoencoder, and CLIP text encoder. Encodes text, runs DDIM steps on latents, decodes to RGB texture. Supports a “legacy” path and live preview.

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under `diffusion`, `segment-anything`, etc., with filenames referenced directly in code. See `CreateAiliaNet(...)` for exact file names per mode.

## Reusing in Your Own Code

- DDIM loop:
  - Use `AiliaDiffusionDdim.MakeDdimParameters(steps, eta, alphas_cumprod)` and feed model logits per step.
- Inpainting/SR:
  - Open diffusion and autoencoder with `AiliaModel.OpenFile(proto, onnx)`; prepare input tensors (image, mask/cond) as shown; run DDIM and decode.
- Stable Diffusion:
  - Open UNet splits, VAE, and CLIP. Tokenize text, get embeddings, run DDIM over latent, decode with VAE to image.

## Source Links

- Sample controller: `./AiliaDiffusionSample.cs`
- Inpainting: `./AiliaDiffusionInpainting.cs`
- Super‑resolution: `./AiliaDiffusionSuperResolution.cs`
- Stable Diffusion: `./AiliaDiffusionStableDiffusion.cs`
- DDIM/params: `./AiliaDiffusionDdim.cs`, `./AiliaDiffusionAlphasComprod.cs`

