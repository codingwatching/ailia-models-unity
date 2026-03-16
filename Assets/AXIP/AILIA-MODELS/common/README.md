# Common Utilities (ailia SDK for Unity)

Shared helpers used across all samples live here. These utilities handle model download, camera/image input, basic rendering helpers, and image operations so you can reuse the same patterns in your own code.

## What’s Inside

- `Scripts/AiliaDownload.cs`: Model/file downloader with progress UI integration.
- `Scripts/AiliaCamera.cs`: Simple webcam capture and frame access.
- `Scripts/AiliaImageSource.cs`: Image loader/resizer for Texture2D sources.
- `Scripts/AiliaVideoSource.cs`: Video source helper (where used by samples).
- `Scripts/AiliaRenderer.cs`: Base renderer with small drawing helpers (text/rect, line UI hooks).
- `Scripts/AiliaImageUtil.cs`: Crop/resize/palette utilities and safe pixel extraction.
- `ComputeShader`, `Shader`, `UI`: Shared compute kernels, shaders, and UI prefabs used by samples.

## Key Utilities and Usage

- `AiliaDownload`
  - Purpose: Download model assets to `Application.temporaryCachePath` and show progress.
  - UI: Assign `DownloaderProgressPanel` (a GameObject with Image/Text/Button children as expected by the script).
  - Typical flow:
    - Build a `List<ModelDownloadURL>` with `folder_path`, `file_name` (optional `local_name`).
    - Call `DownloadWithProgressFromURL(urlList, () => { /* OpenFile here */ }, base_urlOptional)`.
  - Note: Most samples use the default base URL (`https://storage.googleapis.com/ailia-models/`); some (e.g., TFLite/NNAPI) override with a different base URL.

- `AiliaCamera`
  - Purpose: Access webcam frames.
  - Usage: `CreateCamera(camera_id, crop_squareOptional)`, then each frame `GetWidth() / GetHeight() / GetPixels32()`; check `IsEnable()`; cleanup via `DestroyCamera()`.

- `AiliaImageSource`
  - Purpose: Manage still images as inputs (file/Texture2D).
  - Usage: `CreateSource(uriOrTexture)`, `Resize(width, height)`, `GetPixels32(rect, upsideDownOptional)`, `Width`, `Height`, `IsPrepared`.
  - Notes: Many samples call `Resize()` to match model input tiles, then `GetPixels32(Rect(0,0,w,h), true)` to fetch top-to-bottom pixels.

- `AiliaRenderer`
  - Purpose: Small base class for samples providing `DrawText`, `DrawRect2D`, line/text panel wiring, and `Clear()` for overlay buffers.
  - Tip: If you build your own sample MonoBehaviour, inheriting from `AiliaRenderer` gives you quick text/box drawing on `Color32[]` frames.

- `AiliaImageUtil`
  - Purpose: Image helpers used by many samples.
  - Highlights:
    - `GetCropRect(Texture|W,H, Crop.Center|No)` for square crops.
    - `GetPixels32(texture, rect, upsideDown)` to read pixel blocks safely.
    - `ResizeTexture(texture, w, h)` to scale textures.
    - `CreatePalette(n)` to build color LUTs for segmentation overlays.

## Common Patterns

- GPU backend selection
  - Call `model.Environment(AILIA_ENVIRONMENT_TYPE_GPU)` before `OpenFile(...)` when GPU mode is desired. This applies to `AiliaModel`, and to higher-level wrappers (e.g., detectors/classifiers) that internally wrap an `AiliaModel`.

- Memory optimization
  - Many samples set `AILIA_MEMORY_REDUCE_CONSTANT | AILIA_MEMORY_REDUCE_CONSTANT_WITH_INPUT_INITIALIZER | AILIA_MEMORY_REUSE_INTERSTAGE` prior to `OpenFile` via `SetMemoryMode(...)`.

- File paths
  - Downloads target `Application.temporaryCachePath`; some samples load fixed assets from `Application.streamingAssetsPath` (e.g., `rvc_f0.onnx`).

- Image orientation
  - Unity textures are bottom-to-top; many models expect top-to-bottom. Samples either fetch `GetPixels32(..., true)` or flip before rendering results (`VerticalFlip` in sample scripts).

## Minimal Examples

- Download + open model
  - Assign UI: `ailia_download.DownloaderProgressPanel = UICanvas.transform.Find("DownloaderProgressPanel").gameObject;`
  - Build URLs: `var urls = new List<ModelDownloadURL>{ new(){ folder_path = "resnet50", file_name = "resnet50.opt.onnx.prototxt" }, new(){ folder_path = "resnet50", file_name = "resnet50.opt.onnx" } };`
  - Download+open: `StartCoroutine(ailia_download.DownloadWithProgressFromURL(urls, () => { model.OpenFile(cache+"/resnet50.opt.onnx.prototxt", cache+"/resnet50.opt.onnx"); }));`

- Camera frame fetch
  - `ailia_camera.CreateCamera(camera_id);`
  - Per frame: `if (!ailia_camera.IsEnable()) return; var w=ailia_camera.GetWidth(); var h=ailia_camera.GetHeight(); Color32[] pixels = ailia_camera.GetPixels32();`

- Image source fetch
  - `AiliaImageSource.CreateSource("file://" + pathToPng);`
  - `AiliaImageSource.Resize(inputW, inputH);`
  - `var input = AiliaImageSource.GetPixels32(new Rect(0,0,inputW,inputH), true);`

