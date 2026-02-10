# Text Recognition (ailia SDK for Unity)

This folder demonstrates OCR using PaddleOCR models with ailia SDK. The scene is `TextRecognition/TextRecognitionSample.unity`, controlled by `TextRecognition/AiliaTextRecognizersSample.cs`.

## Supported AI Models

- PaddleOCR V1
  - Purpose: End‑to‑end OCR (detection → angle classification → recognition) with language‑specific dictionaries.
  - Sources: `./AiliaTextRecognizersSample.cs`, `./AiliaPaddleOCR.cs`
  - Script behavior: Downloads 3 ONNX models (det/cls/rec) plus a dictionary (`.txt`). Runs detection, crops ROIs, classifies orientation, recognizes text, and overlays either ROIs or strings.

- PaddleOCR V3 (PP‑OCRv5)
  - Purpose: Updated OCR pipeline with improved accuracy; server and mobile variants selectable.
  - Sources: `./AiliaTextRecognizersSample.cs`, `./AiliaPaddleOCR.cs`
  - Script behavior: Uses PP‑OCRv5 det/rec and the same cls model. Chooses server/mobile models based on the `ModelSize` field and downloads `PP-OCRv5_rec.txt` as dictionary.

## Languages and Variants

- Languages: Japanese, English, Chinese, German, French, Korean.
- Sizes: Server (higher accuracy) or Mobile (lighter, faster).

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under `paddle_ocr` or `paddle_ocr_v3`.
- Dictionaries: text files stored alongside models; loaded at runtime.

## Reusing in Your Own Code

- Open models:
  - `det.OpenFile(detProto, detOnnx);`
  - `cls.OpenFile(clsProto, clsOnnx);`
  - `rec.OpenFile(recProto, recOnnx);`
- Pipeline (see `AiliaPaddleOCR`):
  - `Detection(...)` → list of text boxes; `Classification(...)` → angle; `Recognition(...)` → strings.
- Switch language by choosing the recognition model and dictionary file that match.

## Source Links

- Sample controller: `./AiliaTextRecognizersSample.cs`
- Paddle OCR helpers: `./AiliaPaddleOCR.cs`

