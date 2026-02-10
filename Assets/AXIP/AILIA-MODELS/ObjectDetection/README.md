# Object Detection (ailia SDK for Unity)

This folder contains real‑time object detection and instance segmentation samples using the ailia SDK. The scene is `ObjectDetection/ObjectDetectionSample.unity`, driven by `ObjectDetection/AiliaDetectorsSample.cs`.

## Supported AI Models

- YOLOv1 Tiny
  - Purpose: Fast object detector (VOC categories).
  - Source: `./AiliaDetectorsSample.cs`
  - Script behavior: Configures detector to YOLOv1, loads Caffe model (`yolov1-tiny.prototxt`/`.caffemodel`), sets VOC labels and draws boxes.

- YOLOv1 Face
  - Purpose: Face detection using YOLOv1.
  - Source: `./AiliaDetectorsSample.cs`

- YOLOv2 / YOLOv2 Tiny (COCO)
  - Purpose: General object detection (COCO categories).
  - Source: `./AiliaDetectorsSample.cs`
  - Script behavior: Loads ONNX (`yolov2*.onnx`), sets anchors (COCO) for tiny, decodes predictions, and NMS.

- YOLOv3 / YOLOv3 Tiny (COCO)
  - Purpose: General object detection with improved accuracy vs YOLOv2.
  - Source: `./AiliaDetectorsSample.cs`

- YOLOv3 Face / YOLOv3 Hand
  - Purpose: Single‑class detectors (face/hand).
  - Source: `./AiliaDetectorsSample.cs`

- YOLOv4 / YOLOv4 Tiny (COCO)
  - Purpose: General object detection with higher accuracy/speed trade‑off.
  - Source: `./AiliaDetectorsSample.cs`

- YOLOX (nano/tiny/s)
  - Purpose: Anchor‑free real‑time detection.
  - Source: `./AiliaDetectorsSample.cs`
  - Script behavior: Sets image format BGR + INT8 range as required by exported YOLOX ONNX.

- YOLOX (NNAPI: tiny/s) [Android]
  - Purpose: Mobile inference via NNAPI using TFLite models.
  - Sources: `./AiliaDetectorsSample.cs`, `./AiliaTFLiteYoloxSample.cs`
  - Script behavior: Downloads TFLite models and delegates to `AiliaTFLiteYoloxSample` for device execution.

- YOLOv11‑Seg
  - Purpose: Object detection with per‑instance segmentation masks.
  - Sources: `./AiliaDetectorsSample.cs`, `./Yolov11Seg.cs`, `./Yolov11SegMathUtils.cs`, `./Yolov11SegNMSUtils.cs`
  - Script behavior: Postprocesses mask embeddings to produce per‑box binary masks and overlays them on the preview.

## Model Files and Paths

- Models are downloaded to `Application.temporaryCachePath` via `AiliaDownload` using the folder names visible in code (e.g., `yolov3`, `yolov4`, `yolox`). Some legacy models use Caffe (`.caffemodel`).

## Reusing in Your Own Code

- Detector setup (ONNX examples):
  - `var det = new AiliaDetectorModel();`
  - Optional: `det.Environment(AILIA_ENVIRONMENT_TYPE_GPU);`
  - Configure per model: `det.Settings(format, channelOrder, valueRange, algorithm, numClasses, flags);`
  - `det.OpenFile(pathToPrototxt, pathToOnnx);`
  - Optional (YOLOv2 tiny): `det.Anchors(AiliaClassifierLabel.COCO_ANCHORS);`
- Inference from a Unity `Color32[]` frame:
  - `var results = det.ComputeFromImage(cameraPixels, width, height, threshold, iou);`
  - Iterate results to draw boxes/labels; for YOLOv11‑Seg, use the provided helpers to overlay `box.mask`.

## Source Links

- Main controller: `./AiliaDetectorsSample.cs`
- YOLOX (TFLite/NNAPI): `./AiliaTFLiteYoloxSample.cs`
- YOLOv11‑Seg helpers: `./Yolov11Seg.cs`, `./Yolov11SegMathUtils.cs`, `./Yolov11SegNMSUtils.cs`

