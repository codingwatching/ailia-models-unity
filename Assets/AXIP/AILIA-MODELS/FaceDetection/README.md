# Face Detection (ailia SDK for Unity)

This folder provides face detection and mesh estimation samples. The scene is `FaceDetection/FaceDetectionSample.unity`, controlled by `FaceDetection/AiliaFaceDetectorsSample.cs`.

## Supported AI Models

- BlazeFace
  - Purpose: Real‑time face detection.
  - Sources: `./AiliaBlazeface.cs`, `./AiliaBlazefaceAnchors.cs`, `./AiliaFaceDetectorsSample.cs`
  - Variants: `blazeface` (front), `blazeface_back` (rear camera friendly).

- FaceMesh (v1)
  - Purpose: 3D face landmarks after detection.
  - Sources: `./AiliaFaceMesh.cs`, `./AiliaFaceMeshDrawUtils.cs`
  - Script behavior: Detect with BlazeFace, crop ROI, then run Mesh to obtain dense landmarks; draw wireframe.

- FaceMesh v2 (detector + landmarks + blendshapes)
  - Purpose: Robust facial landmarks and blendshape regression.
  - Sources: `./AiliaFaceMeshV2.cs`, `./AiliaFaceMeshDrawUtils.cs`
  - Script behavior: Loads three ONNX (detector, landmarks, blendshapes) and composes results.

- RetinaFace (ResNet50)
  - Purpose: Accurate face detection with keypoints.
  - Sources: `./AiliaRetinaface.cs`
  - Script behavior: Runs RetinaFace model, decodes anchors, outputs boxes and 5‑point landmarks.

## Model Files and Paths

- Models download to `Application.temporaryCachePath` under `blazeface`, `facemesh`, `facemesh_v2`, `retinaface`.

## Reusing in Your Own Code

- BlazeFace only:
  - `blaze_face.Detection(model, cameraPixels, w, h)` → face boxes; draw boxes/landmarks.
- FaceMesh pipeline:
  - Open BlazeFace, then Mesh; crop ROI per BlazeFace and pass to Mesh; render with `AiliaFaceMeshDrawUtils`.
- RetinaFace:
  - `retina_face.Detection(model, cameraPixels, w, h)` → boxes + keypoints.

## Source Links

- Controller: `./AiliaFaceDetectorsSample.cs`
- BlazeFace: `./AiliaBlazeface.cs`, `./AiliaBlazefaceAnchors.cs`
- FaceMesh: `./AiliaFaceMesh.cs`, `./AiliaFaceMeshV2.cs`, `./AiliaFaceMeshDrawUtils.cs`
- RetinaFace: `./AiliaRetinaface.cs`

