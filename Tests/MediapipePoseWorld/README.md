# MediaPipe Pose World Landmarks - Python/Unity Comparison Tests

## Overview

These tests verify that the C# MediaPipe Pose World Landmarks processing logic
produces results consistent with the Python reference implementation.

## Test Structure

| File | Description |
|------|-------------|
| `PythonComparisonTest.cs` | C# NUnit tests comparing against Python ground truth |
| `MediapipePoseWorldProcessor.cs` | Testable pure logic extracted from Unity implementation |
| `generate_mediapipe_reference.py` | Python script to generate reference values |
| `MediapipePoseWorldTests.csproj` | .NET 8.0 test project configuration |

## Processing Steps Validated

1. **Sigmoid activation**: `1 / (1 + exp(-x))`
2. **Pixel normalization**: `value / 255.0`
3. **Box decoding**: Raw SSD outputs + anchors → bounding boxes
4. **Non-maximum suppression (NMS)**: Overlapping box merging
5. **Landmark decoding**: 195-value buffer → 33 landmarks (x, y, z, visibility, presence)
6. **Affine inverse coordinate transform**: ROI space → image space
7. **Derived keypoints**: Shoulder center, body center

## Running Tests

### Prerequisites

- .NET 8.0 SDK
- Python 3.x with NumPy (for regenerating reference values)

### Run C# Tests

```bash
cd Tests/MediapipePoseWorld
dotnet test
```

### Regenerate Python Reference Values

```bash
python generate_mediapipe_reference.py
```

## Tolerance

- C# uses `float32`, Python uses `float64` by default
- Tolerance: `1e-4` for most comparisons (accounts for float32 precision)
- Sigmoid: `1e-6` tolerance (simple function, high precision)
