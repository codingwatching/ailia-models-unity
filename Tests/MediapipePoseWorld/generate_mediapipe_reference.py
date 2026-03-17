#!/usr/bin/env python3
"""
Generate reference values for MediaPipe Pose World Landmarks C# tests.

This script computes exact numerical values using the same math as the
Python/C# MediaPipe implementation, to be used as ground truth in
PythonComparisonTest.cs.

Processing steps validated:
  1. Sigmoid activation
  2. Pixel normalization (value / 255.0)
  3. Box decoding from raw SSD outputs + anchors
  4. Landmark decoding from raw model output (33 landmarks x 5 values)
  5. Affine inverse coordinate transform
  6. Derived keypoints (shoulder center, body center)

Usage:
  python generate_mediapipe_reference.py
"""

import numpy as np
import json

# =============================================================
# 1. Sigmoid
# =============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

print("=" * 60)
print("1. Sigmoid Reference Values")
print("=" * 60)
test_values = [0.0, 1.0, -1.0, 5.0, -5.0, 100.0, -100.0, 0.5, -0.5, 2.3, -2.3]
for v in test_values:
    result = sigmoid(v)
    print(f"  sigmoid({v:8.1f}) = {result:.10f}")

# =============================================================
# 2. Pixel normalization
# =============================================================
print("\n" + "=" * 60)
print("2. Pixel Normalization (value / 255.0)")
print("=" * 60)
test_pixels = [0, 1, 127, 128, 254, 255]
for p in test_pixels:
    result = p / 255.0
    print(f"  {p:3d} / 255.0 = {result:.10f}")

# =============================================================
# 3. Box decoding with known anchors
# =============================================================
print("\n" + "=" * 60)
print("3. Box Decoding (SSD anchor-based)")
print("=" * 60)

INPUT_RES = 224.0

# Simulated anchors: [xCenter, yCenter, width, height]
# Using 3 anchors for simplicity
anchors = np.array([
    [0.5, 0.5, 1.0, 1.0],   # anchor 0: centered, full size
    [0.25, 0.25, 0.5, 0.5], # anchor 1: top-left quadrant
    [0.75, 0.75, 0.5, 0.5], # anchor 2: bottom-right quadrant
], dtype=np.float64)

# Simulated raw outputs: 3 tensors x 12 values
# [xCenter_offset, yCenter_offset, width, height, kp0_x, kp0_y, kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y]
raw_boxes = np.array([
    [10.0, 20.0, 50.0, 60.0,  15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],  # tensor 0
    [5.0,  10.0, 30.0, 40.0,  8.0,  12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0],   # tensor 1
    [8.0,  15.0, 45.0, 55.0,  12.0, 18.0, 22.0, 28.0, 35.0, 40.0, 42.0, 48.0],   # tensor 2
], dtype=np.float64)

# Raw scores (pre-sigmoid)
raw_scores = np.array([2.0, 3.0, -1.0], dtype=np.float64)

for tI in range(3):
    score = sigmoid(np.clip(raw_scores[tI], -100, 100))
    print(f"\n  Tensor {tI}: raw_score={raw_scores[tI]:.1f}, sigmoid_score={score:.10f}")

    if score < 0.5:
        print(f"    -> FILTERED (score < 0.5)")
        continue

    xCenter = raw_boxes[tI, 0] / INPUT_RES * anchors[tI, 2] + anchors[tI, 0]
    yCenter = raw_boxes[tI, 1] / INPUT_RES * anchors[tI, 3] + anchors[tI, 1]
    width = raw_boxes[tI, 2] / INPUT_RES * anchors[tI, 2]
    height = raw_boxes[tI, 3] / INPUT_RES * anchors[tI, 3]

    print(f"    xCenter = {xCenter:.10f}")
    print(f"    yCenter = {yCenter:.10f}")
    print(f"    width   = {width:.10f}")
    print(f"    height  = {height:.10f}")
    print(f"    xMin    = {xCenter - width / 2:.10f}")
    print(f"    yMin    = {yCenter - height / 2:.10f}")
    print(f"    xMax    = {xCenter + width / 2:.10f}")
    print(f"    yMax    = {yCenter + height / 2:.10f}")

    for kp in range(4):
        idx = 4 + 2 * kp
        kpx = raw_boxes[tI, idx] / INPUT_RES * anchors[tI, 2] + anchors[tI, 0]
        kpy = raw_boxes[tI, idx + 1] / INPUT_RES * anchors[tI, 3] + anchors[tI, 1]
        print(f"    kp{kp} = ({kpx:.10f}, {kpy:.10f})")

# =============================================================
# 4. Landmark decoding
# =============================================================
print("\n" + "=" * 60)
print("4. Landmark Decoding (33 landmarks x 5 values)")
print("=" * 60)

ESTIMATOR_RES = 256.0

# Generate test raw output: 195 values (33 * 5)
np.random.seed(42)
raw_output = np.zeros(195, dtype=np.float64)
# Set some known values for first few landmarks
known_values = [
    # landmark 0 (Nose): x=128, y=51.2, z=-230.4, vis=3.0, pres=2.5
    128.0, 51.2, -230.4, 3.0, 2.5,
    # landmark 1 (LeftInnerEye): x=120, y=45.0, z=-220.0, vis=2.8, pres=2.2
    120.0, 45.0, -220.0, 2.8, 2.2,
    # landmark 2 (LeftEye): x=115, y=44.0, z=-225.0, vis=2.5, pres=2.0
    115.0, 44.0, -225.0, 2.5, 2.0,
    # landmark 11 (LeftShoulder) - fill rest with zeros, set later
]
for i, v in enumerate(known_values):
    raw_output[i] = v

# Set landmarks 11-12 (shoulders) and 23-24 (hips) for derived keypoint test
# LeftShoulder (index 11): x=80, y=120, z=-100, vis=2.0, pres=1.8
raw_output[11 * 5 + 0] = 80.0
raw_output[11 * 5 + 1] = 120.0
raw_output[11 * 5 + 2] = -100.0
raw_output[11 * 5 + 3] = 2.0
raw_output[11 * 5 + 4] = 1.8
# RightShoulder (index 12): x=176, y=118, z=-95, vis=2.2, pres=1.9
raw_output[12 * 5 + 0] = 176.0
raw_output[12 * 5 + 1] = 118.0
raw_output[12 * 5 + 2] = -95.0
raw_output[12 * 5 + 3] = 2.2
raw_output[12 * 5 + 4] = 1.9
# LeftHip (index 23): x=100, y=200, z=-80, vis=1.5, pres=1.3
raw_output[23 * 5 + 0] = 100.0
raw_output[23 * 5 + 1] = 200.0
raw_output[23 * 5 + 2] = -80.0
raw_output[23 * 5 + 3] = 1.5
raw_output[23 * 5 + 4] = 1.3
# RightHip (index 24): x=156, y=198, z=-78, vis=1.6, pres=1.4
raw_output[24 * 5 + 0] = 156.0
raw_output[24 * 5 + 1] = 198.0
raw_output[24 * 5 + 2] = -78.0
raw_output[24 * 5 + 3] = 1.6
raw_output[24 * 5 + 4] = 1.4

print("\n  Raw output values (first 3 landmarks + shoulders + hips):")
for idx in [0, 1, 2, 11, 12, 23, 24]:
    x = raw_output[idx * 5 + 0] / ESTIMATOR_RES
    y = raw_output[idx * 5 + 1] / ESTIMATOR_RES
    z = raw_output[idx * 5 + 2] / ESTIMATOR_RES
    vis = raw_output[idx * 5 + 3]
    pres = raw_output[idx * 5 + 4]
    conf = sigmoid(min(vis, pres))
    print(f"  Landmark {idx:2d}: x={x:.10f}, y={y:.10f}, z={z:.10f}, "
          f"vis={vis:.1f}, pres={pres:.1f}, conf={conf:.10f}")

# =============================================================
# 5. Affine inverse coordinate transform
# =============================================================
print("\n" + "=" * 60)
print("5. Affine Inverse Coordinate Transform")
print("=" * 60)

# Test with known keypoints
kp1_x, kp1_y = 0.5, 0.4  # affine_xc, affine_yc
kp2_x, kp2_y = 0.5, 0.6  # affine_x1, affine_y1
dscale = 1.1

scale = dscale * np.sqrt((kp1_x - kp2_x)**2 + (kp1_y - kp2_y)**2) * 2
angle = np.arctan2(kp1_y - kp2_y, kp1_x - kp2_x) - np.pi / 2

print(f"  kp1 = ({kp1_x}, {kp1_y})")
print(f"  kp2 = ({kp2_x}, {kp2_y})")
print(f"  scale = {scale:.10f}")
print(f"  angle = {angle:.10f}")

# Apply inverse transform to test landmark positions
test_landmarks = [
    (0.5, 0.5),    # center of ROI
    (0.0, 0.0),    # top-left of ROI
    (1.0, 1.0),    # bottom-right of ROI
    (0.25, 0.75),  # arbitrary point
]

cs = np.cos(-angle)
ss = np.sin(-angle)

for lx, ly in test_landmarks:
    x_img = ((lx - 0.5) * cs + (ly - 0.5) * ss) * scale + kp1_x
    y_img = ((lx - 0.5) * -ss + (ly - 0.5) * cs) * scale + kp1_y
    print(f"  landmark({lx:.2f}, {ly:.2f}) -> image({x_img:.10f}, {y_img:.10f})")

# =============================================================
# 6. Derived keypoints
# =============================================================
print("\n" + "=" * 60)
print("6. Derived Keypoints (Shoulder Center, Body Center)")
print("=" * 60)

# Using decoded landmark values from section 4
ls_x = raw_output[11 * 5 + 0] / ESTIMATOR_RES
ls_y = raw_output[11 * 5 + 1] / ESTIMATOR_RES
ls_z = raw_output[11 * 5 + 2] / ESTIMATOR_RES
rs_x = raw_output[12 * 5 + 0] / ESTIMATOR_RES
rs_y = raw_output[12 * 5 + 1] / ESTIMATOR_RES
rs_z = raw_output[12 * 5 + 2] / ESTIMATOR_RES
lh_x = raw_output[23 * 5 + 0] / ESTIMATOR_RES
lh_y = raw_output[23 * 5 + 1] / ESTIMATOR_RES
lh_z = raw_output[23 * 5 + 2] / ESTIMATOR_RES
rh_x = raw_output[24 * 5 + 0] / ESTIMATOR_RES
rh_y = raw_output[24 * 5 + 1] / ESTIMATOR_RES
rh_z = raw_output[24 * 5 + 2] / ESTIMATOR_RES

sc_x = (ls_x + rs_x) / 2
sc_y = (ls_y + rs_y) / 2
sc_z = (ls_z + rs_z) / 2
print(f"  Shoulder Center: x={sc_x:.10f}, y={sc_y:.10f}, z={sc_z:.10f}")

ls_conf = sigmoid(min(raw_output[11*5+3], raw_output[11*5+4]))
rs_conf = sigmoid(min(raw_output[12*5+3], raw_output[12*5+4]))
sc_conf = min(ls_conf, rs_conf)
print(f"  Shoulder Center conf: {sc_conf:.10f}")

bc_x = (ls_x + rs_x + lh_x + rh_x) / 4
bc_y = (ls_y + rs_y + lh_y + rh_y) / 4
bc_z = (ls_z + rs_z + lh_z + rh_z) / 4
print(f"  Body Center: x={bc_x:.10f}, y={bc_y:.10f}, z={bc_z:.10f}")

lh_conf = sigmoid(min(raw_output[23*5+3], raw_output[23*5+4]))
rh_conf = sigmoid(min(raw_output[24*5+3], raw_output[24*5+4]))
bc_conf = min(min(ls_conf, rs_conf), min(lh_conf, rh_conf))
print(f"  Body Center conf: {bc_conf:.10f}")

# =============================================================
# Summary of C# test reference values
# =============================================================
print("\n" + "=" * 60)
print("Reference values for C# tests (copy-paste ready)")
print("=" * 60)

print("\n// Sigmoid values")
for v in test_values:
    print(f"// sigmoid({v}) = {sigmoid(v):.10f}")

print("\n// Landmark 0 (Nose)")
print(f"// x = {raw_output[0]/ESTIMATOR_RES:.10f}")
print(f"// y = {raw_output[1]/ESTIMATOR_RES:.10f}")
print(f"// z = {raw_output[2]/ESTIMATOR_RES:.10f}")
print(f"// conf = {sigmoid(min(raw_output[3], raw_output[4])):.10f}")

print(f"\n// Shoulder Center")
print(f"// x = {sc_x:.10f}")
print(f"// y = {sc_y:.10f}")
print(f"// z = {sc_z:.10f}")
print(f"// conf = {sc_conf:.10f}")

print(f"\n// Body Center")
print(f"// x = {bc_x:.10f}")
print(f"// y = {bc_y:.10f}")
print(f"// z = {bc_z:.10f}")
print(f"// conf = {bc_conf:.10f}")

print(f"\n// Affine params")
print(f"// scale = {scale:.10f}")
print(f"// angle = {angle:.10f}")
for lx, ly in test_landmarks:
    x_img = ((lx - 0.5) * cs + (ly - 0.5) * ss) * scale + kp1_x
    y_img = ((lx - 0.5) * -ss + (ly - 0.5) * cs) * scale + kp1_y
    print(f"// ({lx}, {ly}) -> ({x_img:.10f}, {y_img:.10f})")
