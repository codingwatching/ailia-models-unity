#!/usr/bin/env python3
"""
Generate E2E reference values for MediaPipe Pose World Landmarks.

Runs the full pipeline on a test image using ailia SDK and saves
intermediate/final values as .npy files for C# comparison.

Usage:
    python generate_e2e_reference.py [--image PATH] [--model_dir PATH] [--output_dir PATH]

Prerequisites:
    - ailia SDK (Python): pip install ailia
    - ailia license: curl -o ~/.shalo/AILIA.lic https://axip-console.appspot.com/license/download/product/AILIA
    - NumPy, OpenCV
"""

import sys
import os
import math
import argparse
from collections import namedtuple

import cv2
import numpy as np

import ailia

# Add ailia-models util path for helper functions
AILIA_MODELS_DIR = '/tmp/ailia-models'
sys.path.append(os.path.join(AILIA_MODELS_DIR, 'util'))
sys.path.append(os.path.join(AILIA_MODELS_DIR, 'pose_estimation_3d', 'mediapipe_pose_world_landmarks'))

from detection_utils import pose_detection

IMAGE_DET_SIZE = 224
IMAGE_LMK_SIZE = 256


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_detection(img):
    """Preprocess image for pose detector - matches Python reference exactly."""
    im_h, im_w, _ = img.shape

    box_size = max(im_h, im_w)
    rotated_rect = ((im_w // 2, im_h // 2), (box_size, box_size), 0)
    pts1 = cv2.boxPoints(rotated_rect)

    h = w = IMAGE_DET_SIZE
    pts2 = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    pad_h = pad_w = 0
    dst_aspect_ratio = h / w
    src_aspect_ratio = im_h / im_w
    if dst_aspect_ratio > src_aspect_ratio:
        pad_h = (1 - src_aspect_ratio / dst_aspect_ratio) / 2
    else:
        pad_w = (1 - dst_aspect_ratio / src_aspect_ratio) / 2

    # Normalize to [-1, 1]
    img_norm = img_warped.astype(np.float32) / 127.5 - 1.0

    # HWC -> CHW -> NCHW
    img_norm = img_norm.transpose(2, 0, 1)
    img_norm = np.expand_dims(img_norm, axis=0)

    return img_norm, (pad_h, pad_w), img_warped


def preprocess_landmark(img, center, box_size, rotation):
    """Preprocess for landmark estimation - matches Python reference exactly."""
    im_h, im_w, _ = img.shape

    rotated_rect = (center, (box_size, box_size), rotation * 180. / np.pi)
    pts1 = cv2.boxPoints(rotated_rect)

    h = w = IMAGE_LMK_SIZE
    pts2 = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Normalize to [0, 1]
    transformed = transformed.astype(np.float32) / 255.0
    transformed = transformed.transpose(2, 0, 1)
    transformed = np.expand_dims(transformed, axis=0)

    return transformed, M


def to_landmark(landmarks):
    """Decode landmark tensor."""
    num_landmarks = 39
    num = len(landmarks)
    num_dimensions = landmarks.shape[1] // num_landmarks
    output = np.zeros((num, num_landmarks, num_dimensions))
    for i in range(num):
        xx = landmarks[i]
        for j in range(num_landmarks):
            offset = j * num_dimensions
            x = xx[offset]
            y = xx[offset + 1]
            z = xx[offset + 2]
            if 3 < num_dimensions:
                visibility = xx[offset + 3]
                presence = xx[offset + 4]
                output[i, j] = (x, y, z, sigmoid(visibility), sigmoid(presence))
            else:
                output[i, j] = (x, y, z)
    return output


def refine_landmark_from_heatmap(landmarks, heatmap):
    """Refine landmarks using heatmap - matches Python reference."""
    min_confidence_to_refine = 0.5
    kernel_size = 9
    offset = (kernel_size - 1) // 2

    hm_height, hm_width, hm_channels = heatmap.shape

    for i, lm in enumerate(landmarks):
        center_col = int(lm[0] * hm_width)
        center_row = int(lm[1] * hm_height)
        if center_col < 0 or center_col >= hm_width or center_row < 0 or center_col >= hm_height:
            continue

        begin_col = max(0, center_col - offset)
        end_col = min(hm_width, center_col + offset + 1)
        begin_row = max(0, center_row - offset)
        end_row = min(hm_height, center_row + offset + 1)

        val_sum = 0
        weighted_col = 0
        weighted_row = 0
        max_confidence_value = 0
        for row in range(begin_row, end_row):
            for col in range(begin_col, end_col):
                confidence = sigmoid(heatmap[row, col, i])
                val_sum += confidence
                max_confidence_value = max(max_confidence_value, confidence)
                weighted_col += col * confidence
                weighted_row += row * confidence

        if max_confidence_value >= min_confidence_to_refine and val_sum > 0:
            lm[0] = (weighted_col / hm_width / val_sum)
            lm[1] = (weighted_row / hm_height / val_sum)

    return landmarks


def run_pipeline(image_path, model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    im_h, im_w = img_bgr.shape[:2]
    print(f"Image: {image_path} ({im_w}x{im_h})")

    img_rgb = img_bgr[:, :, ::-1]  # BGR -> RGB

    # Save image info
    np.save(os.path.join(output_dir, 'image_shape.npy'), np.array([im_h, im_w]))

    # ========================================
    # Step 1: Load models with ailia SDK
    # ========================================
    det_net = ailia.Net(
        os.path.join(model_dir, 'pose_detection.onnx.prototxt'),
        os.path.join(model_dir, 'pose_detection.onnx'))
    est_net = ailia.Net(
        os.path.join(model_dir, 'pose_landmark_heavy.onnx.prototxt'),
        os.path.join(model_dir, 'pose_landmark_heavy.onnx'))
    print("ailia models loaded")

    # ========================================
    # Step 2: Preprocess for detection
    # ========================================
    det_input, pad, det_img_warped = preprocess_detection(img_rgb)
    np.save(os.path.join(output_dir, 'det_input.npy'), det_input)
    np.save(os.path.join(output_dir, 'det_pad.npy'), np.array(pad))
    cv2.imwrite(os.path.join(output_dir, 'det_preprocessed.png'), det_img_warped)
    print(f"Detection input shape: {det_input.shape}, pad: {pad}")

    # ========================================
    # Step 3: Run detector with ailia
    # ========================================
    det_outputs = det_net.predict([det_input])
    detections, scores = det_outputs
    np.save(os.path.join(output_dir, 'det_raw_boxes.npy'), detections)
    np.save(os.path.join(output_dir, 'det_raw_scores.npy'), scores)
    print(f"Detector outputs: boxes={detections.shape}, scores={scores.shape}")

    # ========================================
    # Step 4: Decode boxes (using reference detection_utils)
    # ========================================
    box, score = pose_detection(detections, scores, pad)
    if len(box) == 0:
        print("No pose detected!")
        np.save(os.path.join(output_dir, 'detected_box.npy'), np.array([]))
        return

    np.save(os.path.join(output_dir, 'detected_box.npy'), box)
    np.save(os.path.join(output_dir, 'detected_score.npy'), np.array([score]))
    print(f"Detection box: {box}")
    print(f"Detection score: {score}")

    # ========================================
    # Step 5: Compute ROI parameters
    # ========================================
    x_center_kp, y_center_kp = box[4:6]
    x_scale_kp, y_scale_kp = box[6:8]
    x_center = x_center_kp * im_w
    y_center = y_center_kp * im_h
    x_scale = x_scale_kp * im_w
    y_scale = y_scale_kp * im_h
    center = (x_center, y_center)

    box_size = (((x_scale - x_center) ** 2 + (y_scale - y_center) ** 2) ** 0.5) * 2
    box_size *= 1.25

    angle = (np.pi * 90 / 180) - math.atan2(-(y_scale - y_center), x_scale - x_center)
    rotation = angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))

    roi_params = np.array([x_center, y_center, box_size, rotation,
                           x_center_kp, y_center_kp, x_scale_kp, y_scale_kp])
    np.save(os.path.join(output_dir, 'roi_params.npy'), roi_params)
    print(f"ROI: center=({x_center:.1f}, {y_center:.1f}), box_size={box_size:.1f}, rotation={rotation:.4f}")

    # ========================================
    # Step 6: Preprocess for landmark estimation
    # ========================================
    lmk_input, M = preprocess_landmark(img_rgb, center, box_size, rotation)
    np.save(os.path.join(output_dir, 'lmk_input.npy'), lmk_input)
    np.save(os.path.join(output_dir, 'perspective_matrix.npy'), M)
    print(f"Landmark input shape: {lmk_input.shape}")

    # ========================================
    # Step 7: Run landmark estimator with ailia
    # ========================================
    est_outputs = est_net.predict([lmk_input])
    landmark_tensor = est_outputs[0]
    pose_flag_tensor = est_outputs[1]
    segmentation_tensor = est_outputs[2]
    heatmap_tensor = est_outputs[3]
    world_landmark_tensor = est_outputs[4]

    np.save(os.path.join(output_dir, 'est_raw_landmarks.npy'), landmark_tensor)
    np.save(os.path.join(output_dir, 'est_pose_score.npy'), pose_flag_tensor)
    np.save(os.path.join(output_dir, 'est_world_landmarks.npy'), world_landmark_tensor)
    np.save(os.path.join(output_dir, 'est_heatmap.npy'), heatmap_tensor)
    print(f"Pose score: {pose_flag_tensor[0, 0]:.4f}")

    # ========================================
    # Step 8: Decode landmarks
    # ========================================
    raw_landmarks = to_landmark(landmark_tensor)
    all_world_landmarks = to_landmark(world_landmark_tensor)

    h = w = IMAGE_LMK_SIZE
    raw_landmarks[:, :, 0] = raw_landmarks[:, :, 0] / w
    raw_landmarks[:, :, 1] = raw_landmarks[:, :, 1] / h
    raw_landmarks[:, :, 2] = raw_landmarks[:, :, 2] / w

    # Heatmap refinement
    refined_landmarks = refine_landmark_from_heatmap(raw_landmarks[0].copy(), heatmap_tensor[0])

    all_landmarks = refined_landmarks[:33, ...]
    all_world_landmarks = all_world_landmarks[0, :33, ...]

    np.save(os.path.join(output_dir, 'landmarks_normalized.npy'), all_landmarks)
    np.save(os.path.join(output_dir, 'world_landmarks_raw.npy'), all_world_landmarks)

    # ========================================
    # Step 9: Transform landmarks to image coordinates
    # ========================================
    cosa = math.cos(rotation)
    sina = math.sin(rotation)
    image_landmarks = all_landmarks.copy()
    for lm in image_landmarks:
        x = lm[0] - 0.5
        y = lm[1] - 0.5
        lm[0] = ((cosa * x - sina * y) * box_size + x_center) / im_w
        lm[1] = ((sina * x + cosa * y) * box_size + y_center) / im_h
        lm[2] = lm[2] * box_size / im_w

    # Transform world landmarks
    world_landmarks_transformed = all_world_landmarks.copy()
    for lm in world_landmarks_transformed:
        x = lm[0]
        y = lm[1]
        lm[0] = cosa * x - sina * y
        lm[1] = sina * x + cosa * y

    np.save(os.path.join(output_dir, 'landmarks_image.npy'), image_landmarks)
    np.save(os.path.join(output_dir, 'world_landmarks_transformed.npy'), world_landmarks_transformed)

    # ========================================
    # Step 10: Print final results for validation
    # ========================================
    print("\n=== Final Landmarks (Image Coordinates, first 10) ===")
    body_parts = ['Nose', 'L_InnerEye', 'L_Eye', 'L_OuterEye', 'R_InnerEye',
                  'R_Eye', 'R_OuterEye', 'L_Ear', 'R_Ear', 'L_Mouth']
    for i, name in enumerate(body_parts):
        lm = image_landmarks[i]
        print(f"  {name}: x={lm[0]:.6f}, y={lm[1]:.6f}, z={lm[2]:.6f}")

    print("\n=== World Landmarks (first 10) ===")
    for i, name in enumerate(body_parts):
        lm = world_landmarks_transformed[i]
        print(f"  {name}: x={lm[0]:.6f}, y={lm[1]:.6f}, z={lm[2]:.6f}")

    # Save the test image as PNG for C# loading
    test_image_png = os.path.join(output_dir, 'test_image.png')
    cv2.imwrite(test_image_png, img_bgr)
    print(f"\nSaved test image: {test_image_png}")
    print(f"All reference data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None, help='Input image path')
    parser.add_argument('--model_dir',
                        default='/tmp/ailia-models/pose_estimation_3d/mediapipe_pose_world_landmarks',
                        help='Model directory')
    parser.add_argument('--output_dir', default='/tmp/mediapipe_pose_test_output',
                        help='Output directory')
    args = parser.parse_args()

    if args.image is None:
        args.image = os.path.join(args.model_dir, 'demo.png')

    run_pipeline(args.image, args.model_dir, args.output_dir)


if __name__ == '__main__':
    main()
