/* MediaPipe Pose World Landmarks Shared Inference Engine */
/* Copyright 2025 AXELL CORPORATION */
/*
 * Shared logic for MediaPipe Pose World Landmarks, used by both Unity and standalone tests.
 * Contains:
 *   - IMediapipePoseBackend interface (backend-agnostic model inference)
 *   - Data structs (DetectorOutput, EstimatorOutput)
 *   - MediapipePoseWorldEngine class (all pure computational methods)
 *   - AiliaMediapipePoseBackend class (ailia SDK backend)
 *
 * The logic matches the Python reference implementation:
 *   - ailia-models/pose_estimation_3d/mediapipe_pose_world_landmarks/
 *
 * Unity: included directly via the project
 * Tests: linked via <Compile Include="...MediapipePoseWorldEngine.cs" Link="..." />
 */

using System;
using System.Collections.Generic;
using UnityEngine;

#if !UNITY_2017_1_OR_NEWER
namespace UnityEngine
{
    public class Debug
    {
        public static void Log(string text) { System.Console.WriteLine(text); }
        public static void LogError(string text) { System.Console.WriteLine("[ERROR] " + text); }
        public static void LogWarning(string text) { System.Console.WriteLine("[WARN] " + text); }
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct Color32
    {
        public byte r, g, b, a;
        public Color32(byte r, byte g, byte b, byte a) { this.r = r; this.g = g; this.b = b; this.a = a; }
    }

    public struct Vector2
    {
        public float x, y;
        public Vector2(float x, float y) { this.x = x; this.y = y; }
        public static Vector2 operator *(Vector2 v, float s) { return new Vector2(v.x * s, v.y * s); }
        public static Vector2 operator *(float s, Vector2 v) { return new Vector2(v.x * s, v.y * s); }
        public static Vector2 one => new Vector2(1, 1);
    }

    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static Vector3 zero => new Vector3(0, 0, 0);
        public static Vector3 operator +(Vector3 a, Vector3 b) { return new Vector3(a.x + b.x, a.y + b.y, a.z + b.z); }
        public static Vector3 operator /(Vector3 v, float s) { return new Vector3(v.x / s, v.y / s, v.z / s); }
    }

    public static class Mathf
    {
        public const float PI = 3.14159274f;
        public static float Max(float a, float b) { return Math.Max(a, b); }
        public static float Min(float a, float b) { return Math.Min(a, b); }
        public static float Sqrt(float v) { return (float)Math.Sqrt(v); }
        public static float Pow(float b, float e) { return (float)Math.Pow(b, e); }
        public static float Atan2(float y, float x) { return (float)Math.Atan2(y, x); }
        public static float Cos(float v) { return (float)Math.Cos(v); }
        public static float Sin(float v) { return (float)Math.Sin(v); }
        public static float Exp(float v) { return (float)Math.Exp(v); }
        public static float Clamp(float v, float min, float max) { return Math.Max(min, Math.Min(max, v)); }
        public static float Abs(float v) { return Math.Abs(v); }
        public static float Floor(float v) { return (float)Math.Floor(v); }
    }

    public enum SystemLanguage
    {
        Japanese, Chinese, ChineseSimplified, ChineseTraditional, English
    }

    public static class Application
    {
        public static SystemLanguage systemLanguage => SystemLanguage.English;
    }
}

namespace UnityEngine.UI { }
namespace UnityEngine.Video { }
#endif

// ============================================================
// Data structures
// ============================================================

public struct DetectorOutput
{
    public float[] RawBoxes;   // [tensorCount * 12]
    public float[] RawScores;  // [tensorCount]
}

public struct EstimatorOutput
{
    public float[] Landmarks;  // [195] = 33 * 5 (x, y, z, visibility, presence)
    public float Score;
}

public struct PoseLandmarkResult
{
    public float X;
    public float Y;
    public float Z;
    public float Confidence;
}

// ============================================================
// Backend interface
// ============================================================

public interface IMediapipePoseBackend : IDisposable
{
    void LoadModels(string detectorPath, string estimatorPath);
    DetectorOutput RunDetector(float[] inputData, int width, int height, int channels);
    EstimatorOutput RunEstimator(float[] inputData, int width, int height, int channels);
    string EnvironmentName();
}

// ============================================================
// Engine: pure computational logic (no Unity/model dependencies)
// ============================================================

public class MediapipePoseWorldEngine
{
    public const int DETECTOR_INPUT_RESOLUTION = 224;
    public const int DETECTOR_INPUT_CHANNEL_COUNT = 3;
    public const int DETECTOR_TENSOR_COUNT = 2254;
    public const int DETECTOR_TENSOR_SIZE = 12;
    public const int DETECTOR_KEYPOINT_COUNT = 4;
    public const float DETECTOR_RAW_SCORE_THRESHOLD = 100f;
    public const float DETECTOR_MINIMUM_SCORE_THRESHOLD = 0.5f;
    public const float DETECTOR_MINIMUM_OVERLAP_THRESHOLD = 0.3f;

    public const int ESTIMATOR_INPUT_RESOLUTION = 256;
    public const int ESTIMATOR_LANDMARK_COUNT = 33;
    public const int ESTIMATOR_VALUES_PER_LANDMARK = 5;
    public const int ESTIMATOR_TENSOR_SIZE = 195; // 33 * 5

    // ROI scale factor (matches Python's 1.25)
    public const float ROI_SCALE_FACTOR = 1.25f;

    // BodyPartIndex mapping to output keypoint order (17 body parts)
    public static readonly int[] KEYPOINT_MAPPING = {
        0,  // Nose
        2,  // LeftEye
        5,  // RightEye
        7,  // LeftEar
        8,  // RightEar
        11, // LeftShoulder
        12, // RightShoulder
        13, // LeftElbow
        14, // RightElbow
        15, // LeftWrist
        16, // RightWrist
        23, // LeftHip
        24, // RightHip
        25, // LeftKnee
        26, // RightKnee
        27, // LeftAnkle
        28, // RightAnkle
    };

    // ROI parameters (set after ExtractROI)
    public float RoiCenterX { get; private set; }
    public float RoiCenterY { get; private set; }
    public float RoiBoxSize { get; private set; }
    public float RoiRotation { get; private set; }

    // Decoded landmarks (set after DecodeLandmarks)
    public PoseLandmarkResult[] Landmarks { get; private set; }

    // -------------------------------------------------------
    // Sigmoid
    // -------------------------------------------------------
    public float Sigmoid(float x)
    {
        return 1.0f / (1.0f + Mathf.Exp(-x));
    }

    // -------------------------------------------------------
    // Preprocess image for detector
    // Input: Color32[] in T2B order (top-to-bottom, like OpenCV)
    // Output: float[] HWC normalized to [-1, 1]
    // Also outputs padding amounts for box denormalization
    //
    // Matches Python: warpPerspective + normalize_type='127.5'
    // -------------------------------------------------------
    public float[] PreprocessDetection(Color32[] pixels, int srcWidth, int srcHeight,
        out float padH, out float padW)
    {
        int dstSize = DETECTOR_INPUT_RESOLUTION;
        float[] result = new float[dstSize * dstSize * DETECTOR_INPUT_CHANNEL_COUNT];

        // The Python uses warpPerspective to map a box_size × box_size square
        // centered at image center to dstSize × dstSize. This is equivalent to
        // a simple scale+translate for non-rotated rectangles.
        float boxSize = Math.Max(srcWidth, srcHeight);
        float centerX = srcWidth / 2.0f;
        float centerY = srcHeight / 2.0f;

        // Compute padding amounts (matching Python's formula)
        padH = 0;
        padW = 0;
        float srcAspect = (float)srcHeight / srcWidth;
        if (srcAspect < 1.0f) // landscape: pad_h
            padH = (1.0f - srcAspect) / 2.0f;
        else if (srcAspect > 1.0f) // portrait: pad_w
            padW = (1.0f - 1.0f / srcAspect) / 2.0f;

        // For each destination pixel, find the source pixel
        // Mapping: dst(dx,dy) -> src at (srcX, srcY)
        // srcX = dx * boxSize / dstSize + (centerX - boxSize/2)
        // srcY = dy * boxSize / dstSize + (centerY - boxSize/2)
        float scaleF = boxSize / dstSize;
        float offsetX = centerX - boxSize / 2.0f;
        float offsetY = centerY - boxSize / 2.0f;

        for (int dy = 0; dy < dstSize; dy++)
        {
            for (int dx = 0; dx < dstSize; dx++)
            {
                int idx = (dy * dstSize + dx) * DETECTOR_INPUT_CHANNEL_COUNT;

                float srcXf = dx * scaleF + offsetX;
                float srcYf = dy * scaleF + offsetY;

                if (srcXf < 0 || srcXf >= srcWidth - 1 || srcYf < 0 || srcYf >= srcHeight - 1)
                {
                    // BORDER_CONSTANT (0) - but [-1,-1,-1] after normalization
                    result[idx + 0] = -1.0f;
                    result[idx + 1] = -1.0f;
                    result[idx + 2] = -1.0f;
                    continue;
                }

                int sx0 = (int)srcXf;
                int sy0 = (int)srcYf;
                int sx1 = Math.Min(sx0 + 1, srcWidth - 1);
                int sy1 = Math.Min(sy0 + 1, srcHeight - 1);
                float fx = srcXf - sx0;
                float fy = srcYf - sy0;

                Color32 p00 = pixels[sy0 * srcWidth + sx0];
                Color32 p10 = pixels[sy0 * srcWidth + sx1];
                Color32 p01 = pixels[sy1 * srcWidth + sx0];
                Color32 p11 = pixels[sy1 * srcWidth + sx1];

                // Bilinear interpolation + normalize to [-1, 1]
                float r = p00.r * (1 - fx) * (1 - fy) + p10.r * fx * (1 - fy) +
                          p01.r * (1 - fx) * fy + p11.r * fx * fy;
                float g = p00.g * (1 - fx) * (1 - fy) + p10.g * fx * (1 - fy) +
                          p01.g * (1 - fx) * fy + p11.g * fx * fy;
                float b = p00.b * (1 - fx) * (1 - fy) + p10.b * fx * (1 - fy) +
                          p01.b * (1 - fx) * fy + p11.b * fx * fy;

                result[idx + 0] = r / 127.5f - 1.0f;
                result[idx + 1] = g / 127.5f - 1.0f;
                result[idx + 2] = b / 127.5f - 1.0f;
            }
        }

        return result;
    }

    // Overload without padding output (for Unity compatibility)
    public float[] PreprocessDetection(Color32[] pixels, int srcWidth, int srcHeight)
    {
        return PreprocessDetection(pixels, srcWidth, srcHeight, out _, out _);
    }

    // -------------------------------------------------------
    // Decode detection boxes from raw model output
    // -------------------------------------------------------
    public struct DecodedBox
    {
        public float xMin, yMin, xMax, yMax;
        public float score;
        public float area;
        public float[][] keypoints; // [4][2]
    }

    public List<DecodedBox> DecodeAndProcessBoxes(float[] rawBoxes, float[] rawScores,
        float[,] anchors, float padH = 0, float padW = 0)
    {
        List<DecodedBox> remainingBoxes = new List<DecodedBox>();
        List<DecodedBox> result = new List<DecodedBox>();

        float xScale = DETECTOR_INPUT_RESOLUTION;
        float yScale = DETECTOR_INPUT_RESOLUTION;

        for (int tI = 0; tI < DETECTOR_TENSOR_COUNT; ++tI)
        {
            float score = Sigmoid(Mathf.Clamp(rawScores[tI],
                -DETECTOR_RAW_SCORE_THRESHOLD, DETECTOR_RAW_SCORE_THRESHOLD));

            if (score < DETECTOR_MINIMUM_SCORE_THRESHOLD)
                continue;

            int baseIdx = tI * DETECTOR_TENSOR_SIZE;
            float xCenter = rawBoxes[baseIdx + 0] / xScale * anchors[tI, 2] + anchors[tI, 0];
            float yCenter = rawBoxes[baseIdx + 1] / yScale * anchors[tI, 3] + anchors[tI, 1];
            float width = rawBoxes[baseIdx + 2] / xScale * anchors[tI, 2];
            float height = rawBoxes[baseIdx + 3] / yScale * anchors[tI, 3];

            float[][] keypoints = new float[DETECTOR_KEYPOINT_COUNT][];
            for (int i = 0; i < DETECTOR_KEYPOINT_COUNT; ++i)
            {
                int index = 4 + 2 * i;
                keypoints[i] = new float[] {
                    rawBoxes[baseIdx + index] / xScale * anchors[tI, 2] + anchors[tI, 0],
                    rawBoxes[baseIdx + index + 1] / yScale * anchors[tI, 3] + anchors[tI, 1]
                };
            }

            // Box format: xMin, yMin, xMax, yMax (matching Python convention)
            remainingBoxes.Add(new DecodedBox
            {
                xMin = xCenter - width / 2,
                yMin = yCenter - height / 2,
                xMax = xCenter + width / 2,
                yMax = yCenter + height / 2,
                keypoints = keypoints,
                area = width * height,
                score = score
            });
        }

        // Sort by score descending
        remainingBoxes.Sort((a, b) => Math.Sign(b.score - a.score));

        // Weighted NMS (matches Python's weighted_nms)
        while (remainingBoxes.Count > 0)
        {
            DecodedBox refBox = remainingBoxes[0];
            remainingBoxes.RemoveAt(0);

            float totalScore = refBox.score;
            float sxMin = refBox.xMin * refBox.score;
            float syMin = refBox.yMin * refBox.score;
            float sxMax = refBox.xMax * refBox.score;
            float syMax = refBox.yMax * refBox.score;
            float[][] skp = new float[DETECTOR_KEYPOINT_COUNT][];
            for (int k = 0; k < DETECTOR_KEYPOINT_COUNT; k++)
                skp[k] = new float[] { refBox.keypoints[k][0] * refBox.score, refBox.keypoints[k][1] * refBox.score };

            int mergeCount = 0;
            for (int i = 0; i < remainingBoxes.Count; ++i)
            {
                if (ComputeIoU(refBox, remainingBoxes[i]) > DETECTOR_MINIMUM_OVERLAP_THRESHOLD)
                {
                    var other = remainingBoxes[i];
                    sxMin += other.xMin * other.score;
                    syMin += other.yMin * other.score;
                    sxMax += other.xMax * other.score;
                    syMax += other.yMax * other.score;
                    for (int k = 0; k < DETECTOR_KEYPOINT_COUNT; k++)
                    {
                        skp[k][0] += other.keypoints[k][0] * other.score;
                        skp[k][1] += other.keypoints[k][1] * other.score;
                    }
                    totalScore += other.score;
                    mergeCount++;
                    remainingBoxes.RemoveAt(i);
                    --i;
                }
            }

            if (mergeCount > 0)
            {
                refBox.xMin = sxMin / totalScore;
                refBox.yMin = syMin / totalScore;
                refBox.xMax = sxMax / totalScore;
                refBox.yMax = syMax / totalScore;
                for (int k = 0; k < DETECTOR_KEYPOINT_COUNT; k++)
                {
                    refBox.keypoints[k][0] = skp[k][0] / totalScore;
                    refBox.keypoints[k][1] = skp[k][1] / totalScore;
                }
            }

            // Apply padding correction (denormalize from letterboxed space)
            // Matches Python: (x - pad_w) / (1.0 - pad_w * 2)
            if (padW > 0)
            {
                refBox.xMin = (refBox.xMin - padW) / (1.0f - padW * 2);
                refBox.xMax = (refBox.xMax - padW) / (1.0f - padW * 2);
                for (int k = 0; k < DETECTOR_KEYPOINT_COUNT; k++)
                    refBox.keypoints[k][0] = (refBox.keypoints[k][0] - padW) / (1.0f - padW * 2);
            }
            if (padH > 0)
            {
                refBox.yMin = (refBox.yMin - padH) / (1.0f - padH * 2);
                refBox.yMax = (refBox.yMax - padH) / (1.0f - padH * 2);
                for (int k = 0; k < DETECTOR_KEYPOINT_COUNT; k++)
                    refBox.keypoints[k][1] = (refBox.keypoints[k][1] - padH) / (1.0f - padH * 2);
            }

            result.Add(refBox);
        }

        return result;
    }

    private float ComputeIoU(DecodedBox a, DecodedBox b)
    {
        float widthOverlap = Math.Max(0, Math.Min(a.xMax, b.xMax) - Math.Max(a.xMin, b.xMin));
        float heightOverlap = Math.Max(0, Math.Min(a.yMax, b.yMax) - Math.Max(a.yMin, b.yMin));
        float intersection = widthOverlap * heightOverlap;
        float union = a.area + b.area - intersection;
        if (union <= 0) return 0;
        return intersection / union;
    }

    // -------------------------------------------------------
    // Extract ROI from detected box
    // Matches Python: detection2roi + preprocess_landmark (perspective warp)
    // Input: pixels in T2B order (OpenCV convention)
    // -------------------------------------------------------
    public (float[] roiInput, int roiW, int roiH) ExtractROI(
        Color32[] pixels, int srcWidth, int srcHeight, DecodedBox box)
    {
        // Convert keypoints to pixel coordinates (T2B, no flip)
        // Keypoint 0 = center point, Keypoint 1 = scale point
        float xCenterKp = box.keypoints[0][0];
        float yCenterKp = box.keypoints[0][1];
        float xScaleKp = box.keypoints[1][0];
        float yScaleKp = box.keypoints[1][1];

        float xCenter = xCenterKp * srcWidth;
        float yCenter = yCenterKp * srcHeight;
        float xScale = xScaleKp * srcWidth;
        float yScale = yScaleKp * srcHeight;

        // Compute ROI box size (matches Python)
        float dist = Mathf.Sqrt(Mathf.Pow(xScale - xCenter, 2) + Mathf.Pow(yScale - yCenter, 2));
        float boxSize = dist * 2 * ROI_SCALE_FACTOR;

        // Compute rotation angle (matches Python)
        // Python: angle = pi/2 - atan2(-(y_scale - y_center), x_scale - x_center)
        //       = pi/2 - atan2(y_center - y_scale, x_scale - x_center)
        float rawAngle = (float)(Math.PI / 2) -
            Mathf.Atan2(-(yScale - yCenter), xScale - xCenter);
        // Wrap to [-pi, pi]
        float rotation = rawAngle - 2.0f * Mathf.PI * Mathf.Floor((rawAngle + Mathf.PI) / (2.0f * Mathf.PI));

        // Store ROI parameters for inverse transform
        RoiCenterX = xCenter;
        RoiCenterY = yCenter;
        RoiBoxSize = boxSize;
        RoiRotation = rotation;

        // Compute perspective transform (rotated rect -> 256x256)
        // Using cv2.boxPoints equivalent for rotated rect
        float halfSize = boxSize / 2.0f;
        float rotDeg = rotation * 180.0f / Mathf.PI;
        float cosR = Mathf.Cos(rotation);
        float sinR = Mathf.Sin(rotation);

        // boxPoints of rotated rect centered at (xCenter, yCenter) with size (boxSize, boxSize) and angle rotDeg
        // OpenCV boxPoints returns corners in order: bottom-left, top-left, top-right, bottom-right
        // For rect with angle θ:
        //   corners = center + R(θ) * [±w/2, ±h/2]
        float[][] corners = new float[4][];
        float[] offX = { -halfSize, -halfSize, halfSize, halfSize };
        float[] offY = { halfSize, -halfSize, -halfSize, halfSize };
        for (int i = 0; i < 4; i++)
        {
            corners[i] = new float[] {
                xCenter + offX[i] * cosR - offY[i] * sinR,
                yCenter + offX[i] * sinR + offY[i] * cosR
            };
        }

        // Map from pts1 (corners) to pts2 ([[0,h],[0,0],[w,0],[w,h]])
        int res = ESTIMATOR_INPUT_RESOLUTION;
        float[] dstPtsX = { 0, 0, res, res };
        float[] dstPtsY = { res, 0, 0, res };

        // Compute 3x3 perspective transform matrix from 4 point pairs
        // Using first 3 points for affine (sufficient for rotation+scale+translate)
        // pts1[0] -> (0, res), pts1[1] -> (0, 0), pts1[2] -> (res, 0)
        float[] M = ComputePerspectiveTransform(
            corners[0][0], corners[0][1], corners[1][0], corners[1][1],
            corners[2][0], corners[2][1], corners[3][0], corners[3][1],
            dstPtsX[0], dstPtsY[0], dstPtsX[1], dstPtsY[1],
            dstPtsX[2], dstPtsY[2], dstPtsX[3], dstPtsY[3]);

        // Compute inverse matrix for warpPerspective (dst -> src mapping)
        float[] Minv = InvertMatrix3x3(M);

        // Extract ROI using perspective warp with bilinear interpolation
        float[] roiData = new float[res * res * DETECTOR_INPUT_CHANNEL_COUNT];
        const float factor = 1.0f / 255.0f;

        for (int ry = 0; ry < res; ry++)
        {
            for (int rx = 0; rx < res; rx++)
            {
                // Apply inverse perspective transform: (rx, ry) -> (srcX, srcY)
                float w = Minv[6] * rx + Minv[7] * ry + Minv[8];
                float srcXf = (Minv[0] * rx + Minv[1] * ry + Minv[2]) / w;
                float srcYf = (Minv[3] * rx + Minv[4] * ry + Minv[5]) / w;

                int idx = (ry * res + rx) * DETECTOR_INPUT_CHANNEL_COUNT;

                // BORDER_REPLICATE: clamp to image bounds
                srcXf = Mathf.Clamp(srcXf, 0, srcWidth - 1);
                srcYf = Mathf.Clamp(srcYf, 0, srcHeight - 1);

                int x0 = (int)srcXf;
                int y0 = (int)srcYf;
                int x1i = Math.Min(x0 + 1, srcWidth - 1);
                int y1i = Math.Min(y0 + 1, srcHeight - 1);
                float fx = srcXf - x0;
                float fy = srcYf - y0;

                Color32 p00 = pixels[y0 * srcWidth + x0];
                Color32 p10 = pixels[y0 * srcWidth + x1i];
                Color32 p01 = pixels[y1i * srcWidth + x0];
                Color32 p11 = pixels[y1i * srcWidth + x1i];

                roiData[idx + 0] = (p00.r * (1 - fx) * (1 - fy) + p10.r * fx * (1 - fy) +
                                    p01.r * (1 - fx) * fy + p11.r * fx * fy) * factor;
                roiData[idx + 1] = (p00.g * (1 - fx) * (1 - fy) + p10.g * fx * (1 - fy) +
                                    p01.g * (1 - fx) * fy + p11.g * fx * fy) * factor;
                roiData[idx + 2] = (p00.b * (1 - fx) * (1 - fy) + p10.b * fx * (1 - fy) +
                                    p01.b * (1 - fx) * fy + p11.b * fx * fy) * factor;
            }
        }

        return (roiData, res, res);
    }

    // -------------------------------------------------------
    // Compute 3x3 perspective transform from 4 point pairs
    // Solves M such that M * src = dst (in homogeneous coords)
    // -------------------------------------------------------
    private float[] ComputePerspectiveTransform(
        float sx0, float sy0, float sx1, float sy1, float sx2, float sy2, float sx3, float sy3,
        float dx0, float dy0, float dx1, float dy1, float dx2, float dy2, float dx3, float dy3)
    {
        // Set up 8x8 linear system for perspective transform
        // M = [[a, b, c], [d, e, f], [g, h, 1]]
        // For each point: dx_i = (a*sx_i + b*sy_i + c) / (g*sx_i + h*sy_i + 1)
        //                 dy_i = (d*sx_i + e*sy_i + f) / (g*sx_i + h*sy_i + 1)
        double[,] A = new double[8, 8];
        double[] B = new double[8];

        float[] srcX = { sx0, sx1, sx2, sx3 };
        float[] srcY = { sy0, sy1, sy2, sy3 };
        float[] dstX = { dx0, dx1, dx2, dx3 };
        float[] dstY = { dy0, dy1, dy2, dy3 };

        for (int i = 0; i < 4; i++)
        {
            A[i * 2, 0] = srcX[i];
            A[i * 2, 1] = srcY[i];
            A[i * 2, 2] = 1;
            A[i * 2, 3] = 0;
            A[i * 2, 4] = 0;
            A[i * 2, 5] = 0;
            A[i * 2, 6] = -srcX[i] * dstX[i];
            A[i * 2, 7] = -srcY[i] * dstX[i];
            B[i * 2] = dstX[i];

            A[i * 2 + 1, 0] = 0;
            A[i * 2 + 1, 1] = 0;
            A[i * 2 + 1, 2] = 0;
            A[i * 2 + 1, 3] = srcX[i];
            A[i * 2 + 1, 4] = srcY[i];
            A[i * 2 + 1, 5] = 1;
            A[i * 2 + 1, 6] = -srcX[i] * dstY[i];
            A[i * 2 + 1, 7] = -srcY[i] * dstY[i];
            B[i * 2 + 1] = dstY[i];
        }

        // Solve using Gaussian elimination
        double[] x = SolveLinearSystem(A, B, 8);

        return new float[] {
            (float)x[0], (float)x[1], (float)x[2],
            (float)x[3], (float)x[4], (float)x[5],
            (float)x[6], (float)x[7], 1.0f
        };
    }

    private double[] SolveLinearSystem(double[,] A, double[] b, int n)
    {
        // Gaussian elimination with partial pivoting
        double[,] aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            double maxVal = Math.Abs(aug[col, col]);
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(aug[row, col]) > maxVal)
                {
                    maxVal = Math.Abs(aug[row, col]);
                    maxRow = row;
                }
            }
            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j <= n; j++)
                {
                    double tmp = aug[col, j];
                    aug[col, j] = aug[maxRow, j];
                    aug[maxRow, j] = tmp;
                }
            }
            // Eliminate
            for (int row = col + 1; row < n; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= n; j++)
                    aug[row, j] -= factor * aug[col, j];
            }
        }

        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            x[i] /= aug[i, i];
        }
        return x;
    }

    private float[] InvertMatrix3x3(float[] M)
    {
        double a = M[0], b = M[1], c = M[2];
        double d = M[3], e = M[4], f = M[5];
        double g = M[6], h = M[7], k = M[8];

        double det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
        if (Math.Abs(det) < 1e-10) det = 1e-10;
        double invDet = 1.0 / det;

        return new float[] {
            (float)((e * k - f * h) * invDet), (float)((c * h - b * k) * invDet), (float)((b * f - c * e) * invDet),
            (float)((f * g - d * k) * invDet), (float)((a * k - c * g) * invDet), (float)((c * d - a * f) * invDet),
            (float)((d * h - e * g) * invDet), (float)((b * g - a * h) * invDet), (float)((a * e - b * d) * invDet)
        };
    }

    // -------------------------------------------------------
    // Decode landmarks from estimator output
    // -------------------------------------------------------
    public PoseLandmarkResult[] DecodeLandmarks(float[] rawOutput)
    {
        var landmarks = new PoseLandmarkResult[ESTIMATOR_LANDMARK_COUNT];
        for (int i = 0; i < ESTIMATOR_LANDMARK_COUNT; ++i)
        {
            float x = rawOutput[i * 5] / ESTIMATOR_INPUT_RESOLUTION;
            float y = rawOutput[i * 5 + 1] / ESTIMATOR_INPUT_RESOLUTION;
            float z = rawOutput[i * 5 + 2] / ESTIMATOR_INPUT_RESOLUTION;
            float visibility = rawOutput[i * 5 + 3];
            float presence = rawOutput[i * 5 + 4];

            landmarks[i] = new PoseLandmarkResult
            {
                X = x,
                Y = y,
                Z = z,
                Confidence = Sigmoid(Math.Min(visibility, presence))
            };
        }
        Landmarks = landmarks;
        return landmarks;
    }

    // -------------------------------------------------------
    // Get image coordinate results (with inverse rotation+scale+translate)
    // Matches Python: ((cos*x - sin*y) * box_size + x_center) / im_w
    // -------------------------------------------------------
    public PoseLandmarkResult[] GetImageResult(int imgWidth, int imgHeight)
    {
        if (Landmarks == null || RoiBoxSize == 0) return null;

        float cosa = Mathf.Cos(RoiRotation);
        float sina = Mathf.Sin(RoiRotation);

        var result = new PoseLandmarkResult[ESTIMATOR_LANDMARK_COUNT];
        for (int i = 0; i < ESTIMATOR_LANDMARK_COUNT; i++)
        {
            float x = Landmarks[i].X - 0.5f;
            float y = Landmarks[i].Y - 0.5f;
            result[i] = new PoseLandmarkResult
            {
                X = ((cosa * x - sina * y) * RoiBoxSize + RoiCenterX) / imgWidth,
                Y = ((sina * x + cosa * y) * RoiBoxSize + RoiCenterY) / imgHeight,
                Z = Landmarks[i].Z * RoiBoxSize / imgWidth,
                Confidence = Landmarks[i].Confidence
            };
        }
        return result;
    }

    // -------------------------------------------------------
    // Get world coordinate results (17 body parts + shoulder center + body center)
    // Used by Unity for AiliaPoseEstimator compatibility
    // -------------------------------------------------------
    public PoseLandmarkResult[] GetWorldResult()
    {
        if (Landmarks == null) return null;

        // 17 body keypoints + shoulder center + body center = 19
        var result = new PoseLandmarkResult[19];

        for (int i = 0; i < 17; i++)
        {
            var lm = Landmarks[KEYPOINT_MAPPING[i]];
            result[i] = new PoseLandmarkResult
            {
                X = lm.X, Y = lm.Y, Z = lm.Z, Confidence = lm.Confidence
            };
        }

        // Shoulder center (index 17)
        var ls = Landmarks[11]; // LeftShoulder
        var rs = Landmarks[12]; // RightShoulder
        result[17] = new PoseLandmarkResult
        {
            X = (ls.X + rs.X) / 2, Y = (ls.Y + rs.Y) / 2, Z = (ls.Z + rs.Z) / 2,
            Confidence = Math.Min(ls.Confidence, rs.Confidence)
        };

        // Body center (index 18)
        var lh = Landmarks[23]; // LeftHip
        var rh = Landmarks[24]; // RightHip
        result[18] = new PoseLandmarkResult
        {
            X = (ls.X + rs.X + lh.X + rh.X) / 4,
            Y = (ls.Y + rs.Y + lh.Y + rh.Y) / 4,
            Z = (ls.Z + rs.Z + lh.Z + rh.Z) / 4,
            Confidence = Math.Min(Math.Min(ls.Confidence, rs.Confidence),
                                  Math.Min(lh.Confidence, rh.Confidence))
        };

        return result;
    }

    // Last detected box (for frame-to-frame caching)
    public DecodedBox? LastDetectedBox { get; set; }

    // -------------------------------------------------------
    // Full pipeline: detect + estimate (for E2E testing)
    // -------------------------------------------------------
    public PoseLandmarkResult[] RunFullPipeline(
        IMediapipePoseBackend backend,
        Color32[] pixels, int width, int height,
        float[,] anchors,
        bool worldCoordinate = true,
        DecodedBox? cachedDetection = null)
    {
        DecodedBox detectionBox;

        if (cachedDetection.HasValue)
        {
            detectionBox = cachedDetection.Value;
        }
        else
        {
            // Step 1: Preprocess for detection
            float padH, padW;
            float[] detInput = PreprocessDetection(pixels, width, height, out padH, out padW);

            // Step 2: Run detector
            var detOutput = backend.RunDetector(detInput,
                DETECTOR_INPUT_RESOLUTION, DETECTOR_INPUT_RESOLUTION, DETECTOR_INPUT_CHANNEL_COUNT);

            // Step 3: Decode boxes with padding correction
            var boxes = DecodeAndProcessBoxes(detOutput.RawBoxes, detOutput.RawScores, anchors, padH, padW);
            if (boxes.Count == 0)
            {
                Debug.Log("No pose detected");
                LastDetectedBox = null;
                return null;
            }

            Debug.Log($"Detected {boxes.Count} pose(s), best score={boxes[0].score:F4}");
            detectionBox = boxes[0];
        }

        LastDetectedBox = detectionBox;

        // Step 4: Extract ROI
        var (roiInput, roiW, roiH) = ExtractROI(pixels, width, height, detectionBox);

        // Step 5: Run estimator
        var estOutput = backend.RunEstimator(roiInput, roiW, roiH, DETECTOR_INPUT_CHANNEL_COUNT);
        Debug.Log($"Pose score={estOutput.Score:F4}");

        // Step 6: Decode landmarks
        DecodeLandmarks(estOutput.Landmarks);

        // Step 7: Return results
        if (worldCoordinate)
            return GetWorldResult();
        else
            return GetImageResult(width, height);
    }
}

// ============================================================
// ailia SDK Backend
// ============================================================

public class AiliaMediapipePoseBackend : IMediapipePoseBackend
{
    private ailia.AiliaModel detector;
    private ailia.AiliaModel estimator;
    private bool gpuMode;

    public AiliaMediapipePoseBackend(bool gpuMode = false)
    {
        this.gpuMode = gpuMode;
    }

    public void LoadModels(string detectorPath, string estimatorPath)
    {
        detector = new ailia.AiliaModel();
        estimator = new ailia.AiliaModel();

        if (gpuMode)
        {
            detector.Environment(ailia.Ailia.AILIA_ENVIRONMENT_TYPE_GPU);
            estimator.Environment(ailia.Ailia.AILIA_ENVIRONMENT_TYPE_GPU);
        }

        string detProto = detectorPath + ".prototxt";
        string estProto = estimatorPath + ".prototxt";

        if (!detector.OpenFile(detProto, detectorPath))
            throw new Exception("Failed to open detector: " + detector.GetErrorDetail());
        if (!estimator.OpenFile(estProto, estimatorPath))
            throw new Exception("Failed to open estimator: " + estimator.GetErrorDetail());

        Debug.Log("Models loaded successfully");
    }

    public string EnvironmentName()
    {
        return detector?.EnvironmentName() ?? "";
    }

    /// <summary>Convert HWC float array to CHW layout for ONNX models.</summary>
    private static float[] HwcToChw(float[] hwcData, int width, int height, int channels)
    {
        float[] chw = new float[hwcData.Length];
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    chw[c * height * width + y * width + x] = hwcData[(y * width + x) * channels + c];
        return chw;
    }

    public DetectorOutput RunDetector(float[] inputData, int width, int height, int channels)
    {
        int inputBlobIndex = detector.FindBlobIndexByName("input_1");

        // Convert HWC input to CHW for ONNX model (NCHW format)
        float[] chwData = HwcToChw(inputData, width, height, channels);

        // ailia shape: x=W, y=H, z=C, w=N
        if (!detector.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            x = (uint)width, y = (uint)height, z = (uint)channels, w = 1, dim = 4
        }, inputBlobIndex))
            throw new Exception("SetInputBlobShape failed for detector");

        if (!detector.SetInputBlobData(chwData, inputBlobIndex))
            throw new Exception("SetInputBlobData failed for detector");

        if (!detector.Update())
            throw new Exception("Detector inference failed: " + detector.GetErrorDetail());

        int boxBlobIndex = detector.FindBlobIndexByName("Identity");
        int scoreBlobIndex = detector.FindBlobIndexByName("Identity_1");

        float[] rawBoxes = new float[MediapipePoseWorldEngine.DETECTOR_TENSOR_COUNT *
            MediapipePoseWorldEngine.DETECTOR_TENSOR_SIZE];
        float[] rawScores = new float[MediapipePoseWorldEngine.DETECTOR_TENSOR_COUNT];

        detector.GetBlobData(rawBoxes, boxBlobIndex);
        detector.GetBlobData(rawScores, scoreBlobIndex);

        return new DetectorOutput { RawBoxes = rawBoxes, RawScores = rawScores };
    }

    public EstimatorOutput RunEstimator(float[] inputData, int width, int height, int channels)
    {
        int inputBlobIndex = estimator.FindBlobIndexByName("input_1");

        // Convert HWC input to CHW for ONNX model (NCHW format)
        float[] chwData = HwcToChw(inputData, width, height, channels);

        // ailia shape: x=W, y=H, z=C, w=N
        if (!estimator.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            x = (uint)width, y = (uint)height, z = (uint)channels, w = 1, dim = 4
        }, inputBlobIndex))
            throw new Exception("SetInputBlobShape failed for estimator");

        if (!estimator.SetInputBlobData(chwData, inputBlobIndex))
            throw new Exception("SetInputBlobData failed for estimator");

        if (!estimator.Update())
            throw new Exception("Estimator inference failed: " + estimator.GetErrorDetail());

        int scoreBlobIndex = estimator.FindBlobIndexByName("Identity_1");
        float[] scoreBuffer = new float[1];
        estimator.GetBlobData(scoreBuffer, scoreBlobIndex);

        int landmarkBlobIndex = estimator.FindBlobIndexByName("Identity");
        float[] landmarks = new float[MediapipePoseWorldEngine.ESTIMATOR_TENSOR_SIZE];
        estimator.GetBlobData(landmarks, landmarkBlobIndex);

        return new EstimatorOutput { Landmarks = landmarks, Score = scoreBuffer[0] };
    }

    public void Dispose()
    {
        detector?.Close();
        estimator?.Close();
        detector = null;
        estimator = null;
    }
}
