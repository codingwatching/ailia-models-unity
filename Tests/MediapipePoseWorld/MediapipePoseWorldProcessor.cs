/* MediaPipe Pose World Landmarks Processor (test-only pure logic) */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * Extracts pure computation logic from AiliaMediapipePoseWorldLandmarks.cs
 * for unit testing without Unity dependencies (no Texture2D, ComputeShader, etc.)
 *
 * This mirrors the processing steps in the Unity implementation:
 *   1. Pixel normalization: value / 255.0 (to [0,1] range)
 *   2. Sigmoid activation: 1 / (1 + exp(-x))
 *   3. Box decoding: raw outputs + anchors -> bounding boxes
 *   4. Non-maximum suppression (NMS)
 *   5. Landmark decoding: raw 195-value output -> 33 landmarks
 *   6. Affine inverse coordinate transform
 *
 * Python reference:
 *   - mediapipe/python/solutions/pose.py
 *   - mediapipe SSD anchor generation
 */

using System;
using System.Collections.Generic;

public struct TestLandmark
{
    public float x;
    public float y;
    public float z;
    public float confidence;
}

public struct TestBox
{
    public float xMin;
    public float xMax;
    public float yMin;
    public float yMax;
    public float area;
    public float score;
    public float[][] keypoints; // [keypointIdx][x,y]

    private int mergedBoxCount;

    public float GetJaccardOverlap(TestBox other)
    {
        float widthOverlap = Math.Max(0, Math.Min(xMax, other.xMax) - Math.Max(xMin, other.xMin));
        float heightOverlap = Math.Max(0, Math.Min(yMax, other.yMax) - Math.Max(yMin, other.yMin));
        float intersection = widthOverlap * heightOverlap;
        float union = area + other.area - intersection;
        return intersection / union;
    }

    public void Merge(TestBox other)
    {
        if (mergedBoxCount == 0)
        {
            xMin *= score;
            yMin *= score;
            xMax *= score;
            yMax *= score;
            for (int i = 0; i < keypoints.Length; i++)
            {
                keypoints[i][0] *= score;
                keypoints[i][1] *= score;
            }
        }
        xMin += other.xMin * other.score;
        yMin += other.yMin * other.score;
        xMax += other.xMax * other.score;
        yMax += other.yMax * other.score;
        for (int i = 0; i < keypoints.Length; i++)
        {
            keypoints[i][0] += other.keypoints[i][0] * other.score;
            keypoints[i][1] += other.keypoints[i][1] * other.score;
        }
        score += other.score;
        mergedBoxCount++;
    }

    public void FinalizeMerge()
    {
        if (mergedBoxCount == 0) return;
        xMin /= score;
        yMin /= score;
        xMax /= score;
        yMax /= score;
        for (int i = 0; i < keypoints.Length; i++)
        {
            keypoints[i][0] /= score;
            keypoints[i][1] /= score;
        }
    }
}

public class MediapipePoseWorldProcessor
{
    // Constants matching AiliaMediapipePoseWorldLandmarks
    public const int DETECTOR_INPUT_RESOLUTION = 224;
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

    /// <summary>
    /// Sigmoid activation function.
    /// Python: 1.0 / (1.0 + np.exp(-x))
    /// </summary>
    public float Sigmoid(float x)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-x));
    }

    /// <summary>
    /// Normalize pixel values to [0, 1] range.
    /// Python: pixel_value / 255.0
    /// </summary>
    public float NormalizePixel(byte value)
    {
        return value / 255f;
    }

    /// <summary>
    /// Preprocess an image (Color32 array) to float array in HWC format, normalized to [0,1].
    /// This matches PreprocessTexture/PreprocessTextureEstimation in the Unity implementation.
    /// Python: img.astype(np.float32) / 255.0
    /// </summary>
    public float[] PreprocessToFloat(byte[] rgbPixels, int width, int height)
    {
        float[] result = new float[width * height * 3];
        const float factor = 1.0f / 255.0f;

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                int srcIdx = (h * width + w) * 3;
                int dstIdx = (h * width + w) * 3;
                result[dstIdx + 0] = rgbPixels[srcIdx + 0] * factor;
                result[dstIdx + 1] = rgbPixels[srcIdx + 1] * factor;
                result[dstIdx + 2] = rgbPixels[srcIdx + 2] * factor;
            }
        }
        return result;
    }

    /// <summary>
    /// Decode detection boxes from raw model outputs and anchors.
    /// Matches DecodeAndProcessBoxes in AiliaMediapipePoseWorldLandmarks.
    ///
    /// Python reference (mediapipe SSD decoding):
    ///   xCenter = raw[tI, 0] / xScale * anchors[tI, 2] + anchors[tI, 0]
    ///   yCenter = raw[tI, 1] / yScale * anchors[tI, 3] + anchors[tI, 1]
    ///   width   = raw[tI, 2] / wScale * anchors[tI, 2]
    ///   height  = raw[tI, 3] / hScale * anchors[tI, 3]
    /// </summary>
    public List<TestBox> DecodeBoxes(float[] rawBoxes, float[] rawScores,
        float[,] anchors, int tensorCount)
    {
        List<TestBox> remainingBoxes = new List<TestBox>();
        List<TestBox> boxes = new List<TestBox>();

        float xScale = DETECTOR_INPUT_RESOLUTION;
        float yScale = DETECTOR_INPUT_RESOLUTION;
        float wScale = DETECTOR_INPUT_RESOLUTION;
        float hScale = DETECTOR_INPUT_RESOLUTION;

        for (int tI = 0; tI < tensorCount; ++tI)
        {
            float score = Sigmoid(Math.Max(-DETECTOR_RAW_SCORE_THRESHOLD,
                Math.Min(DETECTOR_RAW_SCORE_THRESHOLD, rawScores[tI])));

            if (score < DETECTOR_MINIMUM_SCORE_THRESHOLD)
                continue;

            float xCenter = rawBoxes[tI * DETECTOR_TENSOR_SIZE + 0] / xScale * anchors[tI, 2] + anchors[tI, 0];
            float yCenter = rawBoxes[tI * DETECTOR_TENSOR_SIZE + 1] / yScale * anchors[tI, 3] + anchors[tI, 1];
            float width = rawBoxes[tI * DETECTOR_TENSOR_SIZE + 2] / wScale * anchors[tI, 2];
            float height = rawBoxes[tI * DETECTOR_TENSOR_SIZE + 3] / hScale * anchors[tI, 3];

            float[][] keypoints = new float[DETECTOR_KEYPOINT_COUNT][];
            for (int i = 0; i < DETECTOR_KEYPOINT_COUNT; ++i)
            {
                int index = 4 + 2 * i;
                keypoints[i] = new float[] {
                    rawBoxes[tI * DETECTOR_TENSOR_SIZE + index] / xScale * anchors[tI, 2] + anchors[tI, 0],
                    rawBoxes[tI * DETECTOR_TENSOR_SIZE + index + 1] / yScale * anchors[tI, 3] + anchors[tI, 1]
                };
            }

            remainingBoxes.Add(new TestBox
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

        // NMS
        while (remainingBoxes.Count > 0)
        {
            TestBox referenceBox = remainingBoxes[0];
            TestBox mergedBox = referenceBox;
            remainingBoxes.RemoveAt(0);

            for (int i = 0; i < remainingBoxes.Count; ++i)
            {
                if (referenceBox.GetJaccardOverlap(remainingBoxes[i]) > DETECTOR_MINIMUM_OVERLAP_THRESHOLD)
                {
                    mergedBox.Merge(remainingBoxes[i]);
                    remainingBoxes.RemoveAt(i);
                    --i;
                }
            }

            mergedBox.FinalizeMerge();
            boxes.Add(mergedBox);
        }

        return boxes;
    }

    /// <summary>
    /// Decode landmarks from raw model output buffer.
    /// Matches DecodeAndProcessLandmarks in AiliaMediapipePoseWorldLandmarks.
    ///
    /// Python reference:
    ///   For each of 33 landmarks:
    ///     x = output[i*5 + 0] / 256.0
    ///     y = output[i*5 + 1] / 256.0
    ///     z = output[i*5 + 2] / 256.0
    ///     visibility = output[i*5 + 3]
    ///     presence   = output[i*5 + 4]
    ///     confidence = sigmoid(min(visibility, presence))
    /// </summary>
    public TestLandmark[] DecodeLandmarks(float[] rawOutput)
    {
        if (rawOutput.Length < ESTIMATOR_TENSOR_SIZE)
            throw new ArgumentException($"Expected at least {ESTIMATOR_TENSOR_SIZE} values, got {rawOutput.Length}");

        TestLandmark[] landmarks = new TestLandmark[ESTIMATOR_LANDMARK_COUNT];

        for (int i = 0; i < ESTIMATOR_LANDMARK_COUNT; ++i)
        {
            float x = rawOutput[i * 5] / ESTIMATOR_INPUT_RESOLUTION;
            float y = rawOutput[i * 5 + 1] / ESTIMATOR_INPUT_RESOLUTION;
            float z = rawOutput[i * 5 + 2] / ESTIMATOR_INPUT_RESOLUTION;
            float visibility = rawOutput[i * 5 + 3];
            float presence = rawOutput[i * 5 + 4];

            landmarks[i] = new TestLandmark
            {
                x = x,
                y = y,
                z = z,
                confidence = Sigmoid(Math.Min(visibility, presence))
            };
        }

        return landmarks;
    }

    /// <summary>
    /// Apply affine inverse transform to convert landmark coordinates
    /// from normalized ROI space back to original image space.
    /// Matches the non-world-coordinate path in GetResult.
    ///
    /// Python reference:
    ///   x_img = (x - 0.5) * cos(-angle) + (y - 0.5) * sin(-angle)) * scale + xc
    ///   y_img = (x - 0.5) * -sin(-angle) + (y - 0.5) * cos(-angle)) * scale + yc
    /// </summary>
    public (float x, float y) ApplyAffineInverse(float landmarkX, float landmarkY,
        float affineXc, float affineYc, float affineScale, float affineAngle)
    {
        float cs = (float)Math.Cos(-affineAngle);
        float ss = (float)Math.Sin(-affineAngle);

        float x = ((landmarkX - 0.5f) * cs + (landmarkY - 0.5f) * ss) * affineScale + affineXc;
        float y = ((landmarkX - 0.5f) * -ss + (landmarkY - 0.5f) * cs) * affineScale + affineYc;
        return (x, y);
    }

    /// <summary>
    /// Compute affine transform parameters from two keypoints.
    /// Matches ExtractROIFromBox in AiliaMediapipePoseWorldLandmarks.
    ///
    /// Python reference:
    ///   scale = dscale * sqrt((xc - x1)^2 + (yc - y1)^2) * 2
    ///   angle = atan2(yc - y1, xc - x1) - pi/2
    /// </summary>
    public (float scale, float angle) ComputeAffineParams(
        float kp1X, float kp1Y, float kp2X, float kp2Y, float dscale = 1.1f)
    {
        float scale = dscale * (float)Math.Sqrt(
            Math.Pow(kp1X - kp2X, 2) + Math.Pow(kp1Y - kp2Y, 2)) * 2;
        float angle = (float)Math.Atan2(kp1Y - kp2Y, kp1X - kp2X) - (float)Math.PI / 2f;
        return (scale, angle);
    }

    /// <summary>
    /// Compute derived keypoints (shoulder center, body center) from landmarks.
    /// Matches GetResult in AiliaMediapipePoseWorldLandmarks.
    /// </summary>
    public (float x, float y, float z, float confidence) ComputeShoulderCenter(
        TestLandmark leftShoulder, TestLandmark rightShoulder)
    {
        return (
            (leftShoulder.x + rightShoulder.x) / 2,
            (leftShoulder.y + rightShoulder.y) / 2,
            (leftShoulder.z + rightShoulder.z) / 2,
            Math.Min(leftShoulder.confidence, rightShoulder.confidence)
        );
    }

    public (float x, float y, float z, float confidence) ComputeBodyCenter(
        TestLandmark leftShoulder, TestLandmark rightShoulder,
        TestLandmark leftHip, TestLandmark rightHip)
    {
        return (
            (leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4,
            (leftShoulder.y + rightShoulder.y + leftHip.y + rightHip.y) / 4,
            (leftShoulder.z + rightShoulder.z + leftHip.z + rightHip.z) / 4,
            Math.Min(
                Math.Min(leftShoulder.confidence, rightShoulder.confidence),
                Math.Min(leftHip.confidence, rightHip.confidence))
        );
    }
}
