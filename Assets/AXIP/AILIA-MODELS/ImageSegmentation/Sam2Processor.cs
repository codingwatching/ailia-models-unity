/* SAM2 Processor */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * Core SAM2 processing pipeline, independent of Unity.
 * Takes ISam2Backend for model inference and Sam2InferenceEngine for computation.
 *
 * Handles:
 *   - Click point / box management
 *   - ProcessEmbedding: B2T -> T2B flip -> encode -> prepare features
 *   - ProcessMask: coords -> inference -> best mask -> T2B -> B2T flip
 *   - Mask overlay on Color32[] pixels
 *   - Click point visualization on Color32[] pixels
 *
 * Unity: included directly via the project
 * Tests: linked via <Compile Include="...Sam2Processor.cs" Link="..." />
 */

using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Result of ProcessMask.
/// </summary>
public struct Sam2MaskResult
{
    public bool HasMask;
    public bool[,] B2TMask;
    public int BestMaskIndex;
    public float BestScore;
    public float[] IouPredictions;
}

/// <summary>
/// Core SAM2 processing pipeline. Uses ISam2Backend for inference.
/// All methods are pure (no Unity Texture2D, MonoBehaviour, etc.).
/// </summary>
public class Sam2Processor
{
    private readonly ISam2Backend backend;
    private readonly Sam2InferenceEngine engine = new Sam2InferenceEngine();
    private readonly int targetSize = 1024;

    private float[] encoderOutput = null;
    private float[][] highResFeats = null;

    public Sam2Processor(ISam2Backend backend)
    {
        this.backend = backend ?? throw new ArgumentNullException(nameof(backend));
    }

    // ---------------------------------------------------
    // Click point / box management (delegated to engine)
    // ---------------------------------------------------
    public void AddClickPoint(int x, int y, bool negativePoint = false)
        => engine.AddClickPoint(x, y, negativePoint);

    public void SetBoxCoords(Rect box)
        => engine.SetBoxCoords(box);

    public void ResetClickPoint()
        => engine.ResetClickPoint();

    public float[,] GetClickPoints(int imageHeight)
        => engine.GetClickPoints(imageHeight);

    public float[] GetPointLabels()
        => engine.GetPointLabels();

    // ---------------------------------------------------
    // Embedding
    // ---------------------------------------------------

    /// <summary>
    /// Check if embedding has been computed.
    /// </summary>
    public bool EmbeddingExist() => encoderOutput != null;

    /// <summary>
    /// Compute embedding from B2T image (Unity native pixel order).
    /// Internally flips B2T -> T2B for SAM2 inference.
    /// </summary>
    public void ProcessEmbedding(Color32[] b2tImage, int imgWidth, int imgHeight)
    {
        Color32[] t2bImage = Sam2InferenceEngine.VerticalFlip(b2tImage, imgWidth, imgHeight);
        RunEmbedding(t2bImage, imgWidth, imgHeight);
    }

    private void RunEmbedding(Color32[] t2bImage, int imgWidth, int imgHeight)
    {
        float[,,,] inputTensor = engine.PreprocessImage(t2bImage, imgWidth, imgHeight, targetSize);
        float[] nchwInput = engine.Flatten4D(inputTensor);

        var encOut = backend.RunEncoder(nchwInput);
        var (imageEmbFlat, highResFeatsResult) = engine.PrepareEncoderFeatures(encOut);
        encoderOutput = imageEmbFlat;
        highResFeats = highResFeatsResult;
    }

    // ---------------------------------------------------
    // Mask inference
    // ---------------------------------------------------

    /// <summary>
    /// Compute mask from B2T image using stored embedding and click points.
    /// Returns B2T mask ready for overlay on B2T pixels.
    /// </summary>
    public Sam2MaskResult ProcessMask(int imgWidth, int imgHeight)
    {
        float[,] coords = engine.GetClickPoints(imgHeight);
        float[] labels = engine.GetPointLabels();

        if (coords.GetLength(0) == 0)
            return new Sam2MaskResult { HasMask = false };

        var (masks, iouPred) = RunInference(imgWidth, imgHeight, coords, labels);

        if (masks == null || masks.Length == 0 || iouPred == null || iouPred.Length == 0)
            return new Sam2MaskResult { HasMask = false };

        int bestIdx = engine.FindBestMaskIndex(iouPred);
        bool[,] b2tMask = Sam2InferenceEngine.VerticalFlipMask(masks[bestIdx]);

        return new Sam2MaskResult
        {
            HasMask = true,
            B2TMask = b2tMask,
            BestMaskIndex = bestIdx,
            BestScore = iouPred[bestIdx],
            IouPredictions = iouPred,
        };
    }

    private (bool[][,] masks, float[] iouPred) RunInference(
        int imgWidth, int imgHeight,
        float[,] pointCoords, float[] pointLabels)
    {
        float[,] scaledCoords = engine.ApplyCoordinateScaling(pointCoords, imgHeight, imgWidth);
        int pointCount = pointLabels.Length;
        float[] flatCoords = engine.FlattenCoords(scaledCoords, pointCount);
        float[] maskInputDummy = new float[256 * 256];

        var promptOut = backend.RunPromptEncoder(flatCoords, pointCount, pointLabels, maskInputDummy, 0f);

        var decOut = backend.RunDecoder(
            encoderOutput,
            promptOut.DensePe, promptOut.DensePeShape,
            promptOut.SparseEmbeddings, promptOut.SparseShape,
            promptOut.DenseEmbeddings, promptOut.DenseShape,
            highResFeats[0],
            highResFeats[1]
        );

        float[,,,] masks4d = engine.ReshapeTo4D(decOut.Masks,
            decOut.MasksShape[0], decOut.MasksShape[1],
            decOut.MasksShape[2], decOut.MasksShape[3]);
        float[,,,] resizedMasks = engine.PostprocessMasks(masks4d, imgHeight, imgWidth);
        bool[][,] boolMasks = engine.ConvertToBoolMasks(resizedMasks, 0.0f);

        return (boolMasks, decOut.IouPred);
    }

    // ---------------------------------------------------
    // Visualization helpers (pure Color32[] operations)
    // ---------------------------------------------------

    private static readonly Color32 DefaultMaskColor = new Color32(255, 0, 0, 255);

    /// <summary>
    /// Overlay mask on pixels. Both must be in the same pixel order (B2T).
    /// Returns a new Color32[] with mask applied.
    /// </summary>
    public static Color32[] ApplyMaskOverlay(
        bool[,] mask, Color32[] pixels, int imageWidth, int imageHeight)
    {
        return ApplyMaskOverlay(mask, pixels, imageWidth, imageHeight, DefaultMaskColor);
    }

    public static Color32[] ApplyMaskOverlay(
        bool[,] mask, Color32[] pixels, int imageWidth, int imageHeight, Color32 maskColor)
    {
        Color32[] result = (Color32[])pixels.Clone();
        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        for (int y = 0; y < maskHeight; y++)
        {
            int rowOffset = y * imageWidth;
            for (int x = 0; x < maskWidth; x++)
            {
                int pixelIndex = rowOffset + x;
                if (pixelIndex >= 0 && pixelIndex < result.Length && mask[y, x])
                {
                    Color32 orig = result[pixelIndex];
                    result[pixelIndex] = new Color32(
                        (byte)(orig.r * 0.6f + maskColor.r * 0.4f),
                        (byte)(orig.g * 0.6f + maskColor.g * 0.4f),
                        (byte)(orig.b * 0.6f + maskColor.b * 0.4f),
                        maskColor.a
                    );
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Convert T2B click coordinates to B2T for visualization.
    /// </summary>
    public static float[,] ConvertClickCoordsToB2T(float[,] t2bCoords, int imageHeight)
    {
        int count = t2bCoords.GetLength(0);
        float[,] b2tCoords = new float[count, 2];
        for (int i = 0; i < count; i++)
        {
            b2tCoords[i, 0] = t2bCoords[i, 0];
            b2tCoords[i, 1] = imageHeight - 1 - t2bCoords[i, 1];
        }
        return b2tCoords;
    }

    /// <summary>
    /// Draw click point markers on pixel array. Both coords and pixels should be in the same order (B2T).
    /// Returns a new Color32[] with markers drawn.
    /// </summary>
    public static Color32[] DrawClickPoints(
        float[,] coords, float[] labels, Color32[] pixels, int imageWidth, int imageHeight)
    {
        Color32[] result = (Color32[])pixels.Clone();
        int numPoints = coords.GetLength(0);
        int markerSize = 15;

        for (int i = 0; i < numPoints; i++)
        {
            if (labels[i] >= 2)
                continue;

            int px = Math.Max(0, Math.Min((int)coords[i, 0], imageWidth - 1));
            int py = Math.Max(0, Math.Min((int)coords[i, 1], imageHeight - 1));

            Color32 markerColor = labels[i] == 1
                ? new Color32(0, 255, 0, 255)
                : new Color32(0, 0, 255, 255);

            for (int dy = -markerSize; dy <= markerSize; dy++)
            {
                for (int dx = -markerSize; dx <= markerSize; dx++)
                {
                    if (Math.Abs(dx) == Math.Abs(dy))
                    {
                        int nx = px + dx;
                        int ny = py + dy;
                        if (nx >= 0 && nx < imageWidth && ny >= 0 && ny < imageHeight)
                        {
                            result[ny * imageWidth + nx] = markerColor;
                        }
                    }
                }
            }
        }
        return result;
    }
}
