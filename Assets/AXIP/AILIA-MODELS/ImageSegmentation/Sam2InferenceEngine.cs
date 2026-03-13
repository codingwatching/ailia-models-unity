/* SAM2 Shared Inference Engine */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * Shared logic for SAM2 image segmentation, used by both Unity and standalone tests.
 * Contains:
 *   - ISam2Backend interface (backend-agnostic model inference)
 *   - Data structs (TensorShape4D, EncoderOutput, PromptEncoderOutput, DecoderOutput)
 *   - Sam2InferenceEngine class (all pure computational methods)
 *
 * The logic matches the Python SAM2 reference implementation:
 *   - sam2/sam2_image_predictor.py
 *   - sam2/modeling/sam2_base.py
 *
 * Unity: included directly via the project
 * Tests: linked via <Compile Include="...Sam2InferenceEngine.cs" Link="..." />
 */

using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

// Unity type stubs for non-Unity builds (standalone tests, console apps)
#if !UNITY_2017_1_OR_NEWER
namespace UnityEngine
{
    public class Debug
    {
        public static void Log(string text) { System.Console.WriteLine(text); }
        public static void LogError(string text) { System.Console.WriteLine(text); }
        public static void LogWarning(string text) { System.Console.WriteLine(text); }
        public static void Assert(bool condition, object message) { System.Diagnostics.Debug.Assert(condition, message.ToString()); }
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct Color32
    {
        public byte r, g, b, a;
        public Color32(byte r, byte g, byte b, byte a) { this.r = r; this.g = g; this.b = b; this.a = a; }
    }

    public struct Vector2Int
    {
        public int x, y;
        public Vector2Int(int x, int y) { this.x = x; this.y = y; }
    }

    public struct Rect
    {
        public float x, y, width, height;
        public float xMin => x;
        public float yMin => y;
        public float xMax => x + width;
        public float yMax => y + height;
        public Rect(float x, float y, float width, float height) { this.x = x; this.y = y; this.width = width; this.height = height; }
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
#endif

// ============================================================
// Backend interface and data structs
// ============================================================

/// <summary>
/// Shape descriptor for a 4D tensor (N, C, H, W).
/// </summary>
public struct TensorShape4D
{
    public int N, C, H, W;
    public int TotalSize => N * C * H * W;
    public TensorShape4D(int n, int c, int h, int w) { N = n; C = c; H = h; W = w; }
}

/// <summary>
/// Encoder output containing all backbone features.
/// </summary>
public struct EncoderOutput
{
    public float[] VisionFeatures;
    public TensorShape4D VisionFeaturesShape;
    public float[] BackboneFpn0, BackboneFpn1, BackboneFpn2;
    public TensorShape4D Fpn0Shape, Fpn1Shape, Fpn2Shape;
}

/// <summary>
/// Prompt encoder output.
/// </summary>
public struct PromptEncoderOutput
{
    public float[] SparseEmbeddings;
    public int[] SparseShape; // [batch, tokens, dim]
    public float[] DenseEmbeddings;
    public int[] DenseShape; // [batch, channels, h, w]
    public float[] DensePe;
    public int[] DensePeShape; // [batch, channels, h, w]
}

/// <summary>
/// Mask decoder output.
/// </summary>
public struct DecoderOutput
{
    public float[] Masks;
    public int[] MasksShape; // [batch, num_masks, h, w]
    public float[] IouPred;
}

/// <summary>
/// Backend-agnostic interface for SAM2 model inference.
/// Implementations: OrtSam2Backend (OnnxRuntime), AiliaSam2Backend (ailia SDK)
/// </summary>
public interface ISam2Backend : IDisposable
{
    void LoadModels(string encoderPath, string decoderPath, string promptEncoderPath);
    EncoderOutput RunEncoder(float[] nchwInput);
    PromptEncoderOutput RunPromptEncoder(
        float[] flatCoords, int pointCount, float[] labels,
        float[] maskInput, float masksEnable);
    DecoderOutput RunDecoder(
        float[] imageEmbeddings,
        float[] imagePe, int[] imagePeShape,
        float[] sparseEmbeddings, int[] sparseShape,
        float[] denseEmbeddings, int[] denseShape,
        float[] highResFeatures1, // (1, 32, 256, 256)
        float[] highResFeatures2  // (1, 64, 128, 128)
    );
}

// ============================================================
// Sam2InferenceEngine: all pure computational logic
// ============================================================

/// <summary>
/// Pure computational logic for SAM2 image segmentation.
/// All methods match the Python SAM2 reference implementation.
/// Shared between Unity (SegmentAnything2Model) and standalone tests.
/// </summary>
public class Sam2InferenceEngine
{
    // ImageNet normalization constants
    // Python SAM2 (sam2/modeling/sam2_base.py):
    //   pixel_mean = torch.tensor([123.675, 116.28, 103.53]) / 255.0
    //   pixel_std  = torch.tensor([58.395, 57.12, 57.375])  / 255.0
    public readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    public readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    public int TargetSize { get; set; } = 1024;

    private List<Vector2Int> clickPoints = new();
    private List<bool> clickPointLabels = new();
    private Rect boxCoords = new();
    private bool addBoxCoords => boxCoords.width > 0 && boxCoords.height > 0;

    // ---------------------------------------------------
    // Click point / box management
    // ---------------------------------------------------
    public void AddClickPoint(int x, int y, bool negativePoint = false)
    {
        clickPoints.Add(new Vector2Int(x, y));
        clickPointLabels.Add(!negativePoint);
    }

    public void SetBoxCoords(Rect box) => boxCoords = box;

    public void ResetClickPoint()
    {
        clickPoints = new();
        clickPointLabels = new();
        boxCoords = new();
    }

    public float[,] GetClickPoints(int imageHeight)
    {
        float[,] points = new float[clickPoints.Count + (addBoxCoords ? 2 : 0), 2];
        int i = 0;
        foreach (var point in clickPoints)
        {
            points[i, 0] = point.x;
            points[i, 1] = point.y;
            i += 1;
        }
        if (addBoxCoords)
        {
            points[clickPoints.Count, 0] = boxCoords.xMin;
            points[clickPoints.Count, 1] = boxCoords.yMin;
            points[clickPoints.Count + 1, 0] = boxCoords.xMax;
            points[clickPoints.Count + 1, 1] = boxCoords.yMax;
        }
        return points;
    }

    public float[] GetPointLabels()
    {
        float[] labels = new float[clickPoints.Count + (addBoxCoords ? 2 : 0)];
        for (int i = 0; i < clickPoints.Count; ++i)
            labels[i] = clickPointLabels[i] ? 1f : 0f;
        if (addBoxCoords)
        {
            labels[clickPoints.Count] = 2;
            labels[clickPoints.Count + 1] = 3;
        }
        return labels;
    }

    // ---------------------------------------------------
    // Image preprocessing
    // Python: img / 255.0 → (val - mean) / std → CHW → batch
    // ---------------------------------------------------
    public float[,,] Color32ArrayToFloatArray(Color32[] image, int height, int width)
    {
        float[,,] result = new float[height, width, 3];
        for (int i = 0; i < height * width; i++)
        {
            Color32 c = image[i];
            int h = i / width;
            int w = i % width;
            result[h, w, 0] = c.r / 255f;
            result[h, w, 1] = c.g / 255f;
            result[h, w, 2] = c.b / 255f;
        }
        return result;
    }

    public float[,,] TransposeHWCtoCHW(float[,,] input)
    {
        int H = input.GetLength(0);
        int W = input.GetLength(1);
        int C = input.GetLength(2);
        float[,,] output = new float[C, H, W];
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                for (int c = 0; c < C; c++)
                    output[c, h, w] = input[h, w, c];
        return output;
    }

    public float[,,] ResizeBilinearHWC(float[,,] src, int srcHeight, int srcWidth, int dstHeight, int dstWidth)
    {
        float[,,] dst = new float[dstHeight, dstWidth, 3];
        float scaleY = (float)srcHeight / dstHeight;
        float scaleX = (float)srcWidth / dstWidth;

        for (int y = 0; y < dstHeight; y++)
        {
            float srcY = (y + 0.5f) * scaleY - 0.5f;
            srcY = Math.Max(0, Math.Min(srcY, srcHeight - 1));
            int y0 = (int)Math.Floor(srcY);
            int y1 = Math.Min(y0 + 1, srcHeight - 1);
            float dy = srcY - y0;

            for (int x = 0; x < dstWidth; x++)
            {
                float srcX = (x + 0.5f) * scaleX - 0.5f;
                srcX = Math.Max(0, Math.Min(srcX, srcWidth - 1));
                int x0 = (int)Math.Floor(srcX);
                int x1 = Math.Min(x0 + 1, srcWidth - 1);
                float dx = srcX - x0;

                for (int c = 0; c < 3; c++)
                {
                    float top = (1 - dx) * src[y0, x0, c] + dx * src[y0, x1, c];
                    float bottom = (1 - dx) * src[y1, x0, c] + dx * src[y1, x1, c];
                    dst[y, x, c] = (1 - dy) * top + dy * bottom;
                }
            }
        }
        return dst;
    }

    // ---------------------------------------------------
    // Pixel order conversion (Unity B2T <-> SAM2 T2B)
    // ---------------------------------------------------
    public static Color32[] VerticalFlip(Color32[] inputImage, int width, int height)
    {
        Color32[] outputImage = new Color32[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                outputImage[(height - 1 - y) * width + x] = inputImage[y * width + x];
        return outputImage;
    }

    public static bool[,] VerticalFlipMask(bool[,] mask)
    {
        int height = mask.GetLength(0);
        int width = mask.GetLength(1);
        bool[,] flipped = new bool[height, width];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                flipped[height - 1 - y, x] = mask[y, x];
        return flipped;
    }

    public float[,,,] PreprocessImage(Color32[] inputImage, int originalWidth, int originalHeight, int imageSize)
    {
        // Match SAM2 original: ToTensor -> Resize(bilinear) -> Normalize
        // 1. Convert to float [0,1]
        float[,,] floatImage = Color32ArrayToFloatArray(inputImage, originalHeight, originalWidth);

        // 2. Bilinear resize to (imageSize, imageSize)
        float[,,] resizedImage = ResizeBilinearHWC(floatImage, originalHeight, originalWidth, imageSize, imageSize);

        // 3. Normalize
        for (int h = 0; h < imageSize; h++)
            for (int w = 0; w < imageSize; w++)
                for (int c = 0; c < 3; c++)
                    resizedImage[h, w, c] = (resizedImage[h, w, c] - Mean[c]) / Std[c];

        float[,,] chw = TransposeHWCtoCHW(resizedImage);

        float[,,,] batch = new float[1, 3, imageSize, imageSize];
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < imageSize; h++)
                for (int w = 0; w < imageSize; w++)
                    batch[0, c, h, w] = chw[c, h, w];

        return batch;
    }

    // ---------------------------------------------------
    // Coordinate scaling
    // Python: coords * target_size / original_size
    // ---------------------------------------------------
    public float[,] ApplyCoordinateScaling(float[,] coords, int imgHeight, int imgWidth)
    {
        float[,] scaledCoords = new float[coords.GetLength(0), coords.GetLength(1)];
        for (int i = 0; i < coords.GetLength(0); i++)
        {
            scaledCoords[i, 0] = coords[i, 0] * TargetSize / imgWidth;
            scaledCoords[i, 1] = coords[i, 1] * TargetSize / imgHeight;
        }
        return scaledCoords;
    }

    public float GetScale(int texWidth, int texHeight)
    {
        return TargetSize / (float)Math.Max(texWidth, texHeight);
    }

    // ---------------------------------------------------
    // Mask postprocessing
    // Python: F.interpolate(masks, (H, W), mode="bilinear")
    // ---------------------------------------------------
    public float[,] ResizeBilinear(float[,] src, int targetHeight, int targetWidth)
    {
        int srcHeight = src.GetLength(0);
        int srcWidth = src.GetLength(1);
        float[,] result = new float[targetHeight, targetWidth];

        float scaleY = (float)srcHeight / targetHeight;
        float scaleX = (float)srcWidth / targetWidth;

        for (int y = 0; y < targetHeight; y++)
        {
            float srcY = (y + 0.5f) * scaleY - 0.5f;
            srcY = Math.Max(0, Math.Min(srcY, srcHeight - 1));
            int y0 = (int)Math.Floor(srcY);
            int y1 = Math.Min(y0 + 1, srcHeight - 1);
            float dy = srcY - y0;

            for (int x = 0; x < targetWidth; x++)
            {
                float srcX = (x + 0.5f) * scaleX - 0.5f;
                srcX = Math.Max(0, Math.Min(srcX, srcWidth - 1));
                int x0 = (int)Math.Floor(srcX);
                int x1 = Math.Min(x0 + 1, srcWidth - 1);
                float dx = srcX - x0;

                float top = (1 - dx) * src[y0, x0] + dx * src[y0, x1];
                float bottom = (1 - dx) * src[y1, x0] + dx * src[y1, x1];
                result[y, x] = (1 - dy) * top + dy * bottom;
            }
        }
        return result;
    }

    public float[,,,] PostprocessMasks(float[,,,] masks, int targetHeight, int targetWidth)
    {
        int batch = masks.GetLength(0);
        int channels = masks.GetLength(1);
        int height = masks.GetLength(2);
        int width = masks.GetLength(3);

        float[,,,] resizedMasks = new float[batch, channels, targetHeight, targetWidth];

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                float[,] mask2D = new float[height, width];
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                        mask2D[y, x] = masks[n, c, y, x];

                float[,] resized2D = ResizeBilinear(mask2D, targetHeight, targetWidth);

                for (int y = 0; y < targetHeight; y++)
                    for (int x = 0; x < targetWidth; x++)
                        resizedMasks[n, c, y, x] = resized2D[y, x];
            }
        }
        return resizedMasks;
    }

    // Python: masks > 0.0
    public bool[][,] ConvertToBoolMasks(float[,,,] masks, float threshold = 0.0f)
    {
        int batch = masks.GetLength(0);
        int channels = masks.GetLength(1);
        int height = masks.GetLength(2);
        int width = masks.GetLength(3);

        bool[][,] result = new bool[batch * channels][,];
        int index = 0;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                bool[,] binaryMask = new bool[height, width];
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                        binaryMask[y, x] = masks[b, c, y, x] > threshold;
                result[index++] = binaryMask;
            }
        }
        return result;
    }

    // ---------------------------------------------------
    // Tensor operations
    // ---------------------------------------------------
    public float[] Flatten4D(float[,,,] input)
    {
        int N = input.GetLength(0), C = input.GetLength(1);
        int H = input.GetLength(2), W = input.GetLength(3);
        float[] flat = new float[N * C * H * W];
        int idx = 0;
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        flat[idx++] = input[n, c, h, w];
        return flat;
    }

    public float[,,,] ReshapeTo4D(float[] array, int w, int z, int y, int x)
    {
        if (array.Length != w * z * y * x)
            throw new ArgumentException("flatArray length does not match the product of dimensions");

        float[,,,] array4D = new float[w, z, y, x];
        int index = 0;
        for (int i = 0; i < w; i++)
            for (int j = 0; j < z; j++)
                for (int k = 0; k < y; k++)
                    for (int l = 0; l < x; l++)
                        array4D[i, j, k, l] = array[index++];
        return array4D;
    }

    public float[,,] ReshapeTo3D(float[] flat, int z, int y, int x)
    {
        if (flat.Length != z * y * x)
            throw new ArgumentException($"Input length {flat.Length} does not match shape ({z}, {y}, {x})");

        float[,,] result = new float[z, y, x];
        int index = 0;
        for (int i = 0; i < z; i++)
            for (int j = 0; j < y; j++)
                for (int k = 0; k < x; k++)
                    result[i, j, k] = flat[index++];
        return result;
    }

    public float[,] ReshapeTo2D(float[] flat, int rows, int cols)
    {
        if (flat.Length != rows * cols)
            throw new ArgumentException("Size mismatch between flat array and target shape.");

        float[,] result = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r, c] = flat[r * cols + c];
        return result;
    }

    public float[,,] BroadcastAdd3D(float[,,] a, float[,,] b)
    {
        int a0 = a.GetLength(0), a1 = a.GetLength(1), a2 = a.GetLength(2);
        int b0 = b.GetLength(0), b1 = b.GetLength(1), b2 = b.GetLength(2);

        if ((b0 != 1 && b0 != a0) || (b1 != 1 && b1 != a1) || (b2 != 1 && b2 != a2))
            throw new ArgumentException("Incompatible shapes for broadcasting.");

        float[,,] result = new float[a0, a1, a2];
        for (int i = 0; i < a0; i++)
            for (int j = 0; j < a1; j++)
                for (int k = 0; k < a2; k++)
                    result[i, j, k] = a[i, j, k] + b[b0 == 1 ? 0 : i, b1 == 1 ? 0 : j, b2 == 1 ? 0 : k];
        return result;
    }

    // Python: np.transpose(arr, (2, 0, 1))
    public float[,,] Transpose201(float[,,] input)
    {
        int d0 = input.GetLength(0), d1 = input.GetLength(1), d2 = input.GetLength(2);
        float[,,] output = new float[d2, d0, d1];
        for (int i0 = 0; i0 < d0; i0++)
            for (int i1 = 0; i1 < d1; i1++)
                for (int i2 = 0; i2 < d2; i2++)
                    output[i2, i0, i1] = input[i0, i1, i2];
        return output;
    }

    // Python: tensor.reshape(d0, d1, d2*d3)
    public float[,,] FourReshapeTo3D(float[,,,] input)
    {
        int d0 = input.GetLength(0), d1 = input.GetLength(1);
        int d2 = input.GetLength(2), d3 = input.GetLength(3);
        int d23 = d2 * d3;
        float[,,] output = new float[d0, d1, d23];
        for (int i0 = 0; i0 < d0; i0++)
            for (int i1 = 0; i1 < d1; i1++)
                for (int i2 = 0; i2 < d2; i2++)
                    for (int i3 = 0; i3 < d3; i3++)
                        output[i0, i1, i2 * d3 + i3] = input[i0, i1, i2, i3];
        return output;
    }

    // Python: [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
    public float[][,,] ProcessVisionPosEmbeds(float[][,,,] visionPosEmbeds4D)
    {
        int n = visionPosEmbeds4D.Length;
        float[][,,] output = new float[n][,,];
        for (int i = 0; i < n; i++)
        {
            float[,,] reshaped = FourReshapeTo3D(visionPosEmbeds4D[i]);
            float[,,] transposed = Transpose201(reshaped);
            output[i] = transposed;
        }
        return output;
    }

    // Python: x.flatten(2).permute(2, 0, 1) for each backbone FPN feature
    public float[][,,] PrepareBackboneFeatures(float[][,,,] backboneFpn)
    {
        int numFeatureLevels = 3;
        float[][,,,] featureMaps = backboneFpn.TakeLast(numFeatureLevels).ToArray();
        float[][,,] visionFeats = new float[numFeatureLevels][,,];

        for (int i = 0; i < numFeatureLevels; i++)
        {
            float[,,,] x = featureMaps[i];
            int N = x.GetLength(0), C = x.GetLength(1);
            int H = x.GetLength(2), W = x.GetLength(3);
            int HW = H * W;

            float[,,] outputFeat = new float[HW, N, C];
            for (int nn = 0; nn < N; nn++)
                for (int c = 0; c < C; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            outputFeat[h * W + w, nn, c] = x[nn, c, h, w];

            visionFeats[i] = outputFeat;
        }
        return visionFeats;
    }

    // ---------------------------------------------------
    // Full inference pipeline orchestration
    // ---------------------------------------------------

    /// <summary>
    /// Prepare encoder output into image embeddings and high-res features.
    /// Returns (imageEmbFlat, highResFeats) where highResFeats[0] = (1,32,256,256) and highResFeats[1] = (1,64,128,128).
    /// </summary>
    public (float[] imageEmbFlat, float[][] highResFeats) PrepareEncoderFeatures(EncoderOutput encOut)
    {
        float[,,,] fpn0_4d = ReshapeTo4D(encOut.BackboneFpn0,
            encOut.Fpn0Shape.N, encOut.Fpn0Shape.C, encOut.Fpn0Shape.H, encOut.Fpn0Shape.W);
        float[,,,] fpn1_4d = ReshapeTo4D(encOut.BackboneFpn1,
            encOut.Fpn1Shape.N, encOut.Fpn1Shape.C, encOut.Fpn1Shape.H, encOut.Fpn1Shape.W);
        float[,,,] fpn2_4d = ReshapeTo4D(encOut.BackboneFpn2,
            encOut.Fpn2Shape.N, encOut.Fpn2Shape.C, encOut.Fpn2Shape.H, encOut.Fpn2Shape.W);

        float[][,,,] backboneFpn = new[] { fpn0_4d, fpn1_4d, fpn2_4d };
        float[][,,] visionFeats = PrepareBackboneFeatures(backboneFpn);

        // Note: no_mem_embed is NOT added for single-image inference

        (int H, int W)[] bbFeatSizes = { (256, 256), (128, 128), (64, 64) };

        float[][,,,] featsArray = new float[visionFeats.Length][,,,];
        for (int i = 0; i < visionFeats.Length; i++)
        {
            int revIdx = visionFeats.Length - 1 - i;
            float[,,] feat = visionFeats[revIdx];
            (int fH, int fW) = bbFeatSizes[revIdx];
            int HW = feat.GetLength(0);
            int C = feat.GetLength(2);

            float[,,] transposed = new float[1, C, HW];
            for (int hw = 0; hw < HW; hw++)
                for (int c = 0; c < C; c++)
                    transposed[0, c, hw] = feat[hw, 0, c];

            float[,,,] reshaped = new float[1, C, fH, fW];
            for (int hw = 0; hw < HW; hw++)
            {
                int h = hw / fW;
                int w = hw % fW;
                for (int c = 0; c < C; c++)
                    reshaped[0, c, h, w] = transposed[0, c, hw];
            }
            featsArray[i] = reshaped;
        }
        Array.Reverse(featsArray);

        float[] imageEmbFlat = Flatten4D(featsArray[^1]);
        float[][] highResFeatsResult = new float[featsArray.Length - 1][];
        for (int i = 0; i < featsArray.Length - 1; i++)
            highResFeatsResult[i] = Flatten4D(featsArray[i]);

        return (imageEmbFlat, highResFeatsResult);
    }

    /// <summary>
    /// Flatten scaled coordinates for prompt encoder input.
    /// </summary>
    public float[] FlattenCoords(float[,] scaledCoords, int pointCount)
    {
        float[] flattenedCoords = new float[pointCount * 2];
        for (int i = 0; i < pointCount; i++)
        {
            flattenedCoords[i * 2] = scaledCoords[i, 0];
            flattenedCoords[i * 2 + 1] = scaledCoords[i, 1];
        }
        return flattenedCoords;
    }

    /// <summary>
    /// Run the full inference pipeline: encoder → prompt encoder → decoder → postprocess.
    /// Returns (boolMasks, iouPred).
    /// </summary>
    public (bool[][,] masks, float[] iouPred) RunFullPipeline(
        ISam2Backend backend,
        Color32[] image, int imgWidth, int imgHeight,
        float[,] pointCoords, float[] pointLabels)
    {
        // Step 1: Preprocess and encode
        float[,,,] inputTensor = PreprocessImage(image, imgWidth, imgHeight, TargetSize);
        float[] nchwInput = Flatten4D(inputTensor);

        var encOut = backend.RunEncoder(nchwInput);
        var (imageEmbFlat, highResFeats) = PrepareEncoderFeatures(encOut);

        // Step 2: Coordinate scaling + prompt encoder
        float[,] scaledCoords = ApplyCoordinateScaling(pointCoords, imgHeight, imgWidth);
        int pointCount = pointLabels.Length;
        float[] flatCoords = FlattenCoords(scaledCoords, pointCount);
        float[] maskInputDummy = new float[256 * 256];

        var promptOut = backend.RunPromptEncoder(flatCoords, pointCount, pointLabels, maskInputDummy, 0f);

        // Step 3: Decoder
        var decOut = backend.RunDecoder(
            imageEmbFlat,
            promptOut.DensePe, promptOut.DensePeShape,
            promptOut.SparseEmbeddings, promptOut.SparseShape,
            promptOut.DenseEmbeddings, promptOut.DenseShape,
            highResFeats[0],
            highResFeats[1]
        );

        // Step 4: Postprocess
        float[,,,] masks4d = ReshapeTo4D(decOut.Masks,
            decOut.MasksShape[0], decOut.MasksShape[1],
            decOut.MasksShape[2], decOut.MasksShape[3]);
        float[,,,] resizedMasks = PostprocessMasks(masks4d, imgHeight, imgWidth);
        bool[][,] boolMasks = ConvertToBoolMasks(resizedMasks, 0.0f);

        return (boolMasks, decOut.IouPred);
    }

    /// <summary>
    /// Find the best mask index based on IoU predictions.
    /// </summary>
    public int FindBestMaskIndex(float[] iouPred)
    {
        int bestIdx = 0;
        float maxScore = float.MinValue;
        for (int i = 0; i < iouPred.Length; i++)
            if (iouPred[i] > maxScore) { maxScore = iouPred[i]; bestIdx = i; }
        return bestIdx;
    }
}
