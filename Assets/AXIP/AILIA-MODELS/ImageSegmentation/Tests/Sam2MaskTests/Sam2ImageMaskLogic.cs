/* SAM2 Image Mask Processing Logic (Unity-independent) */
/* Extracted from SegmentAnything2Model.cs for unit testing */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * This class contains the pure computational logic from SAM2 model
 * without Unity dependencies (no Texture2D, Color32, MonoBehaviour, etc.).
 * The logic must match the Python SAM2 reference implementation:
 *   - sam2/sam2_image_predictor.py
 *   - sam2/modeling/sam2_base.py
 */

using System;
using System.Collections.Generic;
using System.Linq;

namespace Sam2MaskTests
{
    /// <summary>
    /// Lightweight replacement for UnityEngine.Color32 used in pixel processing.
    /// </summary>
    public struct Color32
    {
        public byte r, g, b, a;
        public Color32(byte r, byte g, byte b, byte a) { this.r = r; this.g = g; this.b = b; this.a = a; }
    }

    /// <summary>
    /// Lightweight replacement for UnityEngine.Rect.
    /// </summary>
    public struct Rect
    {
        public float x, y, width, height;
        public float xMin => x;
        public float yMin => y;
        public float xMax => x + width;
        public float yMax => y + height;
        public Rect(float x, float y, float width, float height) { this.x = x; this.y = y; this.width = width; this.height = height; }
    }

    /// <summary>
    /// Lightweight replacement for UnityEngine.Vector2Int.
    /// </summary>
    public struct Vector2Int
    {
        public int x, y;
        public Vector2Int(int x, int y) { this.x = x; this.y = y; }
    }

    /// <summary>
    /// Pure computational logic extracted from SegmentAnything2Model.
    /// All methods match the Python SAM2 reference implementation.
    /// </summary>
    public class Sam2ImageMaskLogic
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
            float scaleY = dstHeight > 1 ? (float)(srcHeight - 1) / (dstHeight - 1) : 0;
            float scaleX = dstWidth > 1 ? (float)(srcWidth - 1) / (dstWidth - 1) : 0;

            for (int y = 0; y < dstHeight; y++)
            {
                float srcY = y * scaleY;
                int y0 = (int)Math.Floor(srcY);
                int y1 = Math.Min(y0 + 1, srcHeight - 1);
                float dy = srcY - y0;

                for (int x = 0; x < dstWidth; x++)
                {
                    float srcX = x * scaleX;
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

            float scaleY = (float)(srcHeight - 1) / (targetHeight - 1);
            float scaleX = (float)(srcWidth - 1) / (targetWidth - 1);

            for (int y = 0; y < targetHeight; y++)
            {
                float srcY = y * scaleY;
                int y0 = (int)Math.Floor(srcY);
                int y1 = Math.Min(y0 + 1, srcHeight - 1);
                float dy = srcY - y0;

                for (int x = 0; x < targetWidth; x++)
                {
                    float srcX = x * scaleX;
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
    }
}
