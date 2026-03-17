/* SAM2 Test Helper */
/* Helper functions moved from production code (Sam2InferenceEngine / Sam2Processor) */
/* These functions are only used in tests. */

using System;
using UnityEngine;

public static class Sam2TestHelper
{
    // --- From Sam2InferenceEngine ---

    public static float[,,] Color32ArrayToFloatArray(Color32[] image, int height, int width)
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

    public static float[,,] TransposeHWCtoCHW(float[,,] input)
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

    public static float[,,] ResizeBilinearHWC(float[,,] src, int srcHeight, int srcWidth, int dstHeight, int dstWidth)
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

    public static float[,,] ReshapeTo3D(float[] flat, int z, int y, int x)
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

    public static float[,] ReshapeTo2D(float[] flat, int rows, int cols)
    {
        if (flat.Length != rows * cols)
            throw new ArgumentException("Size mismatch between flat array and target shape.");

        float[,] result = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r, c] = flat[r * cols + c];
        return result;
    }

    public static float[,,] BroadcastAdd3D(float[,,] a, float[,,] b)
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
    public static float[,,] Transpose201(float[,,] input)
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
    public static float[,,] FourReshapeTo3D(float[,,,] input)
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
    public static float[][,,] ProcessVisionPosEmbeds(float[][,,,] visionPosEmbeds4D)
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

    public static float GetScale(int texWidth, int texHeight, int targetSize = 1024)
    {
        return targetSize / (float)Math.Max(texWidth, texHeight);
    }

    // --- From Sam2Processor ---

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
}
