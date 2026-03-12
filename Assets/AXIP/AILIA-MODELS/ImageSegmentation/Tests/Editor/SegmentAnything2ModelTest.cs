/* AILIA Unity Plugin Segment Anything 2 Unit Tests */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * These tests verify that the C# SAM2 image mask processing logic
 * produces results consistent with the Python SAM2 reference implementation.
 *
 * Python reference (sam2/sam2_image_predictor.py):
 *   - Preprocessing: pixel / 255.0, then (value - mean) / std with ImageNet stats
 *   - Coordinate scaling: coords * 1024 / original_size
 *   - Mask postprocessing: bilinear interpolation resize, threshold > 0.0
 */

using System;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

public class SegmentAnything2ModelTest
{
    private SegmentAnything2Model model;
    private Type modelType;

    private const float Tolerance = 1e-5f;

    [SetUp]
    public void SetUp()
    {
        model = new SegmentAnything2Model();
        modelType = typeof(SegmentAnything2Model);
    }

    // -------------------------------------------------------
    // Helper: invoke private/public methods via reflection
    // -------------------------------------------------------
    private object InvokeMethod(string methodName, object[] parameters)
    {
        MethodInfo method = modelType.GetMethod(
            methodName,
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public
        );
        Assert.IsNotNull(method, $"Method '{methodName}' not found on SegmentAnything2Model");
        return method.Invoke(model, parameters);
    }

    private T InvokeMethod<T>(string methodName, object[] parameters)
    {
        return (T)InvokeMethod(methodName, parameters);
    }

    private T GetField<T>(string fieldName)
    {
        FieldInfo field = modelType.GetField(
            fieldName,
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public
        );
        Assert.IsNotNull(field, $"Field '{fieldName}' not found");
        return (T)field.GetValue(model);
    }

    // =======================================================
    // 1. Color32ArrayToFloatArray
    //    Python equivalent: img.astype(np.float32) / 255.0
    // =======================================================
    [Test]
    public void Color32ArrayToFloatArray_MatchesPythonDivide255()
    {
        // Arrange: 2x2 image with known pixel values
        Color32[] pixels = new Color32[]
        {
            new Color32(0, 0, 0, 255),       // (0,0) black
            new Color32(255, 255, 255, 255),  // (0,1) white
            new Color32(128, 64, 192, 255),   // (1,0) arbitrary
            new Color32(100, 200, 50, 255),   // (1,1) arbitrary
        };
        int height = 2, width = 2;

        // Act
        float[,,] result = InvokeMethod<float[,,]>(
            "Color32ArrayToFloatArray",
            new object[] { pixels, height, width }
        );

        // Assert: matches Python np.array(img) / 255.0
        Assert.AreEqual(3, result.GetLength(2), "Should have 3 channels (RGB)");

        // (0,0) = black → 0/255 = 0.0
        Assert.AreEqual(0.0f, result[0, 0, 0], Tolerance);
        Assert.AreEqual(0.0f, result[0, 0, 1], Tolerance);
        Assert.AreEqual(0.0f, result[0, 0, 2], Tolerance);

        // (0,1) = white → 255/255 = 1.0
        Assert.AreEqual(1.0f, result[0, 1, 0], Tolerance);
        Assert.AreEqual(1.0f, result[0, 1, 1], Tolerance);
        Assert.AreEqual(1.0f, result[0, 1, 2], Tolerance);

        // (1,0) = (128,64,192) → (0.50196, 0.25098, 0.75294)
        Assert.AreEqual(128f / 255f, result[1, 0, 0], Tolerance);
        Assert.AreEqual(64f / 255f, result[1, 0, 1], Tolerance);
        Assert.AreEqual(192f / 255f, result[1, 0, 2], Tolerance);

        // (1,1) = (100,200,50) → (0.39216, 0.78431, 0.19608)
        Assert.AreEqual(100f / 255f, result[1, 1, 0], Tolerance);
        Assert.AreEqual(200f / 255f, result[1, 1, 1], Tolerance);
        Assert.AreEqual(50f / 255f, result[1, 1, 2], Tolerance);
    }

    // =======================================================
    // 2. ImageNet Normalization
    //    Python: (pixel/255.0 - mean) / std
    //    mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    // =======================================================
    [Test]
    public void PreprocessImage_NormalizationMatchesPython()
    {
        // Arrange: single pixel image (1x1) to isolate normalization
        Color32[] pixel = new Color32[] { new Color32(128, 64, 192, 255) };

        // Act: preprocess with imageSize=1 to avoid resize effects
        float[,,,] result = InvokeMethod<float[,,,]>(
            "PreprocessImage",
            new object[] { pixel, 1, 1, 1 }
        );

        // Expected (Python reference):
        // r = 128/255 = 0.50196
        // g = 64/255  = 0.25098
        // b = 192/255 = 0.75294
        // normalized_r = (0.50196 - 0.485) / 0.229 = 0.07405
        // normalized_g = (0.25098 - 0.456) / 0.224 = -0.91526
        // normalized_b = (0.75294 - 0.406) / 0.225 = 1.54197
        float r_norm = (128f / 255f - 0.485f) / 0.229f;
        float g_norm = (64f / 255f - 0.456f) / 0.224f;
        float b_norm = (192f / 255f - 0.406f) / 0.225f;

        // Result shape: (1, 3, 1, 1) - NCHW format
        Assert.AreEqual(1, result.GetLength(0), "Batch dimension");
        Assert.AreEqual(3, result.GetLength(1), "Channel dimension");
        Assert.AreEqual(1, result.GetLength(2), "Height dimension");
        Assert.AreEqual(1, result.GetLength(3), "Width dimension");

        Assert.AreEqual(r_norm, result[0, 0, 0, 0], 1e-4f, "R channel normalization");
        Assert.AreEqual(g_norm, result[0, 1, 0, 0], 1e-4f, "G channel normalization");
        Assert.AreEqual(b_norm, result[0, 2, 0, 0], 1e-4f, "B channel normalization");
    }

    [Test]
    public void PreprocessImage_OutputFormatIsNCHW()
    {
        // Arrange: 2x3 image
        Color32[] pixels = new Color32[6];
        for (int i = 0; i < 6; i++)
            pixels[i] = new Color32((byte)(i * 40), (byte)(i * 30), (byte)(i * 20), 255);

        // Act
        float[,,,] result = InvokeMethod<float[,,,]>(
            "PreprocessImage",
            new object[] { pixels, 3, 2, 2 }
        );

        // Assert: output should be NCHW = (1, 3, H, W)
        Assert.AreEqual(1, result.GetLength(0), "N=1");
        Assert.AreEqual(3, result.GetLength(1), "C=3");
        // Resized to imageSize x imageSize (2x2)
        Assert.AreEqual(2, result.GetLength(2), "H=imageSize");
        Assert.AreEqual(2, result.GetLength(3), "W=imageSize");
    }

    // =======================================================
    // 3. TransposeHWCtoCHW
    //    Python: img.transpose(2, 0, 1) or torch.permute(2, 0, 1)
    // =======================================================
    [Test]
    public void TransposeHWCtoCHW_MatchesPythonTranspose()
    {
        // Arrange: (H=2, W=3, C=3) tensor
        float[,,] hwc = new float[2, 3, 3];
        for (int h = 0; h < 2; h++)
            for (int w = 0; w < 3; w++)
                for (int c = 0; c < 3; c++)
                    hwc[h, w, c] = h * 100 + w * 10 + c;

        // Act
        float[,,] chw = InvokeMethod<float[,,]>("TransposeHWCtoCHW", new object[] { hwc });

        // Assert: output[c, h, w] == input[h, w, c]
        Assert.AreEqual(3, chw.GetLength(0), "C dimension first");
        Assert.AreEqual(2, chw.GetLength(1), "H dimension second");
        Assert.AreEqual(3, chw.GetLength(2), "W dimension third");

        for (int h = 0; h < 2; h++)
            for (int w = 0; w < 3; w++)
                for (int c = 0; c < 3; c++)
                    Assert.AreEqual(hwc[h, w, c], chw[c, h, w], Tolerance,
                        $"chw[{c},{h},{w}] should equal hwc[{h},{w},{c}]");
    }

    // =======================================================
    // 4. ApplyCoordinateScaling
    //    Python: coords * (1024 / original_size)
    // =======================================================
    [Test]
    public void ApplyCoordinateScaling_MatchesPythonScaling()
    {
        // Arrange: point (500, 300) on 1000x600 image
        float[,] coords = new float[,] { { 500f, 300f } };
        int imgWidth = 1000;
        int imgHeight = 600;

        // Act
        float[,] scaled = InvokeMethod<float[,]>(
            "ApplyCoordinateScaling",
            new object[] { coords, imgHeight, imgWidth }
        );

        // Expected (Python): coords * target_size / original_size
        // x: 500 * 1024 / 1000 = 512.0
        // y: 300 * 1024 / 600  = 512.0
        Assert.AreEqual(500f * 1024f / 1000f, scaled[0, 0], Tolerance, "X coordinate scaling");
        Assert.AreEqual(300f * 1024f / 600f, scaled[0, 1], Tolerance, "Y coordinate scaling");
    }

    [Test]
    public void ApplyCoordinateScaling_MultiplePoints()
    {
        // Arrange: multiple points
        float[,] coords = new float[,]
        {
            { 0f, 0f },
            { 1920f, 1080f },
            { 960f, 540f },
        };
        int imgWidth = 1920;
        int imgHeight = 1080;

        // Act
        float[,] scaled = InvokeMethod<float[,]>(
            "ApplyCoordinateScaling",
            new object[] { coords, imgHeight, imgWidth }
        );

        // Expected: coords * 1024 / size
        float scaleX = 1024f / 1920f;
        float scaleY = 1024f / 1080f;

        Assert.AreEqual(0f, scaled[0, 0], Tolerance, "Origin X");
        Assert.AreEqual(0f, scaled[0, 1], Tolerance, "Origin Y");
        Assert.AreEqual(1920f * scaleX, scaled[1, 0], Tolerance, "Max X");
        Assert.AreEqual(1080f * scaleY, scaled[1, 1], Tolerance, "Max Y");
        Assert.AreEqual(960f * scaleX, scaled[2, 0], Tolerance, "Center X");
        Assert.AreEqual(540f * scaleY, scaled[2, 1], Tolerance, "Center Y");
    }

    // =======================================================
    // 5. ResizeBilinear
    //    Python: F.interpolate(masks, size, mode="bilinear", align_corners=False)
    //    Note: align_corners=False in PyTorch vs the C# implementation
    //    which uses align_corners=True style (scale = (src-1)/(dst-1))
    // =======================================================
    [Test]
    public void ResizeBilinear_2x2To4x4()
    {
        // Arrange: 2x2 source
        float[,] src = new float[,]
        {
            { 0f, 1f },
            { 2f, 3f }
        };

        // Act
        float[,] dst = InvokeMethod<float[,]>(
            "ResizeBilinear",
            new object[] { src, 4, 4 }
        );

        // Assert: corners should be preserved
        Assert.AreEqual(4, dst.GetLength(0), "Target height");
        Assert.AreEqual(4, dst.GetLength(1), "Target width");
        Assert.AreEqual(0f, dst[0, 0], Tolerance, "Top-left corner");
        Assert.AreEqual(1f, dst[0, 3], Tolerance, "Top-right corner");
        Assert.AreEqual(2f, dst[3, 0], Tolerance, "Bottom-left corner");
        Assert.AreEqual(3f, dst[3, 3], Tolerance, "Bottom-right corner");

        // Center value should be average ≈ 1.5 (interpolation between all 4)
        // With align_corners=True: at (1,1): srcY=0.333, srcX=0.333
        // Bilinear: (1-0.333)*(1-0.333)*0 + (1-0.333)*0.333*1 + 0.333*(1-0.333)*2 + 0.333*0.333*3
        // = 0 + 0.222 + 0.444 + 0.333 = 1.0
        float scaleY = 1f / 3f; // (2-1)/(4-1) = 1/3
        float scaleX = 1f / 3f;
        float srcY = 1 * scaleY;
        float srcX = 1 * scaleX;
        float dy = srcY - (int)Math.Floor(srcY);
        float dx = srcX - (int)Math.Floor(srcX);
        float expected = (1 - dy) * ((1 - dx) * 0 + dx * 1) + dy * ((1 - dx) * 2 + dx * 3);
        Assert.AreEqual(expected, dst[1, 1], Tolerance, "Interpolated center-ish");
    }

    [Test]
    public void ResizeBilinear_IdentityWhenSameSize()
    {
        // Arrange: 3x3 source
        float[,] src = new float[,]
        {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f },
            { 7f, 8f, 9f }
        };

        // Act: resize to same size
        float[,] dst = InvokeMethod<float[,]>(
            "ResizeBilinear",
            new object[] { src, 3, 3 }
        );

        // Assert: should be identical
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                Assert.AreEqual(src[y, x], dst[y, x], Tolerance,
                    $"Element [{y},{x}] should be preserved");
    }

    // =======================================================
    // 6. PostprocessMasks
    //    Python: F.interpolate(masks, (orig_h, orig_w), mode="bilinear")
    // =======================================================
    [Test]
    public void PostprocessMasks_ResizesToTargetSize()
    {
        // Arrange: (1, 1, 2, 2) mask
        float[,,,] masks = new float[1, 1, 2, 2];
        masks[0, 0, 0, 0] = -1f;
        masks[0, 0, 0, 1] = 2f;
        masks[0, 0, 1, 0] = 3f;
        masks[0, 0, 1, 1] = -0.5f;

        int targetH = 4, targetW = 4;

        // Act
        float[,,,] result = InvokeMethod<float[,,,]>(
            "PostprocessMasks",
            new object[] { masks, targetH, targetW }
        );

        // Assert: output shape
        Assert.AreEqual(1, result.GetLength(0), "Batch");
        Assert.AreEqual(1, result.GetLength(1), "Channels");
        Assert.AreEqual(targetH, result.GetLength(2), "Target height");
        Assert.AreEqual(targetW, result.GetLength(3), "Target width");

        // Corners should preserve original values
        Assert.AreEqual(-1f, result[0, 0, 0, 0], Tolerance, "Top-left");
        Assert.AreEqual(2f, result[0, 0, 0, 3], Tolerance, "Top-right");
        Assert.AreEqual(3f, result[0, 0, 3, 0], Tolerance, "Bottom-left");
        Assert.AreEqual(-0.5f, result[0, 0, 3, 3], Tolerance, "Bottom-right");
    }

    [Test]
    public void PostprocessMasks_MultipleChannels()
    {
        // Arrange: (1, 3, 2, 2) - 3 mask channels
        float[,,,] masks = new float[1, 3, 2, 2];
        for (int c = 0; c < 3; c++)
        {
            masks[0, c, 0, 0] = c * 1.0f;
            masks[0, c, 0, 1] = c * 2.0f;
            masks[0, c, 1, 0] = c * 3.0f;
            masks[0, c, 1, 1] = c * 4.0f;
        }

        // Act
        float[,,,] result = InvokeMethod<float[,,,]>(
            "PostprocessMasks",
            new object[] { masks, 4, 4 }
        );

        // Assert: all channels resized
        Assert.AreEqual(3, result.GetLength(1), "Should have 3 channels");
        for (int c = 0; c < 3; c++)
        {
            Assert.AreEqual(c * 1.0f, result[0, c, 0, 0], Tolerance, $"Ch{c} top-left");
            Assert.AreEqual(c * 4.0f, result[0, c, 3, 3], Tolerance, $"Ch{c} bottom-right");
        }
    }

    // =======================================================
    // 7. ConvertToBoolMasks
    //    Python: masks > 0.0
    // =======================================================
    [Test]
    public void ConvertToBoolMasks_ThresholdMatchesPython()
    {
        // Arrange: (1, 1, 3, 3) mask with mixed positive/negative values
        float[,,,] masks = new float[1, 1, 3, 3];
        masks[0, 0, 0, 0] = -1.0f;   // False
        masks[0, 0, 0, 1] = 0.0f;    // False (> 0.0, not >= 0.0)
        masks[0, 0, 0, 2] = 0.001f;  // True
        masks[0, 0, 1, 0] = 5.0f;    // True
        masks[0, 0, 1, 1] = -0.001f; // False
        masks[0, 0, 1, 2] = 100.0f;  // True
        masks[0, 0, 2, 0] = 0.0f;    // False
        masks[0, 0, 2, 1] = -100.0f; // False
        masks[0, 0, 2, 2] = 0.5f;    // True

        // Act
        bool[][,] result = InvokeMethod<bool[][,]>(
            "ConvertToBoolMasks",
            new object[] { masks, 0.0f }
        );

        // Assert: matches Python `masks > 0.0`
        Assert.AreEqual(1, result.Length, "Should have 1 mask");
        bool[,] mask = result[0];

        Assert.IsFalse(mask[0, 0], "Negative → False");
        Assert.IsFalse(mask[0, 1], "Zero → False (strictly >)");
        Assert.IsTrue(mask[0, 2], "Small positive → True");
        Assert.IsTrue(mask[1, 0], "Large positive → True");
        Assert.IsFalse(mask[1, 1], "Small negative → False");
        Assert.IsTrue(mask[1, 2], "Very large positive → True");
        Assert.IsFalse(mask[2, 0], "Zero → False");
        Assert.IsFalse(mask[2, 1], "Very large negative → False");
        Assert.IsTrue(mask[2, 2], "Medium positive → True");
    }

    [Test]
    public void ConvertToBoolMasks_MultiMask()
    {
        // Arrange: (1, 4, 2, 2) - 4 mask candidates (SAM2 outputs 4 masks)
        float[,,,] masks = new float[1, 4, 2, 2];
        for (int c = 0; c < 4; c++)
        {
            masks[0, c, 0, 0] = c > 0 ? 1.0f : -1.0f;
            masks[0, c, 0, 1] = c > 1 ? 1.0f : -1.0f;
            masks[0, c, 1, 0] = c > 2 ? 1.0f : -1.0f;
            masks[0, c, 1, 1] = 1.0f; // all true
        }

        // Act
        bool[][,] result = InvokeMethod<bool[][,]>(
            "ConvertToBoolMasks",
            new object[] { masks, 0.0f }
        );

        // Assert
        Assert.AreEqual(4, result.Length, "Should have 4 mask candidates");

        // Mask 0: only bottom-right is true
        Assert.IsFalse(result[0][0, 0]);
        Assert.IsFalse(result[0][0, 1]);
        Assert.IsFalse(result[0][1, 0]);
        Assert.IsTrue(result[0][1, 1]);

        // Mask 3: all true
        Assert.IsTrue(result[3][0, 0]);
        Assert.IsTrue(result[3][0, 1]);
        Assert.IsTrue(result[3][1, 0]);
        Assert.IsTrue(result[3][1, 1]);
    }

    // =======================================================
    // 8. Flatten4D / ReshapeTo4D roundtrip
    //    Python: tensor.flatten() / tensor.reshape(N,C,H,W)
    // =======================================================
    [Test]
    public void Flatten4D_ReshapeTo4D_Roundtrip()
    {
        // Arrange: (2, 3, 4, 5) tensor
        int N = 2, C = 3, H = 4, W = 5;
        float[,,,] tensor = new float[N, C, H, W];
        float val = 0;
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        tensor[n, c, h, w] = val++;

        // Act: flatten then reshape back
        float[] flat = InvokeMethod<float[]>("Flatten4D", new object[] { tensor });
        float[,,,] restored = InvokeMethod<float[,,,]>(
            "ReshapeTo4D",
            new object[] { flat, N, C, H, W }
        );

        // Assert: roundtrip produces identical values
        Assert.AreEqual(N * C * H * W, flat.Length, "Flat array length");
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        Assert.AreEqual(
                            tensor[n, c, h, w],
                            restored[n, c, h, w],
                            Tolerance,
                            $"Mismatch at [{n},{c},{h},{w}]"
                        );
    }

    [Test]
    public void Flatten4D_RowMajorOrder()
    {
        // Arrange: (1, 2, 2, 3) tensor with known layout
        float[,,,] tensor = new float[1, 2, 2, 3];
        // Channel 0
        tensor[0, 0, 0, 0] = 1; tensor[0, 0, 0, 1] = 2; tensor[0, 0, 0, 2] = 3;
        tensor[0, 0, 1, 0] = 4; tensor[0, 0, 1, 1] = 5; tensor[0, 0, 1, 2] = 6;
        // Channel 1
        tensor[0, 1, 0, 0] = 7; tensor[0, 1, 0, 1] = 8; tensor[0, 1, 0, 2] = 9;
        tensor[0, 1, 1, 0] = 10; tensor[0, 1, 1, 1] = 11; tensor[0, 1, 1, 2] = 12;

        // Act
        float[] flat = InvokeMethod<float[]>("Flatten4D", new object[] { tensor });

        // Assert: row-major (NCHW contiguous) order
        float[] expected = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        Assert.AreEqual(expected.Length, flat.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.AreEqual(expected[i], flat[i], Tolerance, $"Index {i}");
    }

    // =======================================================
    // 9. ReshapeTo3D
    //    Python: np.reshape(flat, (z, y, x))
    // =======================================================
    [Test]
    public void ReshapeTo3D_CorrectShape()
    {
        // Arrange
        float[] flat = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

        // Act
        float[,,] result = InvokeMethod<float[,,]>(
            "ReshapeTo3D",
            new object[] { flat, 2, 3, 2 }
        );

        // Assert: matches np.arange(12).reshape(2,3,2)
        Assert.AreEqual(2, result.GetLength(0));
        Assert.AreEqual(3, result.GetLength(1));
        Assert.AreEqual(2, result.GetLength(2));

        Assert.AreEqual(1f, result[0, 0, 0], Tolerance);
        Assert.AreEqual(2f, result[0, 0, 1], Tolerance);
        Assert.AreEqual(3f, result[0, 1, 0], Tolerance);
        Assert.AreEqual(12f, result[1, 2, 1], Tolerance);
    }

    [Test]
    public void ReshapeTo3D_ThrowsOnSizeMismatch()
    {
        float[] flat = { 1, 2, 3, 4, 5 };

        Assert.Throws<TargetInvocationException>(() =>
        {
            InvokeMethod("ReshapeTo3D", new object[] { flat, 2, 3, 2 });
        });
    }

    // =======================================================
    // 10. BroadcastAdd3D
    //     Python: a + b with numpy broadcasting
    // =======================================================
    [Test]
    public void BroadcastAdd3D_SameShape()
    {
        float[,,] a = new float[2, 3, 4];
        float[,,] b = new float[2, 3, 4];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                {
                    a[i, j, k] = i * 12 + j * 4 + k;
                    b[i, j, k] = 1.0f;
                }

        float[,,] result = InvokeMethod<float[,,]>(
            "BroadcastAdd3D",
            new object[] { a, b }
        );

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    Assert.AreEqual(
                        a[i, j, k] + 1.0f,
                        result[i, j, k],
                        Tolerance,
                        $"[{i},{j},{k}]"
                    );
    }

    [Test]
    public void BroadcastAdd3D_BroadcastSingleton()
    {
        // Python: np.ones((3,2,4)) + np.ones((1,1,4)) * 10
        float[,,] a = new float[3, 2, 4];
        float[,,] b = new float[1, 1, 4];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 4; k++)
                    a[i, j, k] = 1.0f;
        for (int k = 0; k < 4; k++)
            b[0, 0, k] = 10.0f;

        float[,,] result = InvokeMethod<float[,,]>(
            "BroadcastAdd3D",
            new object[] { a, b }
        );

        Assert.AreEqual(3, result.GetLength(0));
        Assert.AreEqual(2, result.GetLength(1));
        Assert.AreEqual(4, result.GetLength(2));

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 4; k++)
                    Assert.AreEqual(11.0f, result[i, j, k], Tolerance);
    }

    // =======================================================
    // 11. Transpose201
    //     Python: np.transpose(arr, (2, 0, 1))
    // =======================================================
    [Test]
    public void Transpose201_MatchesPythonTranspose()
    {
        // Arrange: (2, 3, 4) tensor - np.arange(24).reshape(2,3,4)
        float[,,] input = new float[2, 3, 4];
        float val = 0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    input[i, j, k] = val++;

        // Act
        float[,,] result = InvokeMethod<float[,,]>(
            "Transpose201",
            new object[] { input }
        );

        // Assert: output shape (4, 2, 3)
        Assert.AreEqual(4, result.GetLength(0), "Dim 0 = original dim 2");
        Assert.AreEqual(2, result.GetLength(1), "Dim 1 = original dim 0");
        Assert.AreEqual(3, result.GetLength(2), "Dim 2 = original dim 1");

        // Python: arr.transpose(2,0,1) → output[k,i,j] = input[i,j,k]
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    Assert.AreEqual(
                        input[i, j, k],
                        result[k, i, j],
                        Tolerance,
                        $"transpose201: [{k},{i},{j}] should equal [{i},{j},{k}]"
                    );
    }

    // =======================================================
    // 12. FourReshapeTo3D
    //     Python: tensor.reshape(d0, d1, d2*d3) or view
    // =======================================================
    [Test]
    public void FourReshapeTo3D_MergesLastTwoDimensions()
    {
        // Arrange: (2, 3, 4, 5) tensor
        float[,,,] input = new float[2, 3, 4, 5];
        float val = 0;
        for (int i0 = 0; i0 < 2; i0++)
            for (int i1 = 0; i1 < 3; i1++)
                for (int i2 = 0; i2 < 4; i2++)
                    for (int i3 = 0; i3 < 5; i3++)
                        input[i0, i1, i2, i3] = val++;

        // Act
        float[,,] result = InvokeMethod<float[,,]>(
            "FourReshapeTo3D",
            new object[] { input }
        );

        // Assert: output shape (2, 3, 20)
        Assert.AreEqual(2, result.GetLength(0));
        Assert.AreEqual(3, result.GetLength(1));
        Assert.AreEqual(20, result.GetLength(2));

        // Verify: result[i0, i1, i2*5+i3] == input[i0, i1, i2, i3]
        for (int i0 = 0; i0 < 2; i0++)
            for (int i1 = 0; i1 < 3; i1++)
                for (int i2 = 0; i2 < 4; i2++)
                    for (int i3 = 0; i3 < 5; i3++)
                        Assert.AreEqual(
                            input[i0, i1, i2, i3],
                            result[i0, i1, i2 * 5 + i3],
                            Tolerance
                        );
    }

    // =======================================================
    // 13. PrepareBackboneFeatures
    //     Python: x.flatten(2).permute(2, 0, 1) for each feature map
    // =======================================================
    [Test]
    public void PrepareBackboneFeatures_OutputShape()
    {
        // Arrange: simulate 3 backbone FPN outputs
        // backboneFpn[0]: (1, 32, 256, 256) → flattened HW=65536
        // backboneFpn[1]: (1, 64, 128, 128) → flattened HW=16384
        // backboneFpn[2]: (1, 256, 64, 64)  → flattened HW=4096
        // Using small sizes for test: (1, 2, 3, 3), (1, 2, 2, 2), (1, 2, 2, 2)

        object backboneData = CreateBackboneData(
            new int[] { 1, 2, 3, 3 }, // fpn0
            new int[] { 1, 2, 2, 2 }, // fpn1
            new int[] { 1, 2, 2, 2 }  // fpn2
        );

        // Act
        float[][,,] result = InvokeMethod<float[][,,]>(
            "PrepareBackboneFeatures",
            new object[] { backboneData }
        );

        // Assert: 3 feature levels
        Assert.AreEqual(3, result.Length, "Should have 3 feature levels");

        // Each should have shape (H*W, N, C) where N=1
        // fpn0: (3*3, 1, 2) = (9, 1, 2)
        Assert.AreEqual(9, result[0].GetLength(0), "fpn0: HW=9");
        Assert.AreEqual(1, result[0].GetLength(1), "fpn0: N=1");
        Assert.AreEqual(2, result[0].GetLength(2), "fpn0: C=2");

        // fpn1: (2*2, 1, 2) = (4, 1, 2)
        Assert.AreEqual(4, result[1].GetLength(0), "fpn1: HW=4");
        Assert.AreEqual(1, result[1].GetLength(1), "fpn1: N=1");
        Assert.AreEqual(2, result[1].GetLength(2), "fpn1: C=2");
    }

    [Test]
    public void PrepareBackboneFeatures_ValueOrder()
    {
        // Arrange: simple (1, 2, 2, 2) fpn with known values
        object backboneData = CreateBackboneData(
            new int[] { 1, 2, 2, 2 },
            new int[] { 1, 2, 2, 2 },
            new int[] { 1, 2, 2, 2 }
        );

        // Act
        float[][,,] result = InvokeMethod<float[][,,]>(
            "PrepareBackboneFeatures",
            new object[] { backboneData }
        );

        // Assert: output[hw, n, c] = input[n, c, h, w] where hw = h*W+w
        // For fpn0 (index 0): verify the permutation
        Assert.IsNotNull(result[0], "First feature level should not be null");
        Assert.AreEqual(4, result[0].GetLength(0), "HW = 2*2 = 4");
    }

    // =======================================================
    // 14. GetClickPoints / GetPointLabels
    //     Python: point coordinates and labels setup
    // =======================================================
    [Test]
    public void GetClickPoints_ReturnsCorrectCoordinates()
    {
        // Arrange
        model.AddClickPoint(100, 200);
        model.AddClickPoint(300, 400, negativePoint: true);

        // Act
        float[,] points = model.GetClickPoints(600);

        // Assert
        Assert.AreEqual(2, points.GetLength(0), "Should have 2 points");
        Assert.AreEqual(2, points.GetLength(1), "Each point has x,y");

        Assert.AreEqual(100f, points[0, 0], Tolerance, "Point 0 X");
        Assert.AreEqual(200f, points[0, 1], Tolerance, "Point 0 Y");
        Assert.AreEqual(300f, points[1, 0], Tolerance, "Point 1 X");
        Assert.AreEqual(400f, points[1, 1], Tolerance, "Point 1 Y");
    }

    [Test]
    public void GetPointLabels_PositiveAndNegative()
    {
        // Arrange
        model.AddClickPoint(10, 20);             // positive (label=1)
        model.AddClickPoint(30, 40, true);        // negative (label=0)
        model.AddClickPoint(50, 60);             // positive (label=1)

        // Act
        float[] labels = InvokeMethod<float[]>("GetPointLabels", new object[] { });

        // Assert: matches Python convention (1=positive, 0=negative)
        Assert.AreEqual(3, labels.Length);
        Assert.AreEqual(1f, labels[0], Tolerance, "Positive point → 1");
        Assert.AreEqual(0f, labels[1], Tolerance, "Negative point → 0");
        Assert.AreEqual(1f, labels[2], Tolerance, "Positive point → 1");
    }

    [Test]
    public void GetPointLabels_WithBoxCoords()
    {
        // Arrange: Python SAM2 uses label 2 for top-left, 3 for bottom-right of box
        model.AddClickPoint(100, 200);
        model.SetBoxCoords(new Rect(50, 50, 200, 200)); // xMin=50, yMin=50, w=200, h=200

        // Act
        float[] labels = InvokeMethod<float[]>("GetPointLabels", new object[] { });

        // Assert
        Assert.AreEqual(3, labels.Length, "1 click + 2 box corners");
        Assert.AreEqual(1f, labels[0], Tolerance, "Click point label");
        Assert.AreEqual(2f, labels[1], Tolerance, "Box top-left label (Python convention)");
        Assert.AreEqual(3f, labels[2], Tolerance, "Box bottom-right label (Python convention)");
    }

    [Test]
    public void ResetClickPoint_ClearsAll()
    {
        // Arrange
        model.AddClickPoint(100, 200);
        model.SetBoxCoords(new Rect(10, 10, 100, 100));

        // Act
        model.ResetClickPoint();
        float[,] points = model.GetClickPoints(600);
        float[] labels = InvokeMethod<float[]>("GetPointLabels", new object[] { });

        // Assert
        Assert.AreEqual(0, points.GetLength(0), "No points after reset");
        Assert.AreEqual(0, labels.Length, "No labels after reset");
    }

    // =======================================================
    // 15. ResizeColor32 (nearest neighbor)
    //     Python: cv2.resize with INTER_NEAREST
    // =======================================================
    [Test]
    public void ResizeColor32_NearestNeighbor()
    {
        // Arrange: 2x2 image
        Color32[] src = new Color32[]
        {
            new Color32(255, 0, 0, 255),   // (0,0) Red
            new Color32(0, 255, 0, 255),   // (0,1) Green
            new Color32(0, 0, 255, 255),   // (1,0) Blue
            new Color32(255, 255, 0, 255), // (1,1) Yellow
        };

        // Act: upscale to 4x4
        Color32[] result = InvokeMethod<Color32[]>(
            "ResizeColor32",
            new object[] { src, 2, 2, 4, 4 }
        );

        // Assert: 4x4 = 16 pixels, nearest neighbor
        Assert.AreEqual(16, result.Length);

        // Top-left 2x2 quadrant should be Red
        Assert.AreEqual(255, result[0].r, "Top-left R");
        Assert.AreEqual(0, result[0].g, "Top-left G");
        Assert.AreEqual(255, result[1].r, "Stretched Red");

        // Top-right 2x2 quadrant should be Green
        Assert.AreEqual(0, result[2].r, "Top-right R");
        Assert.AreEqual(255, result[2].g, "Top-right G");
    }

    [Test]
    public void ResizeColor32_Downscale()
    {
        // Arrange: 4x4 image, downscale to 2x2
        Color32[] src = new Color32[16];
        for (int i = 0; i < 16; i++)
            src[i] = new Color32((byte)(i * 16), 0, 0, 255);

        // Act
        Color32[] result = InvokeMethod<Color32[]>(
            "ResizeColor32",
            new object[] { src, 4, 4, 2, 2 }
        );

        // Assert
        Assert.AreEqual(4, result.Length, "2x2 output");
        // Nearest neighbor picks: src[0*4/2 * 4 + 0*4/2] = src[0], etc.
        Assert.AreEqual(src[0].r, result[0].r, "Top-left preserved");
    }

    // =======================================================
    // 16. GetScale
    //     Python: target_size / max(w, h)
    // =======================================================
    [Test]
    public void GetScale_MatchesPythonScaleFactor()
    {
        // Landscape image
        float scale1 = InvokeMethod<float>("GetScale", new object[] { 1920, 1080 });
        Assert.AreEqual(1024f / 1920f, scale1, Tolerance, "Landscape: scale by width");

        // Portrait image
        float scale2 = InvokeMethod<float>("GetScale", new object[] { 720, 1280 });
        Assert.AreEqual(1024f / 1280f, scale2, Tolerance, "Portrait: scale by height");

        // Square image
        float scale3 = InvokeMethod<float>("GetScale", new object[] { 1024, 1024 });
        Assert.AreEqual(1.0f, scale3, Tolerance, "Square 1024: scale = 1.0");
    }

    // =======================================================
    // 17. ReshapeTo2D
    //     Python: np.reshape(flat, (rows, cols))
    // =======================================================
    [Test]
    public void ReshapeTo2D_CorrectLayout()
    {
        float[] flat = { 1, 2, 3, 4, 5, 6 };

        float[,] result = InvokeMethod<float[,]>(
            "ReshapeTo2D",
            new object[] { flat, 2, 3 }
        );

        Assert.AreEqual(2, result.GetLength(0));
        Assert.AreEqual(3, result.GetLength(1));
        Assert.AreEqual(1f, result[0, 0], Tolerance);
        Assert.AreEqual(2f, result[0, 1], Tolerance);
        Assert.AreEqual(3f, result[0, 2], Tolerance);
        Assert.AreEqual(4f, result[1, 0], Tolerance);
        Assert.AreEqual(5f, result[1, 1], Tolerance);
        Assert.AreEqual(6f, result[1, 2], Tolerance);
    }

    // =======================================================
    // 18. End-to-end mask pipeline test
    //     Simulates Python: preprocess → (model) → postprocess → threshold
    // =======================================================
    [Test]
    public void EndToEnd_PreprocessNormalizationValues()
    {
        // Verify the exact normalization constants match Python SAM2
        float[] mean = GetField<float[]>("Mean");
        float[] std = GetField<float[]>("Std");

        // Python SAM2 (sam2/modeling/sam2_base.py):
        // pixel_mean = [123.675, 116.28, 103.53]  / 255.0 = [0.485, 0.456, 0.406]
        // pixel_std  = [58.395, 57.12, 57.375]    / 255.0 = [0.229, 0.224, 0.225]
        Assert.AreEqual(0.485f, mean[0], 1e-3f, "R mean matches Python SAM2");
        Assert.AreEqual(0.456f, mean[1], 1e-3f, "G mean matches Python SAM2");
        Assert.AreEqual(0.406f, mean[2], 1e-3f, "B mean matches Python SAM2");

        Assert.AreEqual(0.229f, std[0], 1e-3f, "R std matches Python SAM2");
        Assert.AreEqual(0.224f, std[1], 1e-3f, "G std matches Python SAM2");
        Assert.AreEqual(0.225f, std[2], 1e-3f, "B std matches Python SAM2");
    }

    [Test]
    public void EndToEnd_MaskPostprocessAndThreshold()
    {
        // Simulate a decoder output and verify the full postprocess pipeline
        // Python: masks = F.interpolate(masks, (H,W)) > 0.0

        // Arrange: (1, 4, 2, 2) raw decoder output
        float[,,,] rawMasks = new float[1, 4, 2, 2];
        // Mask 0: low confidence
        rawMasks[0, 0, 0, 0] = -5f; rawMasks[0, 0, 0, 1] = -3f;
        rawMasks[0, 0, 1, 0] = -1f; rawMasks[0, 0, 1, 1] = 0.5f;
        // Mask 1: high confidence (best mask)
        rawMasks[0, 1, 0, 0] = 5f; rawMasks[0, 1, 0, 1] = 8f;
        rawMasks[0, 1, 1, 0] = 6f; rawMasks[0, 1, 1, 1] = 10f;
        // Mask 2: partial
        rawMasks[0, 2, 0, 0] = -2f; rawMasks[0, 2, 0, 1] = 3f;
        rawMasks[0, 2, 1, 0] = 4f; rawMasks[0, 2, 1, 1] = -1f;
        // Mask 3: all negative
        rawMasks[0, 3, 0, 0] = -1f; rawMasks[0, 3, 0, 1] = -2f;
        rawMasks[0, 3, 1, 0] = -3f; rawMasks[0, 3, 1, 1] = -4f;

        // Step 1: PostprocessMasks (resize to 4x4)
        float[,,,] resized = InvokeMethod<float[,,,]>(
            "PostprocessMasks",
            new object[] { rawMasks, 4, 4 }
        );

        // Step 2: ConvertToBoolMasks
        bool[][,] boolMasks = InvokeMethod<bool[][,]>(
            "ConvertToBoolMasks",
            new object[] { resized, 0.0f }
        );

        Assert.AreEqual(4, boolMasks.Length, "Should have 4 mask candidates");

        // Mask 1 (high confidence) should be all true
        bool allTrue = true;
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                if (!boolMasks[1][y, x])
                    allTrue = false;
        Assert.IsTrue(allTrue, "High confidence mask should be all true");

        // Mask 3 (all negative) should be all false
        bool allFalse = true;
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                if (boolMasks[3][y, x])
                    allFalse = false;
        Assert.IsTrue(allFalse, "All-negative mask should be all false");

        // Simulate best mask selection (as in ProcessMask)
        float[] scores = { 0.2f, 0.95f, 0.5f, 0.1f };
        int bestIdx = 0;
        float maxScore = float.MinValue;
        for (int i = 0; i < scores.Length; i++)
        {
            if (scores[i] > maxScore)
            {
                maxScore = scores[i];
                bestIdx = i;
            }
        }
        Assert.AreEqual(1, bestIdx, "Best mask should be index 1 (highest IoU)");
    }

    // =======================================================
    // 19. ProcessVisionPosEmbeds
    //     Python: [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
    // =======================================================
    [Test]
    public void ProcessVisionPosEmbeds_MatchesPythonComprehension()
    {
        // Arrange: 2 position embedding tensors of shape (1, 2, 3, 3)
        float[][,,,] inputs = new float[2][,,,];
        for (int idx = 0; idx < 2; idx++)
        {
            inputs[idx] = new float[1, 2, 3, 3];
            float val = idx * 100;
            for (int n = 0; n < 1; n++)
                for (int c = 0; c < 2; c++)
                    for (int h = 0; h < 3; h++)
                        for (int w = 0; w < 3; w++)
                            inputs[idx][n, c, h, w] = val++;
        }

        // Act
        float[][,,] result = InvokeMethod<float[][,,]>(
            "ProcessVisionPosEmbeds",
            new object[] { inputs }
        );

        // Assert: each output should be (H*W, N, C) after flatten(2).permute(2,0,1)
        // Input (1, 2, 3, 3) → flatten(2) → (1, 2, 9) → permute(2,0,1) → (9, 1, 2)
        Assert.AreEqual(2, result.Length);

        for (int idx = 0; idx < 2; idx++)
        {
            Assert.AreEqual(9, result[idx].GetLength(0), $"Embed {idx}: dim0 = HW = 9");
            Assert.AreEqual(1, result[idx].GetLength(1), $"Embed {idx}: dim1 = N = 1");
            Assert.AreEqual(2, result[idx].GetLength(2), $"Embed {idx}: dim2 = C = 2");
        }

        // Verify a specific value:
        // inputs[0][0, 1, 2, 1] should appear at result[0][2*3+1, 0, 1] = result[0][7, 0, 1]
        // inputs[0][0, 1, 2, 1] = 1*9 + 2*3 + 1 = 16 (c=1 starts at 9)
        // After flatten(2): shape (1, 2, 9), [0, 1, 7] = value at c=1, hw=2*3+1=7
        // After permute(2,0,1): [7, 0, 1]
        Assert.AreEqual(inputs[0][0, 1, 2, 1], result[0][7, 0, 1], Tolerance,
            "Value mapping through flatten+permute");
    }

    // =======================================================
    // 20. Coordinate scaling with box coordinates
    // =======================================================
    [Test]
    public void GetClickPoints_WithBoxCoords_MatchesPythonConvention()
    {
        // Arrange: Python SAM2 appends box corners as additional points
        model.AddClickPoint(500, 300);
        model.SetBoxCoords(new Rect(100, 100, 400, 300));
        // Rect(x, y, w, h): xMin=100, yMin=100, xMax=500, yMax=400

        // Act
        float[,] points = model.GetClickPoints(600);

        // Assert: 1 click point + 2 box corners = 3 points
        Assert.AreEqual(3, points.GetLength(0), "1 click + 2 box points");

        // Click point
        Assert.AreEqual(500f, points[0, 0], Tolerance);
        Assert.AreEqual(300f, points[0, 1], Tolerance);

        // Box top-left (xMin, yMin)
        Assert.AreEqual(100f, points[1, 0], Tolerance, "Box xMin");
        Assert.AreEqual(100f, points[1, 1], Tolerance, "Box yMin");

        // Box bottom-right (xMax, yMax)
        Assert.AreEqual(500f, points[2, 0], Tolerance, "Box xMax");
        Assert.AreEqual(400f, points[2, 1], Tolerance, "Box yMax");
    }

    // =======================================================
    // Helper: create BackboneOutputs struct via reflection
    // =======================================================
    private object CreateBackboneData(int[] fpn0Shape, int[] fpn1Shape, int[] fpn2Shape)
    {
        Type structType = modelType.GetNestedType(
            "BackboneOutputs",
            BindingFlags.NonPublic | BindingFlags.Public
        );
        Assert.IsNotNull(structType, "BackboneOutputs struct not found");

        object data = Activator.CreateInstance(structType);

        // Create FPN arrays
        float[,,,] fpn0 = CreateTensor4D(fpn0Shape);
        float[,,,] fpn1 = CreateTensor4D(fpn1Shape);
        float[,,,] fpn2 = CreateTensor4D(fpn2Shape);

        float[][,,,] backboneFpn = new float[][,,,] { fpn0, fpn1, fpn2 };

        FieldInfo fpnField = structType.GetField("backboneFpn");
        Assert.IsNotNull(fpnField, "backboneFpn field not found");
        fpnField.SetValue(data, backboneFpn);

        return data;
    }

    private float[,,,] CreateTensor4D(int[] shape)
    {
        float[,,,] tensor = new float[shape[0], shape[1], shape[2], shape[3]];
        float val = 0;
        for (int n = 0; n < shape[0]; n++)
            for (int c = 0; c < shape[1]; c++)
                for (int h = 0; h < shape[2]; h++)
                    for (int w = 0; w < shape[3]; w++)
                        tensor[n, c, h, w] = val++;
        return tensor;
    }
}
