/* SAM2 Image Mask Unit Tests */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * These tests verify that the C# SAM2 image mask processing logic
 * produces results consistent with the Python SAM2 reference implementation.
 *
 * Python reference:
 *   - sam2/sam2_image_predictor.py  (preprocessing, coordinate transform)
 *   - sam2/modeling/sam2_base.py    (normalization constants, backbone features)
 *   - sam2/utils/transforms.py      (resize, coordinate scaling)
 */

using NUnit.Framework;
using Sam2MaskTests;
using System;

[TestFixture]
public class SegmentAnything2MaskTest
{
    private Sam2ImageMaskLogic logic = null!;
    private const float Tolerance = 1e-5f;

    [SetUp]
    public void SetUp()
    {
        logic = new Sam2ImageMaskLogic();
    }

    // =======================================================
    // 1. Color32ArrayToFloatArray
    //    Python: img.astype(np.float32) / 255.0
    // =======================================================
    [Test]
    public void Color32ArrayToFloatArray_MatchesPythonDivide255()
    {
        Color32[] pixels = new Color32[]
        {
            new Color32(0, 0, 0, 255),
            new Color32(255, 255, 255, 255),
            new Color32(128, 64, 192, 255),
            new Color32(100, 200, 50, 255),
        };

        float[,,] result = logic.Color32ArrayToFloatArray(pixels, 2, 2);

        Assert.That(result.GetLength(2), Is.EqualTo(3), "3 channels (RGB)");

        // Black -> 0.0
        Assert.That(result[0, 0, 0], Is.EqualTo(0.0f).Within(Tolerance));
        Assert.That(result[0, 0, 1], Is.EqualTo(0.0f).Within(Tolerance));
        Assert.That(result[0, 0, 2], Is.EqualTo(0.0f).Within(Tolerance));

        // White -> 1.0
        Assert.That(result[0, 1, 0], Is.EqualTo(1.0f).Within(Tolerance));
        Assert.That(result[0, 1, 1], Is.EqualTo(1.0f).Within(Tolerance));
        Assert.That(result[0, 1, 2], Is.EqualTo(1.0f).Within(Tolerance));

        // (128,64,192) -> (0.50196, 0.25098, 0.75294)
        Assert.That(result[1, 0, 0], Is.EqualTo(128f / 255f).Within(Tolerance));
        Assert.That(result[1, 0, 1], Is.EqualTo(64f / 255f).Within(Tolerance));
        Assert.That(result[1, 0, 2], Is.EqualTo(192f / 255f).Within(Tolerance));

        // (100,200,50)
        Assert.That(result[1, 1, 0], Is.EqualTo(100f / 255f).Within(Tolerance));
        Assert.That(result[1, 1, 1], Is.EqualTo(200f / 255f).Within(Tolerance));
        Assert.That(result[1, 1, 2], Is.EqualTo(50f / 255f).Within(Tolerance));
    }

    // =======================================================
    // 2. ImageNet Normalization
    //    Python: (pixel/255.0 - mean) / std
    // =======================================================
    [Test]
    public void PreprocessImage_NormalizationMatchesPython()
    {
        Color32[] pixel = new Color32[] { new Color32(128, 64, 192, 255) };

        float[,,,] result = logic.PreprocessImage(pixel, 1, 1, 1);

        float r_norm = (128f / 255f - 0.485f) / 0.229f;
        float g_norm = (64f / 255f - 0.456f) / 0.224f;
        float b_norm = (192f / 255f - 0.406f) / 0.225f;

        // NCHW format: (1, 3, 1, 1)
        Assert.That(result.GetLength(0), Is.EqualTo(1), "Batch=1");
        Assert.That(result.GetLength(1), Is.EqualTo(3), "Channels=3");
        Assert.That(result.GetLength(2), Is.EqualTo(1), "H=1");
        Assert.That(result.GetLength(3), Is.EqualTo(1), "W=1");

        Assert.That(result[0, 0, 0, 0], Is.EqualTo(r_norm).Within(1e-4f), "R normalization");
        Assert.That(result[0, 1, 0, 0], Is.EqualTo(g_norm).Within(1e-4f), "G normalization");
        Assert.That(result[0, 2, 0, 0], Is.EqualTo(b_norm).Within(1e-4f), "B normalization");
    }

    [Test]
    public void PreprocessImage_OutputFormatIsNCHW()
    {
        Color32[] pixels = new Color32[6];
        for (int i = 0; i < 6; i++)
            pixels[i] = new Color32((byte)(i * 40), (byte)(i * 30), (byte)(i * 20), 255);

        float[,,,] result = logic.PreprocessImage(pixels, 3, 2, 2);

        Assert.That(result.GetLength(0), Is.EqualTo(1), "N=1");
        Assert.That(result.GetLength(1), Is.EqualTo(3), "C=3");
        Assert.That(result.GetLength(2), Is.EqualTo(2), "H=imageSize");
        Assert.That(result.GetLength(3), Is.EqualTo(2), "W=imageSize");
    }

    [Test]
    public void PreprocessImage_BlackImage_AllChannelsNegative()
    {
        // Python: (0.0 - mean) / std -> all negative
        Color32[] pixel = new Color32[] { new Color32(0, 0, 0, 255) };
        float[,,,] result = logic.PreprocessImage(pixel, 1, 1, 1);

        Assert.That(result[0, 0, 0, 0], Is.LessThan(0), "Black R negative after norm");
        Assert.That(result[0, 1, 0, 0], Is.LessThan(0), "Black G negative after norm");
        Assert.That(result[0, 2, 0, 0], Is.LessThan(0), "Black B negative after norm");

        Assert.That(result[0, 0, 0, 0], Is.EqualTo(-0.485f / 0.229f).Within(1e-4f));
        Assert.That(result[0, 1, 0, 0], Is.EqualTo(-0.456f / 0.224f).Within(1e-4f));
        Assert.That(result[0, 2, 0, 0], Is.EqualTo(-0.406f / 0.225f).Within(1e-4f));
    }

    [Test]
    public void PreprocessImage_WhiteImage_AllChannelsPositive()
    {
        Color32[] pixel = new Color32[] { new Color32(255, 255, 255, 255) };
        float[,,,] result = logic.PreprocessImage(pixel, 1, 1, 1);

        Assert.That(result[0, 0, 0, 0], Is.GreaterThan(0), "White R positive");
        Assert.That(result[0, 1, 0, 0], Is.GreaterThan(0), "White G positive");
        Assert.That(result[0, 2, 0, 0], Is.GreaterThan(0), "White B positive");

        Assert.That(result[0, 0, 0, 0], Is.EqualTo((1.0f - 0.485f) / 0.229f).Within(1e-4f));
        Assert.That(result[0, 1, 0, 0], Is.EqualTo((1.0f - 0.456f) / 0.224f).Within(1e-4f));
        Assert.That(result[0, 2, 0, 0], Is.EqualTo((1.0f - 0.406f) / 0.225f).Within(1e-4f));
    }

    // =======================================================
    // 3. TransposeHWCtoCHW
    //    Python: img.transpose(2, 0, 1)
    // =======================================================
    [Test]
    public void TransposeHWCtoCHW_MatchesPythonTranspose()
    {
        float[,,] hwc = new float[2, 3, 3];
        for (int h = 0; h < 2; h++)
            for (int w = 0; w < 3; w++)
                for (int c = 0; c < 3; c++)
                    hwc[h, w, c] = h * 100 + w * 10 + c;

        float[,,] chw = logic.TransposeHWCtoCHW(hwc);

        Assert.That(chw.GetLength(0), Is.EqualTo(3), "C first");
        Assert.That(chw.GetLength(1), Is.EqualTo(2), "H second");
        Assert.That(chw.GetLength(2), Is.EqualTo(3), "W third");

        for (int h = 0; h < 2; h++)
            for (int w = 0; w < 3; w++)
                for (int c = 0; c < 3; c++)
                    Assert.That(chw[c, h, w], Is.EqualTo(hwc[h, w, c]).Within(Tolerance),
                        $"chw[{c},{h},{w}] == hwc[{h},{w},{c}]");
    }

    // =======================================================
    // 4. ApplyCoordinateScaling
    //    Python: coords * target_size / original_size
    // =======================================================
    [Test]
    public void ApplyCoordinateScaling_MatchesPythonScaling()
    {
        float[,] coords = new float[,] { { 500f, 300f } };

        float[,] scaled = logic.ApplyCoordinateScaling(coords, 600, 1000);

        Assert.That(scaled[0, 0], Is.EqualTo(500f * 1024f / 1000f).Within(Tolerance), "X scaling");
        Assert.That(scaled[0, 1], Is.EqualTo(300f * 1024f / 600f).Within(Tolerance), "Y scaling");
    }

    [Test]
    public void ApplyCoordinateScaling_MultiplePoints()
    {
        float[,] coords = new float[,]
        {
            { 0f, 0f },
            { 1920f, 1080f },
            { 960f, 540f },
        };

        float[,] scaled = logic.ApplyCoordinateScaling(coords, 1080, 1920);

        float scaleX = 1024f / 1920f;
        float scaleY = 1024f / 1080f;

        Assert.That(scaled[0, 0], Is.EqualTo(0f).Within(Tolerance), "Origin X");
        Assert.That(scaled[0, 1], Is.EqualTo(0f).Within(Tolerance), "Origin Y");
        Assert.That(scaled[1, 0], Is.EqualTo(1920f * scaleX).Within(Tolerance), "Max X");
        Assert.That(scaled[1, 1], Is.EqualTo(1080f * scaleY).Within(Tolerance), "Max Y");
        Assert.That(scaled[2, 0], Is.EqualTo(960f * scaleX).Within(Tolerance), "Center X");
        Assert.That(scaled[2, 1], Is.EqualTo(540f * scaleY).Within(Tolerance), "Center Y");
    }

    [Test]
    public void ApplyCoordinateScaling_SquareImage_NoDistortion()
    {
        float[,] coords = new float[,] { { 512f, 512f } };
        float[,] scaled = logic.ApplyCoordinateScaling(coords, 1024, 1024);

        Assert.That(scaled[0, 0], Is.EqualTo(512f).Within(Tolerance));
        Assert.That(scaled[0, 1], Is.EqualTo(512f).Within(Tolerance));
    }

    // =======================================================
    // 5. ResizeBilinear
    // =======================================================
    [Test]
    public void ResizeBilinear_2x2To4x4_CornersPreserved()
    {
        float[,] src = new float[,] { { 0f, 1f }, { 2f, 3f } };

        float[,] dst = logic.ResizeBilinear(src, 4, 4);

        Assert.That(dst.GetLength(0), Is.EqualTo(4));
        Assert.That(dst.GetLength(1), Is.EqualTo(4));
        // align_corners=False: top-left maps to clamped (-0.25,-0.25) -> (0,0)
        Assert.That(dst[0, 0], Is.EqualTo(0f).Within(Tolerance), "Top-left");
        Assert.That(dst[0, 3], Is.EqualTo(1f).Within(Tolerance), "Top-right");
        Assert.That(dst[3, 0], Is.EqualTo(2f).Within(Tolerance), "Bottom-left");
        Assert.That(dst[3, 3], Is.EqualTo(3f).Within(Tolerance), "Bottom-right");
    }

    [Test]
    public void ResizeBilinear_2x2To4x4_InterpolatedValues()
    {
        float[,] src = new float[,] { { 0f, 1f }, { 2f, 3f } };

        float[,] dst = logic.ResizeBilinear(src, 4, 4);

        // align_corners=False: srcPos = (dstPos + 0.5) * (srcSize/dstSize) - 0.5
        // (1,1): srcY = (1+0.5)*(2/4)-0.5 = 0.25, srcX = 0.25
        float dy = 0.25f;
        float dx = 0.25f;
        float expected = (1 - dy) * ((1 - dx) * 0 + dx * 1) + dy * ((1 - dx) * 2 + dx * 3);
        Assert.That(dst[1, 1], Is.EqualTo(expected).Within(Tolerance), "Interpolated at (1,1)");

        // (2,2): srcY = (2+0.5)*(2/4)-0.5 = 0.75, srcX = 0.75
        dy = 0.75f;
        dx = 0.75f;
        expected = (1 - dy) * ((1 - dx) * 0 + dx * 1) + dy * ((1 - dx) * 2 + dx * 3);
        Assert.That(dst[2, 2], Is.EqualTo(expected).Within(Tolerance), "Interpolated at (2,2)");
    }

    [Test]
    public void ResizeBilinear_IdentityWhenSameSize()
    {
        float[,] src = new float[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };

        float[,] dst = logic.ResizeBilinear(src, 3, 3);

        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                Assert.That(dst[y, x], Is.EqualTo(src[y, x]).Within(Tolerance),
                    $"[{y},{x}] preserved");
    }

    [Test]
    public void ResizeBilinear_Downscale4x4To2x2()
    {
        float[,] src = new float[,]
        {
            { 0, 1, 2, 3 },
            { 4, 5, 6, 7 },
            { 8, 9, 10, 11 },
            { 12, 13, 14, 15 }
        };

        float[,] dst = logic.ResizeBilinear(src, 2, 2);

        // align_corners=False: src = (dst+0.5)*(4/2)-0.5
        // (0,0) -> src(0.5, 0.5) = interpolated center of 4 top-left pixels
        // (0,1) -> src(0.5, 2.5), (1,0) -> src(2.5, 0.5), (1,1) -> src(2.5, 2.5)
        Assert.That(dst[0, 0], Is.EqualTo(2.5f).Within(Tolerance), "[0,0] = avg of top-left 2x2");
        Assert.That(dst[0, 1], Is.EqualTo(4.5f).Within(Tolerance), "[0,1] = avg of top-right 2x2");
        Assert.That(dst[1, 0], Is.EqualTo(10.5f).Within(Tolerance), "[1,0] = avg of bottom-left 2x2");
        Assert.That(dst[1, 1], Is.EqualTo(12.5f).Within(Tolerance), "[1,1] = avg of bottom-right 2x2");
    }

    // =======================================================
    // 6. PostprocessMasks
    // =======================================================
    [Test]
    public void PostprocessMasks_ResizesToTargetSize()
    {
        float[,,,] masks = new float[1, 1, 2, 2];
        masks[0, 0, 0, 0] = -1f;
        masks[0, 0, 0, 1] = 2f;
        masks[0, 0, 1, 0] = 3f;
        masks[0, 0, 1, 1] = -0.5f;

        float[,,,] result = logic.PostprocessMasks(masks, 4, 4);

        Assert.That(result.GetLength(0), Is.EqualTo(1), "Batch");
        Assert.That(result.GetLength(1), Is.EqualTo(1), "Channels");
        Assert.That(result.GetLength(2), Is.EqualTo(4), "Target H");
        Assert.That(result.GetLength(3), Is.EqualTo(4), "Target W");

        Assert.That(result[0, 0, 0, 0], Is.EqualTo(-1f).Within(Tolerance), "Top-left");
        Assert.That(result[0, 0, 0, 3], Is.EqualTo(2f).Within(Tolerance), "Top-right");
        Assert.That(result[0, 0, 3, 0], Is.EqualTo(3f).Within(Tolerance), "Bottom-left");
        Assert.That(result[0, 0, 3, 3], Is.EqualTo(-0.5f).Within(Tolerance), "Bottom-right");
    }

    [Test]
    public void PostprocessMasks_MultipleChannels()
    {
        float[,,,] masks = new float[1, 3, 2, 2];
        for (int c = 0; c < 3; c++)
        {
            masks[0, c, 0, 0] = c * 1.0f;
            masks[0, c, 0, 1] = c * 2.0f;
            masks[0, c, 1, 0] = c * 3.0f;
            masks[0, c, 1, 1] = c * 4.0f;
        }

        float[,,,] result = logic.PostprocessMasks(masks, 4, 4);

        Assert.That(result.GetLength(1), Is.EqualTo(3), "3 channels");
        for (int c = 0; c < 3; c++)
        {
            Assert.That(result[0, c, 0, 0], Is.EqualTo(c * 1.0f).Within(Tolerance), $"Ch{c} TL");
            Assert.That(result[0, c, 3, 3], Is.EqualTo(c * 4.0f).Within(Tolerance), $"Ch{c} BR");
        }
    }

    // =======================================================
    // 7. ConvertToBoolMasks
    //    Python: masks > 0.0
    // =======================================================
    [Test]
    public void ConvertToBoolMasks_ThresholdMatchesPython()
    {
        float[,,,] masks = new float[1, 1, 3, 3];
        masks[0, 0, 0, 0] = -1.0f;
        masks[0, 0, 0, 1] = 0.0f;
        masks[0, 0, 0, 2] = 0.001f;
        masks[0, 0, 1, 0] = 5.0f;
        masks[0, 0, 1, 1] = -0.001f;
        masks[0, 0, 1, 2] = 100.0f;
        masks[0, 0, 2, 0] = 0.0f;
        masks[0, 0, 2, 1] = -100.0f;
        masks[0, 0, 2, 2] = 0.5f;

        bool[][,] result = logic.ConvertToBoolMasks(masks, 0.0f);

        Assert.That(result.Length, Is.EqualTo(1));
        bool[,] mask = result[0];

        Assert.That(mask[0, 0], Is.False, "Negative -> False");
        Assert.That(mask[0, 1], Is.False, "Zero -> False (strictly >)");
        Assert.That(mask[0, 2], Is.True, "Small positive -> True");
        Assert.That(mask[1, 0], Is.True, "Large positive -> True");
        Assert.That(mask[1, 1], Is.False, "Small negative -> False");
        Assert.That(mask[1, 2], Is.True, "Very large positive -> True");
        Assert.That(mask[2, 0], Is.False, "Zero -> False");
        Assert.That(mask[2, 1], Is.False, "Very large negative -> False");
        Assert.That(mask[2, 2], Is.True, "Medium positive -> True");
    }

    [Test]
    public void ConvertToBoolMasks_FourMaskCandidates()
    {
        // SAM2 outputs 4 mask candidates
        float[,,,] masks = new float[1, 4, 2, 2];
        for (int c = 0; c < 4; c++)
        {
            masks[0, c, 0, 0] = c > 0 ? 1f : -1f;
            masks[0, c, 0, 1] = c > 1 ? 1f : -1f;
            masks[0, c, 1, 0] = c > 2 ? 1f : -1f;
            masks[0, c, 1, 1] = 1.0f;
        }

        bool[][,] result = logic.ConvertToBoolMasks(masks, 0.0f);

        Assert.That(result.Length, Is.EqualTo(4), "4 mask candidates");

        Assert.That(result[0][0, 0], Is.False);
        Assert.That(result[0][1, 1], Is.True);

        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
                Assert.That(result[3][y, x], Is.True, $"Mask3[{y},{x}]");
    }

    // =======================================================
    // 8. Flatten4D / ReshapeTo4D roundtrip
    // =======================================================
    [Test]
    public void Flatten4D_ReshapeTo4D_Roundtrip()
    {
        int N = 2, C = 3, H = 4, W = 5;
        float[,,,] tensor = new float[N, C, H, W];
        float val = 0;
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        tensor[n, c, h, w] = val++;

        float[] flat = logic.Flatten4D(tensor);
        float[,,,] restored = logic.ReshapeTo4D(flat, N, C, H, W);

        Assert.That(flat.Length, Is.EqualTo(N * C * H * W));
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        Assert.That(restored[n, c, h, w], Is.EqualTo(tensor[n, c, h, w]).Within(Tolerance),
                            $"[{n},{c},{h},{w}]");
    }

    [Test]
    public void Flatten4D_RowMajorOrder()
    {
        float[,,,] tensor = new float[1, 2, 2, 3];
        tensor[0, 0, 0, 0] = 1; tensor[0, 0, 0, 1] = 2; tensor[0, 0, 0, 2] = 3;
        tensor[0, 0, 1, 0] = 4; tensor[0, 0, 1, 1] = 5; tensor[0, 0, 1, 2] = 6;
        tensor[0, 1, 0, 0] = 7; tensor[0, 1, 0, 1] = 8; tensor[0, 1, 0, 2] = 9;
        tensor[0, 1, 1, 0] = 10; tensor[0, 1, 1, 1] = 11; tensor[0, 1, 1, 2] = 12;

        float[] flat = logic.Flatten4D(tensor);

        float[] expected = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        Assert.That(flat.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(flat[i], Is.EqualTo(expected[i]).Within(Tolerance), $"Index {i}");
    }

    // =======================================================
    // 9. ReshapeTo3D
    // =======================================================
    [Test]
    public void ReshapeTo3D_CorrectShape()
    {
        float[] flat = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

        float[,,] result = logic.ReshapeTo3D(flat, 2, 3, 2);

        Assert.That(result.GetLength(0), Is.EqualTo(2));
        Assert.That(result.GetLength(1), Is.EqualTo(3));
        Assert.That(result.GetLength(2), Is.EqualTo(2));
        Assert.That(result[0, 0, 0], Is.EqualTo(1f).Within(Tolerance));
        Assert.That(result[0, 0, 1], Is.EqualTo(2f).Within(Tolerance));
        Assert.That(result[0, 1, 0], Is.EqualTo(3f).Within(Tolerance));
        Assert.That(result[1, 2, 1], Is.EqualTo(12f).Within(Tolerance));
    }

    [Test]
    public void ReshapeTo3D_ThrowsOnSizeMismatch()
    {
        float[] flat = { 1, 2, 3, 4, 5 };
        Assert.Throws<ArgumentException>(() => logic.ReshapeTo3D(flat, 2, 3, 2));
    }

    // =======================================================
    // 10. ReshapeTo2D
    // =======================================================
    [Test]
    public void ReshapeTo2D_CorrectLayout()
    {
        float[] flat = { 1, 2, 3, 4, 5, 6 };
        float[,] result = logic.ReshapeTo2D(flat, 2, 3);

        Assert.That(result[0, 0], Is.EqualTo(1f).Within(Tolerance));
        Assert.That(result[0, 2], Is.EqualTo(3f).Within(Tolerance));
        Assert.That(result[1, 0], Is.EqualTo(4f).Within(Tolerance));
        Assert.That(result[1, 2], Is.EqualTo(6f).Within(Tolerance));
    }

    [Test]
    public void ReshapeTo4D_ThrowsOnSizeMismatch()
    {
        float[] flat = { 1, 2, 3 };
        Assert.Throws<ArgumentException>(() => logic.ReshapeTo4D(flat, 2, 2, 2, 2));
    }

    // =======================================================
    // 11. BroadcastAdd3D
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

        float[,,] result = logic.BroadcastAdd3D(a, b);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    Assert.That(result[i, j, k], Is.EqualTo(a[i, j, k] + 1.0f).Within(Tolerance));
    }

    [Test]
    public void BroadcastAdd3D_BroadcastSingleton()
    {
        float[,,] a = new float[3, 2, 4];
        float[,,] b = new float[1, 1, 4];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 4; k++)
                    a[i, j, k] = 1.0f;
        for (int k = 0; k < 4; k++)
            b[0, 0, k] = 10.0f;

        float[,,] result = logic.BroadcastAdd3D(a, b);

        Assert.That(result.GetLength(0), Is.EqualTo(3));
        Assert.That(result.GetLength(1), Is.EqualTo(2));
        Assert.That(result.GetLength(2), Is.EqualTo(4));

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 4; k++)
                    Assert.That(result[i, j, k], Is.EqualTo(11.0f).Within(Tolerance));
    }

    [Test]
    public void BroadcastAdd3D_ThrowsOnIncompatibleShapes()
    {
        float[,,] a = new float[3, 2, 4];
        float[,,] b = new float[2, 2, 4];
        Assert.Throws<ArgumentException>(() => logic.BroadcastAdd3D(a, b));
    }

    // =======================================================
    // 12. Transpose201
    //     Python: np.transpose(arr, (2, 0, 1))
    // =======================================================
    [Test]
    public void Transpose201_MatchesPythonTranspose()
    {
        float[,,] input = new float[2, 3, 4];
        float val = 0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    input[i, j, k] = val++;

        float[,,] result = logic.Transpose201(input);

        Assert.That(result.GetLength(0), Is.EqualTo(4), "Dim 0 = original dim 2");
        Assert.That(result.GetLength(1), Is.EqualTo(2), "Dim 1 = original dim 0");
        Assert.That(result.GetLength(2), Is.EqualTo(3), "Dim 2 = original dim 1");

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    Assert.That(result[k, i, j], Is.EqualTo(input[i, j, k]).Within(Tolerance),
                        $"[{k},{i},{j}] == [{i},{j},{k}]");
    }

    // =======================================================
    // 13. FourReshapeTo3D
    //     Python: tensor.reshape(d0, d1, d2*d3)
    // =======================================================
    [Test]
    public void FourReshapeTo3D_MergesLastTwoDims()
    {
        float[,,,] input = new float[2, 3, 4, 5];
        float val = 0;
        for (int i0 = 0; i0 < 2; i0++)
            for (int i1 = 0; i1 < 3; i1++)
                for (int i2 = 0; i2 < 4; i2++)
                    for (int i3 = 0; i3 < 5; i3++)
                        input[i0, i1, i2, i3] = val++;

        float[,,] result = logic.FourReshapeTo3D(input);

        Assert.That(result.GetLength(0), Is.EqualTo(2));
        Assert.That(result.GetLength(1), Is.EqualTo(3));
        Assert.That(result.GetLength(2), Is.EqualTo(20));

        for (int i0 = 0; i0 < 2; i0++)
            for (int i1 = 0; i1 < 3; i1++)
                for (int i2 = 0; i2 < 4; i2++)
                    for (int i3 = 0; i3 < 5; i3++)
                        Assert.That(result[i0, i1, i2 * 5 + i3],
                            Is.EqualTo(input[i0, i1, i2, i3]).Within(Tolerance));
    }

    // =======================================================
    // 14. ProcessVisionPosEmbeds
    //     Python: [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
    // =======================================================
    [Test]
    public void ProcessVisionPosEmbeds_MatchesPythonComprehension()
    {
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

        float[][,,] result = logic.ProcessVisionPosEmbeds(inputs);

        Assert.That(result.Length, Is.EqualTo(2));

        // (1, 2, 3, 3) -> flatten(2) -> (1, 2, 9) -> permute(2,0,1) -> (9, 1, 2)
        for (int idx = 0; idx < 2; idx++)
        {
            Assert.That(result[idx].GetLength(0), Is.EqualTo(9), $"Embed {idx}: HW=9");
            Assert.That(result[idx].GetLength(1), Is.EqualTo(1), $"Embed {idx}: N=1");
            Assert.That(result[idx].GetLength(2), Is.EqualTo(2), $"Embed {idx}: C=2");
        }

        // inputs[0][0, 1, 2, 1] -> flatten: [0, 1, 7] -> permute: [7, 0, 1]
        Assert.That(result[0][7, 0, 1], Is.EqualTo(inputs[0][0, 1, 2, 1]).Within(Tolerance),
            "Value mapping through flatten+permute");
    }

    // =======================================================
    // 15. PrepareBackboneFeatures
    // =======================================================
    [Test]
    public void PrepareBackboneFeatures_OutputShape()
    {
        float[][,,,] fpn = new float[3][,,,];
        fpn[0] = CreateTensor4D(1, 2, 3, 3);
        fpn[1] = CreateTensor4D(1, 2, 2, 2);
        fpn[2] = CreateTensor4D(1, 2, 2, 2);

        float[][,,] result = logic.PrepareBackboneFeatures(fpn);

        Assert.That(result.Length, Is.EqualTo(3), "3 feature levels");

        Assert.That(result[0].GetLength(0), Is.EqualTo(9), "fpn0: HW=9");
        Assert.That(result[0].GetLength(1), Is.EqualTo(1), "fpn0: N=1");
        Assert.That(result[0].GetLength(2), Is.EqualTo(2), "fpn0: C=2");

        Assert.That(result[1].GetLength(0), Is.EqualTo(4), "fpn1: HW=4");
    }

    [Test]
    public void PrepareBackboneFeatures_ValueMapping()
    {
        float[,,,] fpn = new float[1, 2, 2, 2];
        fpn[0, 0, 0, 0] = 1; fpn[0, 0, 0, 1] = 2;
        fpn[0, 0, 1, 0] = 3; fpn[0, 0, 1, 1] = 4;
        fpn[0, 1, 0, 0] = 5; fpn[0, 1, 0, 1] = 6;
        fpn[0, 1, 1, 0] = 7; fpn[0, 1, 1, 1] = 8;

        float[][,,,] fpnArray = new float[3][,,,] { fpn, fpn, fpn };
        float[][,,] result = logic.PrepareBackboneFeatures(fpnArray);

        // output[hw, n, c] = input[n, c, h, w] where hw = h*W + w
        Assert.That(result[0][0, 0, 0], Is.EqualTo(1f).Within(Tolerance), "[0,0,0]=1");
        Assert.That(result[0][0, 0, 1], Is.EqualTo(5f).Within(Tolerance), "[0,0,1]=5");
        Assert.That(result[0][1, 0, 0], Is.EqualTo(2f).Within(Tolerance), "[1,0,0]=2");
        Assert.That(result[0][3, 0, 1], Is.EqualTo(8f).Within(Tolerance), "[3,0,1]=8");
    }

    // =======================================================
    // 16. Click points / labels
    // =======================================================
    [Test]
    public void GetClickPoints_ReturnsCorrectCoordinates()
    {
        logic.AddClickPoint(100, 200);
        logic.AddClickPoint(300, 400, negativePoint: true);

        float[,] points = logic.GetClickPoints(600);

        Assert.That(points.GetLength(0), Is.EqualTo(2));
        Assert.That(points[0, 0], Is.EqualTo(100f).Within(Tolerance));
        Assert.That(points[0, 1], Is.EqualTo(200f).Within(Tolerance));
        Assert.That(points[1, 0], Is.EqualTo(300f).Within(Tolerance));
        Assert.That(points[1, 1], Is.EqualTo(400f).Within(Tolerance));
    }

    [Test]
    public void GetPointLabels_PositiveAndNegative()
    {
        logic.AddClickPoint(10, 20);
        logic.AddClickPoint(30, 40, true);
        logic.AddClickPoint(50, 60);

        float[] labels = logic.GetPointLabels();

        Assert.That(labels.Length, Is.EqualTo(3));
        Assert.That(labels[0], Is.EqualTo(1f).Within(Tolerance), "Positive -> 1");
        Assert.That(labels[1], Is.EqualTo(0f).Within(Tolerance), "Negative -> 0");
        Assert.That(labels[2], Is.EqualTo(1f).Within(Tolerance), "Positive -> 1");
    }

    [Test]
    public void GetPointLabels_WithBoxCoords()
    {
        logic.AddClickPoint(100, 200);
        logic.SetBoxCoords(new Rect(50, 50, 200, 200));

        float[] labels = logic.GetPointLabels();

        Assert.That(labels.Length, Is.EqualTo(3), "1 click + 2 box corners");
        Assert.That(labels[0], Is.EqualTo(1f).Within(Tolerance), "Click label");
        Assert.That(labels[1], Is.EqualTo(2f).Within(Tolerance), "Box TL label");
        Assert.That(labels[2], Is.EqualTo(3f).Within(Tolerance), "Box BR label");
    }

    [Test]
    public void GetClickPoints_WithBoxCoords_CoordinateValues()
    {
        logic.AddClickPoint(500, 300);
        logic.SetBoxCoords(new Rect(100, 100, 400, 300));

        float[,] points = logic.GetClickPoints(600);

        Assert.That(points.GetLength(0), Is.EqualTo(3));
        Assert.That(points[1, 0], Is.EqualTo(100f).Within(Tolerance), "xMin");
        Assert.That(points[1, 1], Is.EqualTo(100f).Within(Tolerance), "yMin");
        Assert.That(points[2, 0], Is.EqualTo(500f).Within(Tolerance), "xMax");
        Assert.That(points[2, 1], Is.EqualTo(400f).Within(Tolerance), "yMax");
    }

    [Test]
    public void ResetClickPoint_ClearsAll()
    {
        logic.AddClickPoint(100, 200);
        logic.SetBoxCoords(new Rect(10, 10, 100, 100));

        logic.ResetClickPoint();
        float[,] points = logic.GetClickPoints(600);
        float[] labels = logic.GetPointLabels();

        Assert.That(points.GetLength(0), Is.EqualTo(0), "No points");
        Assert.That(labels.Length, Is.EqualTo(0), "No labels");
    }

    // =======================================================
    // 17. ResizeBilinearHWC (bilinear interpolation for preprocessing)
    // =======================================================
    [Test]
    public void ResizeBilinearHWC_Upscale()
    {
        // 2x2 float image -> 4x4 (align_corners=False)
        float[,,] src = new float[2, 2, 3];
        src[0, 0, 0] = 1.0f; src[0, 0, 1] = 0.0f; src[0, 0, 2] = 0.0f; // red
        src[0, 1, 0] = 0.0f; src[0, 1, 1] = 1.0f; src[0, 1, 2] = 0.0f; // green
        src[1, 0, 0] = 0.0f; src[1, 0, 1] = 0.0f; src[1, 0, 2] = 1.0f; // blue
        src[1, 1, 0] = 1.0f; src[1, 1, 1] = 1.0f; src[1, 1, 2] = 0.0f; // yellow

        float[,,] result = logic.ResizeBilinearHWC(src, 2, 2, 4, 4);

        Assert.That(result.GetLength(0), Is.EqualTo(4));
        Assert.That(result.GetLength(1), Is.EqualTo(4));
        Assert.That(result.GetLength(2), Is.EqualTo(3));

        // With align_corners=False, pixel centers are offset.
        // Top-left pixel (0,0) maps to src (-0.25,-0.25) clamped to (0,0) -> pure red
        Assert.That(result[0, 0, 0], Is.EqualTo(1.0f).Within(Tolerance), "Top-left R");
        // All output pixels should have valid (non-negative) values
        for (int h = 0; h < 4; h++)
            for (int w = 0; w < 4; w++)
                for (int c = 0; c < 3; c++)
                    Assert.That(result[h, w, c], Is.GreaterThanOrEqualTo(0.0f), $"[{h},{w},{c}] >= 0");
    }

    [Test]
    public void ResizeBilinearHWC_SameSize()
    {
        float[,,] src = new float[2, 2, 3];
        src[0, 0, 0] = 0.1f; src[0, 0, 1] = 0.2f; src[0, 0, 2] = 0.3f;
        src[0, 1, 0] = 0.4f; src[0, 1, 1] = 0.5f; src[0, 1, 2] = 0.6f;
        src[1, 0, 0] = 0.7f; src[1, 0, 1] = 0.8f; src[1, 0, 2] = 0.9f;
        src[1, 1, 0] = 1.0f; src[1, 1, 1] = 0.0f; src[1, 1, 2] = 0.5f;

        float[,,] result = logic.ResizeBilinearHWC(src, 2, 2, 2, 2);

        Assert.That(result.GetLength(0), Is.EqualTo(2));
        for (int h = 0; h < 2; h++)
            for (int w = 0; w < 2; w++)
                for (int c = 0; c < 3; c++)
                    Assert.That(result[h, w, c], Is.EqualTo(src[h, w, c]).Within(Tolerance), $"[{h},{w},{c}]");
    }

    // =======================================================
    // 18. GetScale
    // =======================================================
    [Test]
    public void GetScale_MatchesPythonScaleFactor()
    {
        Assert.That(logic.GetScale(1920, 1080), Is.EqualTo(1024f / 1920f).Within(Tolerance), "Landscape");
        Assert.That(logic.GetScale(720, 1280), Is.EqualTo(1024f / 1280f).Within(Tolerance), "Portrait");
        Assert.That(logic.GetScale(1024, 1024), Is.EqualTo(1.0f).Within(Tolerance), "Square");
    }

    // =======================================================
    // 19. End-to-end: normalization constants
    // =======================================================
    [Test]
    public void EndToEnd_NormalizationConstants_MatchPythonSAM2()
    {
        // Python SAM2: pixel_mean=[123.675, 116.28, 103.53] / 255
        //              pixel_std=[58.395, 57.12, 57.375] / 255
        Assert.That(logic.Mean[0], Is.EqualTo(123.675f / 255f).Within(1e-3f), "R mean");
        Assert.That(logic.Mean[1], Is.EqualTo(116.28f / 255f).Within(1e-3f), "G mean");
        Assert.That(logic.Mean[2], Is.EqualTo(103.53f / 255f).Within(1e-3f), "B mean");

        Assert.That(logic.Std[0], Is.EqualTo(58.395f / 255f).Within(1e-3f), "R std");
        Assert.That(logic.Std[1], Is.EqualTo(57.12f / 255f).Within(1e-3f), "G std");
        Assert.That(logic.Std[2], Is.EqualTo(57.375f / 255f).Within(1e-3f), "B std");
    }

    // =======================================================
    // 20. End-to-end: full mask pipeline
    // =======================================================
    [Test]
    public void EndToEnd_MaskPostprocessAndThreshold()
    {
        float[,,,] rawMasks = new float[1, 4, 2, 2];
        rawMasks[0, 0, 0, 0] = -5f; rawMasks[0, 0, 0, 1] = -3f;
        rawMasks[0, 0, 1, 0] = -1f; rawMasks[0, 0, 1, 1] = 0.5f;
        rawMasks[0, 1, 0, 0] = 5f; rawMasks[0, 1, 0, 1] = 8f;
        rawMasks[0, 1, 1, 0] = 6f; rawMasks[0, 1, 1, 1] = 10f;
        rawMasks[0, 2, 0, 0] = -2f; rawMasks[0, 2, 0, 1] = 3f;
        rawMasks[0, 2, 1, 0] = 4f; rawMasks[0, 2, 1, 1] = -1f;
        rawMasks[0, 3, 0, 0] = -1f; rawMasks[0, 3, 0, 1] = -2f;
        rawMasks[0, 3, 1, 0] = -3f; rawMasks[0, 3, 1, 1] = -4f;

        float[,,,] resized = logic.PostprocessMasks(rawMasks, 4, 4);
        bool[][,] boolMasks = logic.ConvertToBoolMasks(resized, 0.0f);

        Assert.That(boolMasks.Length, Is.EqualTo(4), "4 mask candidates");

        bool allTrue = true;
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                if (!boolMasks[1][y, x]) allTrue = false;
        Assert.That(allTrue, Is.True, "High confidence -> all true");

        bool allFalse = true;
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                if (boolMasks[3][y, x]) allFalse = false;
        Assert.That(allFalse, Is.True, "All negative -> all false");

        float[] scores = { 0.2f, 0.95f, 0.5f, 0.1f };
        int bestIdx = 0;
        float maxScore = float.MinValue;
        for (int i = 0; i < scores.Length; i++)
            if (scores[i] > maxScore) { maxScore = scores[i]; bestIdx = i; }
        Assert.That(bestIdx, Is.EqualTo(1), "Best mask = index 1");
    }

    // =======================================================
    // 21. Coordinate + preprocessing pipeline
    // =======================================================
    [Test]
    public void EndToEnd_CoordinateScalingPipeline()
    {
        logic.AddClickPoint(500, 300);
        logic.AddClickPoint(200, 100, negativePoint: true);

        float[,] coords = logic.GetClickPoints(600);
        float[] labels = logic.GetPointLabels();
        float[,] scaled = logic.ApplyCoordinateScaling(coords, 600, 1000);

        Assert.That(labels[0], Is.EqualTo(1f).Within(Tolerance), "Positive");
        Assert.That(labels[1], Is.EqualTo(0f).Within(Tolerance), "Negative");

        Assert.That(scaled[0, 0], Is.EqualTo(500f * 1024f / 1000f).Within(Tolerance));
        Assert.That(scaled[0, 1], Is.EqualTo(300f * 1024f / 600f).Within(Tolerance));
        Assert.That(scaled[1, 0], Is.EqualTo(200f * 1024f / 1000f).Within(Tolerance));
        Assert.That(scaled[1, 1], Is.EqualTo(100f * 1024f / 600f).Within(Tolerance));
    }

    // =======================================================
    // Helper
    // =======================================================
    private float[,,,] CreateTensor4D(int n, int c, int h, int w)
    {
        float[,,,] tensor = new float[n, c, h, w];
        float val = 0;
        for (int in_ = 0; in_ < n; in_++)
            for (int ic = 0; ic < c; ic++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < w; iw++)
                        tensor[in_, ic, ih, iw] = val++;
        return tensor;
    }
}
