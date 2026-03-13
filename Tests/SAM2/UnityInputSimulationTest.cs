/* SAM2 Unity Input Simulation Tests */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * These tests verify that the SAM2 pipeline correctly handles Unity's
 * bottom-to-top (B2T) pixel format by using the engine's VerticalFlip.
 *
 * Unity specifics:
 *   - GetPixels32() returns pixels in bottom-to-top order (y=0 at bottom-left)
 *   - SetPixels32() expects bottom-to-top order
 *   - Color32 has RGBA byte layout (TextureFormat.RGBA32)
 *
 * After fix, the data flow is:
 *   WebCam/Image -> GetPixels32 (B2T) -> ProcessEmbedding/ProcessMask (B2T)
 *   -> [internal: VerticalFlip to T2B -> SAM2 inference -> VerticalFlipMask to B2T]
 *   -> B2T mask + B2T pixels -> SetPixels32 -> correct display
 */

using NUnit.Framework;
using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using UnityEngine;

[TestFixture]
public class UnityInputSimulationTest
{
    private Sam2InferenceEngine logic = null!;
    private const float Tolerance = 1e-5f;
    private const string PNG_IMAGE_PATH = "/tmp/sam2_test_output/truck.png";

    [SetUp]
    public void SetUp()
    {
        logic = new Sam2InferenceEngine();
    }

    // =======================================================
    // SimulateUnityGetPixels32: flip T2B image to B2T,
    // simulating what Unity's GetPixels32() returns
    // =======================================================
    private Color32[] SimulateUnityGetPixels32(Color32[] top2bottom, int width, int height)
    {
        return Sam2InferenceEngine.VerticalFlip(top2bottom, width, height);
    }

    // =======================================================
    // Load PNG as top-to-bottom Color32 array (standard format)
    // =======================================================
    private (Color32[] pixels, int width, int height) LoadPngTopToBottom(string path)
    {
        using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(path);
        int w = image.Width;
        int h = image.Height;
        var pixels = new Color32[w * h];

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    var p = row[x];
                    pixels[y * w + x] = new Color32(p.R, p.G, p.B, 255);
                }
            }
        });
        return (pixels, w, h);
    }

    // =======================================================
    // 1. Engine VerticalFlip roundtrip: flip twice returns original
    // =======================================================
    [Test]
    public void VerticalFlip_Roundtrip_ReturnsOriginal()
    {
        int width = 4, height = 3;
        Color32[] original = new Color32[width * height];
        for (int i = 0; i < original.Length; i++)
        {
            original[i] = new Color32(
                (byte)(i * 17 % 256),
                (byte)(i * 31 % 256),
                (byte)(i * 53 % 256),
                255
            );
        }

        Color32[] flipped = Sam2InferenceEngine.VerticalFlip(original, width, height);
        Color32[] restored = Sam2InferenceEngine.VerticalFlip(flipped, width, height);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.That(restored[i].r, Is.EqualTo(original[i].r), $"Pixel {i} R");
            Assert.That(restored[i].g, Is.EqualTo(original[i].g), $"Pixel {i} G");
            Assert.That(restored[i].b, Is.EqualTo(original[i].b), $"Pixel {i} B");
        }
    }

    // =======================================================
    // 2. VerticalFlip correctly reverses row order
    // =======================================================
    [Test]
    public void VerticalFlip_ReversesRowOrder()
    {
        int width = 3, height = 3;
        Color32[] input = new Color32[width * height];

        for (int x = 0; x < width; x++)
        {
            input[0 * width + x] = new Color32(255, 0, 0, 255);   // row 0 = red
            input[1 * width + x] = new Color32(0, 255, 0, 255);   // row 1 = green
            input[2 * width + x] = new Color32(0, 0, 255, 255);   // row 2 = blue
        }

        Color32[] flipped = Sam2InferenceEngine.VerticalFlip(input, width, height);

        for (int x = 0; x < width; x++)
        {
            Assert.That(flipped[0 * width + x].b, Is.EqualTo(255), $"Row 0 should be blue after flip (x={x})");
            Assert.That(flipped[0 * width + x].r, Is.EqualTo(0), $"Row 0 should have no red (x={x})");
            Assert.That(flipped[1 * width + x].g, Is.EqualTo(255), $"Row 1 should stay green (x={x})");
            Assert.That(flipped[2 * width + x].r, Is.EqualTo(255), $"Row 2 should be red after flip (x={x})");
            Assert.That(flipped[2 * width + x].b, Is.EqualTo(0), $"Row 2 should have no blue (x={x})");
        }
    }

    // =======================================================
    // 3. VerticalFlipMask roundtrip
    // =======================================================
    [Test]
    public void VerticalFlipMask_Roundtrip_ReturnsOriginal()
    {
        int width = 4, height = 3;
        bool[,] original = new bool[height, width];
        original[0, 0] = true;  original[0, 1] = false; original[0, 2] = true;  original[0, 3] = false;
        original[1, 0] = false; original[1, 1] = true;  original[1, 2] = false; original[1, 3] = true;
        original[2, 0] = true;  original[2, 1] = true;  original[2, 2] = false; original[2, 3] = false;

        bool[,] flipped = Sam2InferenceEngine.VerticalFlipMask(original);
        bool[,] restored = Sam2InferenceEngine.VerticalFlipMask(flipped);

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                Assert.That(restored[y, x], Is.EqualTo(original[y, x]), $"Mask[{y},{x}]");
    }

    // =======================================================
    // 4. VerticalFlipMask reverses row order
    // =======================================================
    [Test]
    public void VerticalFlipMask_ReversesRowOrder()
    {
        // T2B mask: row 0 = all true (top), row 2 = all false (bottom)
        bool[,] t2bMask = new bool[3, 4];
        for (int x = 0; x < 4; x++) t2bMask[0, x] = true;   // top row
        for (int x = 0; x < 4; x++) t2bMask[1, x] = false;  // middle row
        for (int x = 0; x < 4; x++) t2bMask[2, x] = false;  // bottom row

        bool[,] b2tMask = Sam2InferenceEngine.VerticalFlipMask(t2bMask);

        // After flip: row 0 (B2T bottom) should be false, row 2 (B2T top) should be true
        for (int x = 0; x < 4; x++)
        {
            Assert.That(b2tMask[0, x], Is.False, $"B2T row 0 (bottom) should be false (x={x})");
            Assert.That(b2tMask[2, x], Is.True, $"B2T row 2 (top) should be true (x={x})");
        }
    }

    // =======================================================
    // 5. B2T -> VerticalFlip -> PreprocessImage matches
    //    direct T2B -> PreprocessImage
    // =======================================================
    [Test]
    public void UnityPipeline_MatchesDirectTopToBottom_SmallImage()
    {
        int width = 4, height = 3;
        Color32[] top2bottom = new Color32[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                top2bottom[y * width + x] = new Color32(
                    (byte)(y * 80 + x * 20),
                    (byte)(y * 40 + x * 10),
                    (byte)(y * 60 + x * 30),
                    255
                );

        // Direct path: T2B -> PreprocessImage
        float[,,,] directResult = logic.PreprocessImage(top2bottom, width, height, 4);

        // Unity path: T2B -> simulate GetPixels32 (B2T) -> VerticalFlip (T2B) -> PreprocessImage
        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, width, height);
        Color32[] flippedBack = Sam2InferenceEngine.VerticalFlip(bottom2top, width, height);
        float[,,,] unityResult = logic.PreprocessImage(flippedBack, width, height, 4);

        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 4; h++)
                for (int w = 0; w < 4; w++)
                    Assert.That(unityResult[0, c, h, w],
                        Is.EqualTo(directResult[0, c, h, w]).Within(Tolerance),
                        $"Mismatch at [0,{c},{h},{w}]");
    }

    // =======================================================
    // 6. Without VerticalFlip, B2T input produces WRONG results
    // =======================================================
    [Test]
    public void WithoutVerticalFlip_Bottom2Top_ProducesDifferentResult()
    {
        int width = 4, height = 3;
        Color32[] top2bottom = new Color32[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                top2bottom[y * width + x] = new Color32(
                    (byte)(y * 80 + x * 20),
                    (byte)(y * 40 + x * 10),
                    (byte)(y * 60 + x * 30),
                    255
                );

        float[,,,] correctResult = logic.PreprocessImage(top2bottom, width, height, 4);

        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, width, height);
        float[,,,] wrongResult = logic.PreprocessImage(bottom2top, width, height, 4);

        bool foundDifference = false;
        for (int c = 0; c < 3 && !foundDifference; c++)
            for (int h = 0; h < 4 && !foundDifference; h++)
                for (int w = 0; w < 4 && !foundDifference; w++)
                    if (Math.Abs(wrongResult[0, c, h, w] - correctResult[0, c, h, w]) > Tolerance)
                        foundDifference = true;

        Assert.That(foundDifference, Is.True,
            "Feeding B2T pixels without VerticalFlip should produce different preprocessing output");
    }

    // =======================================================
    // 7. Color32 RGBA channel mapping verification
    // =======================================================
    [Test]
    public void Color32ArrayToFloatArray_ChannelMapping_RGBA()
    {
        Color32[] pixels = new Color32[]
        {
            new Color32(200, 100, 50, 255)
        };

        float[,,] result = logic.Color32ArrayToFloatArray(pixels, 1, 1);

        Assert.That(result[0, 0, 0], Is.EqualTo(200f / 255f).Within(Tolerance), "Channel 0 = R");
        Assert.That(result[0, 0, 1], Is.EqualTo(100f / 255f).Within(Tolerance), "Channel 1 = G");
        Assert.That(result[0, 0, 2], Is.EqualTo(50f / 255f).Within(Tolerance), "Channel 2 = B");
    }

    // =======================================================
    // 8. ARGB32 channel misalignment detection
    //    GetRawTextureData<Color32>() on ARGB32 causes wrong channels
    // =======================================================
    [Test]
    public void ChannelMisalignment_ARGB32_ProducesWrongValues()
    {
        Color32 correctPixel = new Color32(200, 100, 50, 255);
        Color32 misalignedPixel = new Color32(255, 200, 100, 50);  // ARGB misread

        Color32[] correctInput = new Color32[] { correctPixel };
        Color32[] misalignedInput = new Color32[] { misalignedPixel };

        float[,,] correctResult = logic.Color32ArrayToFloatArray(correctInput, 1, 1);
        float[,,] misalignedResult = logic.Color32ArrayToFloatArray(misalignedInput, 1, 1);

        Assert.That(Math.Abs(correctResult[0, 0, 0] - misalignedResult[0, 0, 0]) > 0.1f, Is.True,
            "ARGB32 misalignment causes R channel error");
        Assert.That(Math.Abs(correctResult[0, 0, 1] - misalignedResult[0, 0, 1]) > 0.1f, Is.True,
            "ARGB32 misalignment causes G channel error");
        Assert.That(Math.Abs(correctResult[0, 0, 2] - misalignedResult[0, 0, 2]) > 0.1f, Is.True,
            "ARGB32 misalignment causes B channel error");
    }

    // =======================================================
    // 9. PreprocessImage value range with B2T Unity input
    // =======================================================
    [Test]
    public void PreprocessImage_OutputRange_WithUnityInput()
    {
        int width = 8, height = 6;
        Color32[] top2bottom = new Color32[width * height];
        Random rng = new Random(42);
        for (int i = 0; i < top2bottom.Length; i++)
            top2bottom[i] = new Color32(
                (byte)rng.Next(256), (byte)rng.Next(256), (byte)rng.Next(256), 255);

        // Simulate Unity: B2T -> flip -> preprocess
        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, width, height);
        Color32[] flipped = Sam2InferenceEngine.VerticalFlip(bottom2top, width, height);
        float[,,,] result = logic.PreprocessImage(flipped, width, height, 8);

        float min = float.MaxValue, max = float.MinValue;
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 8; h++)
                for (int w = 0; w < 8; w++)
                {
                    float v = result[0, c, h, w];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

        Assert.That(min, Is.GreaterThan(-3.0f), $"Min value {min} too low");
        Assert.That(max, Is.LessThan(3.0f), $"Max value {max} too high");
        Assert.That(min, Is.LessThan(0.0f), "Should have negative values");
        Assert.That(max, Is.GreaterThan(0.0f), "Should have positive values");
    }

    // =======================================================
    // 10. B2T mask overlay: mask and pixels in same order
    //     When both mask and pixels are B2T, mask[y,x] correctly
    //     maps to pixels[y*width+x] for SetPixels32
    // =======================================================
    [Test]
    public void MaskOverlay_B2T_MaskAndPixelsAligned()
    {
        int width = 4, height = 4;

        // B2T image: row 0 (bottom) is blue, row 3 (top) is red
        Color32[] b2tPixels = new Color32[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                if (y < height / 2)
                    b2tPixels[y * width + x] = new Color32(0, 0, 255, 255);  // bottom = blue
                else
                    b2tPixels[y * width + x] = new Color32(255, 0, 0, 255);  // top = red
            }

        // B2T mask: only bottom-left quadrant is true (rows 0-1, cols 0-1)
        bool[,] b2tMask = new bool[height, width];
        for (int y = 0; y < height / 2; y++)
            for (int x = 0; x < width / 2; x++)
                b2tMask[y, x] = true;

        // Apply mask overlay (matching CreateMaskedImage logic)
        Color32[] maskedPixels = (Color32[])b2tPixels.Clone();
        Color32 maskColor = new Color32(255, 0, 0, 255);
        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * width;
            for (int x = 0; x < width; x++)
            {
                int pixelIndex = rowOffset + x;
                if (b2tMask[y, x])
                {
                    Color32 orig = maskedPixels[pixelIndex];
                    maskedPixels[pixelIndex] = new Color32(
                        (byte)(orig.r * 0.6f + maskColor.r * 0.4f),
                        (byte)(orig.g * 0.6f + maskColor.g * 0.4f),
                        (byte)(orig.b * 0.6f + maskColor.b * 0.4f),
                        maskColor.a
                    );
                }
            }
        }

        // Verify: only bottom-left quadrant modified (blue pixels with red overlay)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                if (y < height / 2 && x < width / 2)
                {
                    // Bottom-left: blue pixel with red overlay -> should be modified
                    Assert.That(maskedPixels[idx].r, Is.Not.EqualTo(b2tPixels[idx].r),
                        $"Masked pixel [{y},{x}] should be modified");
                }
                else
                {
                    // Unmasked: unchanged
                    Assert.That(maskedPixels[idx].r, Is.EqualTo(b2tPixels[idx].r),
                        $"Unmasked pixel [{y},{x}] R should be unchanged");
                }
            }

        // Verify SetPixels32 compatibility: B2T data → SetPixels32 → correct display
        // Row 0 (bottom of screen) = blue (possibly masked) ✓
        // Row 3 (top of screen) = red (unmasked) ✓
        Assert.That(maskedPixels[(height - 1) * width].r, Is.EqualTo(255),
            "Top row (red) should be unmasked");
        Assert.That(b2tPixels[0].b, Is.EqualTo(255),
            "Bottom row should be blue (original)");
    }

    // =======================================================
    // 11. T2B mask -> VerticalFlipMask -> B2T mask aligns with B2T pixels
    // =======================================================
    [Test]
    public void FlippedMask_AlignsWithB2TPixels()
    {
        int width = 4, height = 4;

        // SAM2 outputs T2B mask: top half = true (the object is at the top)
        bool[,] t2bMask = new bool[height, width];
        for (int y = 0; y < height / 2; y++)
            for (int x = 0; x < width; x++)
                t2bMask[y, x] = true;

        // Flip to B2T for Unity
        bool[,] b2tMask = Sam2InferenceEngine.VerticalFlipMask(t2bMask);

        // In B2T, the top of the image is at row height-1
        // So the "top half" mask should now be at rows height/2..height-1
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                if (y >= height / 2)
                    Assert.That(b2tMask[y, x], Is.True,
                        $"B2T mask [{y},{x}] should be true (top half of image)");
                else
                    Assert.That(b2tMask[y, x], Is.False,
                        $"B2T mask [{y},{x}] should be false (bottom half of image)");
            }
    }

    // =======================================================
    // 12. Click point coordinate system: T2B coords for SAM2
    // =======================================================
    [Test]
    public void ClickPoint_TopToBottom_CoordinateSystem()
    {
        int imageHeight = 1200;
        int imageWidth = 1800;

        int clickX = 500;
        int clickY = 375;  // T2B coordinate (near top of image)

        logic.AddClickPoint(clickX, clickY);
        float[,] points = logic.GetClickPoints(imageHeight);

        Assert.That(points[0, 0], Is.EqualTo((float)clickX).Within(Tolerance), "X unchanged");
        Assert.That(points[0, 1], Is.EqualTo((float)clickY).Within(Tolerance), "Y unchanged (T2B)");

        float[,] scaled = logic.ApplyCoordinateScaling(points, imageHeight, imageWidth);
        Assert.That(scaled[0, 0], Is.EqualTo(clickX * 1024f / imageWidth).Within(Tolerance), "Scaled X");
        Assert.That(scaled[0, 1], Is.EqualTo(clickY * 1024f / imageHeight).Within(Tolerance), "Scaled Y");
        Assert.That(scaled[0, 0], Is.LessThan(512f), "Click should be in left half");
        Assert.That(scaled[0, 1], Is.LessThan(512f), "Click should be in upper half");
    }

    // =======================================================
    // 13. Screen space (B2T y=0 at bottom) to T2B conversion
    // =======================================================
    [Test]
    public void ScreenToImageSpace_YFlip()
    {
        int imageHeight = 1200;

        int unityScreenY = 0;
        int sam2Y = imageHeight - 1 - unityScreenY;
        Assert.That(sam2Y, Is.EqualTo(1199), "Screen bottom -> T2B bottom");

        unityScreenY = 1199;
        sam2Y = imageHeight - 1 - unityScreenY;
        Assert.That(sam2Y, Is.EqualTo(0), "Screen top -> T2B top");
    }

    // =======================================================
    // 14. E2E: B2T -> VerticalFlip -> PreprocessImage matches
    //     direct T2B -> PreprocessImage (truck.png)
    // =======================================================
    [Test]
    public void EndToEnd_UnityPipeline_MatchesDirectPipeline_TruckPng()
    {
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found at " + PNG_IMAGE_PATH);

        var (top2bottom, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);

        float[,,,] directResult = logic.PreprocessImage(top2bottom, imgW, imgH, 1024);

        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, imgW, imgH);
        Color32[] flippedBack = Sam2InferenceEngine.VerticalFlip(bottom2top, imgW, imgH);
        float[,,,] unityResult = logic.PreprocessImage(flippedBack, imgW, imgH, 1024);

        float maxAbsDiff = 0f;
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 1024; h++)
                for (int w = 0; w < 1024; w++)
                {
                    float diff = Math.Abs(unityResult[0, c, h, w] - directResult[0, c, h, w]);
                    if (diff > maxAbsDiff) maxAbsDiff = diff;
                }

        Console.WriteLine($"Unity vs Direct: max diff = {maxAbsDiff}");
        Assert.That(maxAbsDiff, Is.LessThan(Tolerance),
            $"Unity B2T pipeline should match direct T2B pipeline, max diff = {maxAbsDiff}");
    }

    // =======================================================
    // 15. E2E: Without flip, truck.png preprocessing is wrong
    // =======================================================
    [Test]
    public void EndToEnd_WithoutFlip_TruckPng_ProducesWrongResult()
    {
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found at " + PNG_IMAGE_PATH);

        var (top2bottom, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);

        float[,,,] correctResult = logic.PreprocessImage(top2bottom, imgW, imgH, 1024);

        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, imgW, imgH);
        float[,,,] wrongResult = logic.PreprocessImage(bottom2top, imgW, imgH, 1024);

        float maxAbsDiff = 0f;
        double sumAbsDiff = 0;
        int totalPixels = 3 * 1024 * 1024;
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 1024; h++)
                for (int w = 0; w < 1024; w++)
                {
                    float diff = Math.Abs(wrongResult[0, c, h, w] - correctResult[0, c, h, w]);
                    if (diff > maxAbsDiff) maxAbsDiff = diff;
                    sumAbsDiff += diff;
                }

        double meanAbsDiff = sumAbsDiff / totalPixels;
        Console.WriteLine($"Without flip: max={maxAbsDiff:F6}, mean={meanAbsDiff:F6}");

        Assert.That(maxAbsDiff, Is.GreaterThan(0.1f),
            "Skipping VerticalFlip should cause significant error");
        Assert.That(meanAbsDiff, Is.GreaterThan(0.01),
            "Mean difference should be substantial when flip is skipped");
    }

    // =======================================================
    // 16. E2E full inference: B2T pipeline produces same mask
    //     as direct T2B pipeline (requires ONNX models)
    // =======================================================
    [Test]
    public void EndToEnd_UnityPipeline_FullInference_MatchesDirect()
    {
        const string MODEL_DIR = "/tmp/sam2_models";
        string encoderPath = Path.Combine(MODEL_DIR, "image_encoder_hiera_l.onnx");
        string decoderPath = Path.Combine(MODEL_DIR, "mask_decoder_hiera_l.onnx");
        string promptPath = Path.Combine(MODEL_DIR, "prompt_encoder_hiera_l.onnx");

        if (!File.Exists(encoderPath) || !File.Exists(decoderPath) || !File.Exists(promptPath))
            Assert.Ignore("ONNX models not found in " + MODEL_DIR);
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found at " + PNG_IMAGE_PATH);

        var (top2bottom, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);

        // Unity B2T pipeline: B2T -> internal flip -> T2B -> preprocess
        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, imgW, imgH);
        Color32[] flippedBack = Sam2InferenceEngine.VerticalFlip(bottom2top, imgW, imgH);
        float[,,,] unityPreprocessed = logic.PreprocessImage(flippedBack, imgW, imgH, 1024);
        float[] unityNchw = logic.Flatten4D(unityPreprocessed);

        // Direct T2B pipeline
        float[,,,] directPreprocessed = logic.PreprocessImage(top2bottom, imgW, imgH, 1024);
        float[] directNchw = logic.Flatten4D(directPreprocessed);

        using var backend = new OrtSam2Backend();
        backend.LoadModels(encoderPath, decoderPath, promptPath);

        var unityMask = RunInferencePipeline(backend, unityNchw, imgW, imgH);
        var directMask = RunInferencePipeline(backend, directNchw, imgW, imgH);

        int matchCount = 0, totalPixels = imgW * imgH;
        for (int y = 0; y < imgH; y++)
            for (int x = 0; x < imgW; x++)
                if (unityMask[y, x] == directMask[y, x])
                    matchCount++;

        double matchRate = 100.0 * matchCount / totalPixels;
        Console.WriteLine($"Unity B2T vs Direct T2B mask match rate: {matchRate:F4}%");

        Assert.That(matchRate, Is.EqualTo(100.0).Within(0.01),
            "Unity B2T pipeline should produce identical mask to direct T2B pipeline");
    }

    // =======================================================
    // 17. VerticalFlipMask produces correct B2T mask for SetPixels32
    //     Full pipeline: T2B mask -> flip -> B2T mask -> overlay on B2T pixels
    // =======================================================
    [Test]
    public void FullPipeline_B2TMask_OverlayOnB2TPixels_ForSetPixels32()
    {
        int width = 6, height = 4;

        // Create T2B image (standard): top=red, bottom=blue
        Color32[] t2bPixels = new Color32[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                t2bPixels[y * width + x] = y < height / 2
                    ? new Color32(255, 0, 0, 255)
                    : new Color32(0, 0, 255, 255);

        // Simulate Unity: GetPixels32 returns B2T
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, width, height);

        // Verify B2T: row 0 should be blue (bottom of image), row 3 should be red (top)
        Assert.That(b2tPixels[0].b, Is.EqualTo(255), "B2T row 0 = blue (bottom)");
        Assert.That(b2tPixels[(height - 1) * width].r, Is.EqualTo(255), "B2T last row = red (top)");

        // SAM2 produces T2B mask: top half is the object
        bool[,] t2bMask = new bool[height, width];
        for (int y = 0; y < height / 2; y++)
            for (int x = 0; x < width; x++)
                t2bMask[y, x] = true;

        // Flip mask T2B -> B2T
        bool[,] b2tMask = Sam2InferenceEngine.VerticalFlipMask(t2bMask);

        // Apply B2T mask to B2T pixels (both aligned)
        Color32[] overlayPixels = (Color32[])b2tPixels.Clone();
        Color32 maskColor = new Color32(0, 255, 0, 255);  // green overlay
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                if (b2tMask[y, x])
                {
                    int idx = y * width + x;
                    Color32 orig = overlayPixels[idx];
                    overlayPixels[idx] = new Color32(
                        (byte)(orig.r * 0.6f + maskColor.r * 0.4f),
                        (byte)(orig.g * 0.6f + maskColor.g * 0.4f),
                        (byte)(orig.b * 0.6f + maskColor.b * 0.4f),
                        255
                    );
                }

        // The mask covers the top half of the image. In B2T, top = last rows.
        // So rows height/2..height-1 should be modified (red pixels with green overlay)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                if (y >= height / 2)
                {
                    // Top half in B2T = modified (was red, now red+green overlay)
                    Assert.That(overlayPixels[idx].g, Is.GreaterThan(0),
                        $"Top pixel [{y},{x}] should have green overlay");
                }
                else
                {
                    // Bottom half = unmodified blue
                    Assert.That(overlayPixels[idx].b, Is.EqualTo(b2tPixels[idx].b),
                        $"Bottom pixel [{y},{x}] should be unchanged blue");
                }
            }
    }

    // Helper: run full inference pipeline and return best T2B mask
    private bool[,] RunInferencePipeline(ISam2Backend backend, float[] nchwInput, int imgW, int imgH)
    {
        var encOut = backend.RunEncoder(nchwInput);

        float[,,,] fpn0_4d = logic.ReshapeTo4D(encOut.BackboneFpn0,
            encOut.Fpn0Shape.N, encOut.Fpn0Shape.C, encOut.Fpn0Shape.H, encOut.Fpn0Shape.W);
        float[,,,] fpn1_4d = logic.ReshapeTo4D(encOut.BackboneFpn1,
            encOut.Fpn1Shape.N, encOut.Fpn1Shape.C, encOut.Fpn1Shape.H, encOut.Fpn1Shape.W);
        float[,,,] fpn2_4d = logic.ReshapeTo4D(encOut.BackboneFpn2,
            encOut.Fpn2Shape.N, encOut.Fpn2Shape.C, encOut.Fpn2Shape.H, encOut.Fpn2Shape.W);

        float[][,,,] backboneFpn = new[] { fpn0_4d, fpn1_4d, fpn2_4d };
        float[][,,] visionFeats = logic.PrepareBackboneFeatures(backboneFpn);

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

        float[] imageEmbFlat = logic.Flatten4D(featsArray[^1]);
        float[] highResFeat0 = logic.Flatten4D(featsArray[0]);
        float[] highResFeat1 = logic.Flatten4D(featsArray[1]);

        float[,] rawCoords = new float[,] { { 500, 375 } };
        float[,] scaledCoords = logic.ApplyCoordinateScaling(rawCoords, imgH, imgW);
        float[] flatCoords = { scaledCoords[0, 0], scaledCoords[0, 1] };
        float[] labels = { 1 };
        float[] maskInputDummy = new float[256 * 256];

        var promptOut = backend.RunPromptEncoder(flatCoords, 1, labels, maskInputDummy, 0f);
        var decOut = backend.RunDecoder(
            imageEmbFlat,
            promptOut.DensePe, promptOut.DensePeShape,
            promptOut.SparseEmbeddings, promptOut.SparseShape,
            promptOut.DenseEmbeddings, promptOut.DenseShape,
            highResFeat0, highResFeat1
        );

        float[,,,] masks4d = logic.ReshapeTo4D(decOut.Masks,
            decOut.MasksShape[0], decOut.MasksShape[1],
            decOut.MasksShape[2], decOut.MasksShape[3]);
        float[,,,] resizedMasks = logic.PostprocessMasks(masks4d, imgH, imgW);
        bool[][,] boolMasks = logic.ConvertToBoolMasks(resizedMasks, 0.0f);

        int bestIdx = 0;
        float maxScore = float.MinValue;
        for (int i = 0; i < decOut.IouPred.Length; i++)
            if (decOut.IouPred[i] > maxScore) { maxScore = decOut.IouPred[i]; bestIdx = i; }

        return boolMasks[bestIdx];
    }
}
