/* SAM2 Unity Input Simulation Tests */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * These tests simulate Unity's input format to verify that the SAM2
 * preprocessing pipeline handles it correctly.
 *
 * Unity specifics:
 *   - GetPixels32() returns pixels in bottom-to-top order (y=0 at bottom-left)
 *   - VerticalFlip converts bottom-to-top -> top-to-bottom for SAM2
 *   - SetPixels32() expects bottom-to-top order
 *   - Color32 has RGBA byte layout
 *
 * These tests reproduce the data flow:
 *   WebCam/Image -> GetPixels32 (bottom2top) -> VerticalFlip (top2bottom)
 *   -> PreprocessImage -> SAM2 encoder -> mask decoder
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
    // VerticalFlip: matches Unity's AiliaImageSegmentationSample.cs
    // Converts bottom-to-top (Unity GetPixels32) to top-to-bottom (SAM2)
    // =======================================================
    private Color32[] VerticalFlip(Color32[] inputImage, int width, int height)
    {
        Color32[] outputImage = new Color32[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                outputImage[(height - 1 - y) * width + x] = inputImage[y * width + x];
            }
        }
        return outputImage;
    }

    // =======================================================
    // SimulateUnityBottom2Top: flip top-to-bottom image to
    // bottom-to-top, simulating what GetPixels32() returns
    // =======================================================
    private Color32[] SimulateUnityGetPixels32(Color32[] top2bottom, int width, int height)
    {
        // Unity's GetPixels32 returns bottom-to-top, so flip the image
        return VerticalFlip(top2bottom, width, height);
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
    // 1. VerticalFlip roundtrip: flip twice returns original
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

        Color32[] flipped = VerticalFlip(original, width, height);
        Color32[] restored = VerticalFlip(flipped, width, height);

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

        // Row 0 (y=0): red
        // Row 1 (y=1): green
        // Row 2 (y=2): blue
        for (int x = 0; x < width; x++)
        {
            input[0 * width + x] = new Color32(255, 0, 0, 255);   // row 0 = red
            input[1 * width + x] = new Color32(0, 255, 0, 255);   // row 1 = green
            input[2 * width + x] = new Color32(0, 0, 255, 255);   // row 2 = blue
        }

        Color32[] flipped = VerticalFlip(input, width, height);

        // After flip: row 0 should be blue (was row 2), row 2 should be red (was row 0)
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
    // 3. Unity pipeline (bottom2top -> VerticalFlip -> PreprocessImage)
    //    produces identical output to direct top2bottom -> PreprocessImage
    // =======================================================
    [Test]
    public void UnityPipeline_MatchesDirectTopToBottom_SmallImage()
    {
        int width = 4, height = 3;
        // Create a top-to-bottom image with distinct rows
        Color32[] top2bottom = new Color32[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                top2bottom[y * width + x] = new Color32(
                    (byte)(y * 80 + x * 20),
                    (byte)(y * 40 + x * 10),
                    (byte)(y * 60 + x * 30),
                    255
                );

        // Direct path: top2bottom -> PreprocessImage
        float[,,,] directResult = logic.PreprocessImage(top2bottom, width, height, 4);

        // Unity path: top2bottom -> SimulateGetPixels32 (flip to bottom2top) -> VerticalFlip (back to top2bottom) -> PreprocessImage
        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, width, height);
        Color32[] unityFlipped = VerticalFlip(bottom2top, width, height);
        float[,,,] unityResult = logic.PreprocessImage(unityFlipped, width, height, 4);

        // Both should be identical
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 4; h++)
                for (int w = 0; w < 4; w++)
                    Assert.That(unityResult[0, c, h, w],
                        Is.EqualTo(directResult[0, c, h, w]).Within(Tolerance),
                        $"Mismatch at [0,{c},{h},{w}]");
    }

    // =======================================================
    // 4. Without VerticalFlip, bottom2top input produces WRONG results
    //    (This test proves the flip is necessary)
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

        // Direct correct path
        float[,,,] correctResult = logic.PreprocessImage(top2bottom, width, height, 4);

        // Incorrect path: feed bottom2top directly (skipping VerticalFlip)
        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, width, height);
        float[,,,] wrongResult = logic.PreprocessImage(bottom2top, width, height, 4);

        // They should be different (proving the flip matters)
        bool foundDifference = false;
        for (int c = 0; c < 3 && !foundDifference; c++)
            for (int h = 0; h < 4 && !foundDifference; h++)
                for (int w = 0; w < 4 && !foundDifference; w++)
                    if (Math.Abs(wrongResult[0, c, h, w] - correctResult[0, c, h, w]) > Tolerance)
                        foundDifference = true;

        Assert.That(foundDifference, Is.True,
            "Feeding bottom-to-top pixels without VerticalFlip should produce different preprocessing output");
    }

    // =======================================================
    // 5. Color32 channel mapping: verify R=R, G=G, B=B
    //    (Catches issues like GetRawTextureData on ARGB32 textures
    //     where channels would be misaligned: r=A, g=R, b=G, a=B)
    // =======================================================
    [Test]
    public void Color32ArrayToFloatArray_ChannelMapping_RGBA()
    {
        // Create a pixel with distinct channel values
        Color32[] pixels = new Color32[]
        {
            new Color32(200, 100, 50, 255)
        };

        float[,,] result = logic.Color32ArrayToFloatArray(pixels, 1, 1);

        // Verify channel mapping: R->channel0, G->channel1, B->channel2
        Assert.That(result[0, 0, 0], Is.EqualTo(200f / 255f).Within(Tolerance), "Channel 0 should be R (200/255)");
        Assert.That(result[0, 0, 1], Is.EqualTo(100f / 255f).Within(Tolerance), "Channel 1 should be G (100/255)");
        Assert.That(result[0, 0, 2], Is.EqualTo(50f / 255f).Within(Tolerance), "Channel 2 should be B (50/255)");
    }

    // =======================================================
    // 6. Simulated ARGB32 channel misalignment bug
    //    If GetRawTextureData<Color32>() is used on an ARGB32 texture,
    //    the bytes are [A,R,G,B] but Color32 struct expects [R,G,B,A],
    //    causing: c.r=A, c.g=R, c.b=G, alpha=B
    // =======================================================
    [Test]
    public void ChannelMisalignment_ARGB32_ProducesWrongValues()
    {
        // Original pixel: R=200, G=100, B=50, A=255
        // If read via GetRawTextureData on ARGB32: bytes are [255, 200, 100, 50]
        // Color32 struct interprets as: r=255(was A), g=200(was R), b=100(was G), a=50(was B)
        Color32 correctPixel = new Color32(200, 100, 50, 255);
        Color32 misalignedPixel = new Color32(255, 200, 100, 50);  // ARGB misread

        Color32[] correctInput = new Color32[] { correctPixel };
        Color32[] misalignedInput = new Color32[] { misalignedPixel };

        float[,,] correctResult = logic.Color32ArrayToFloatArray(correctInput, 1, 1);
        float[,,] misalignedResult = logic.Color32ArrayToFloatArray(misalignedInput, 1, 1);

        // Channel 0 (R): correct=200/255, misaligned=255/255
        Assert.That(Math.Abs(correctResult[0, 0, 0] - misalignedResult[0, 0, 0]) > 0.1f, Is.True,
            "ARGB32 misalignment causes R channel error");
        // Channel 1 (G): correct=100/255, misaligned=200/255
        Assert.That(Math.Abs(correctResult[0, 0, 1] - misalignedResult[0, 0, 1]) > 0.1f, Is.True,
            "ARGB32 misalignment causes G channel error");
        // Channel 2 (B): correct=50/255, misaligned=100/255
        Assert.That(Math.Abs(correctResult[0, 0, 2] - misalignedResult[0, 0, 2]) > 0.1f, Is.True,
            "ARGB32 misalignment causes B channel error");
    }

    // =======================================================
    // 7. PreprocessImage value range after normalization
    //    Verify the output range is reasonable for ImageNet normalization
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

        // Simulate Unity pipeline
        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, width, height);
        Color32[] flipped = VerticalFlip(bottom2top, width, height);
        float[,,,] result = logic.PreprocessImage(flipped, width, height, 8);

        // ImageNet normalization: (x - mean) / std
        // For x in [0,1], mean ~0.45, std ~0.22:
        // min = (0 - 0.485) / 0.225 ≈ -2.16
        // max = (1 - 0.406) / 0.225 ≈ 2.64
        float min = float.MaxValue, max = float.MinValue;
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 8; h++)
                for (int w = 0; w < 8; w++)
                {
                    float v = result[0, c, h, w];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

        Assert.That(min, Is.GreaterThan(-3.0f), $"Min value {min} is too low for ImageNet normalization");
        Assert.That(max, Is.LessThan(3.0f), $"Max value {max} is too high for ImageNet normalization");
        Assert.That(min, Is.LessThan(0.0f), "Should have negative values after normalization");
        Assert.That(max, Is.GreaterThan(0.0f), "Should have positive values after normalization");
    }

    // =======================================================
    // 8. CreateMaskedImage pixel mapping verification
    //    In SegmentAnything2Model.CreateMaskedImage, mask[y,x] is applied
    //    to pixels[y*width+x], and the result is passed to SetPixels32.
    //    If pixels are top-to-bottom but SetPixels32 expects bottom-to-top,
    //    the overlay will appear vertically flipped.
    // =======================================================
    [Test]
    public void MaskOverlay_TopToBottom_MaskMapsToCorrectPixels()
    {
        int width = 4, height = 4;

        // Create a top-to-bottom image: top half is red, bottom half is blue
        Color32[] pixels = new Color32[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                if (y < height / 2)
                    pixels[y * width + x] = new Color32(255, 0, 0, 255);  // top = red
                else
                    pixels[y * width + x] = new Color32(0, 0, 255, 255);  // bottom = blue
            }

        // Create a mask that is true only in the top-left quadrant
        bool[,] mask = new bool[height, width];
        for (int y = 0; y < height / 2; y++)
            for (int x = 0; x < width / 2; x++)
                mask[y, x] = true;

        // Simulate CreateMaskedImage logic (without Texture2D/SetPixels32)
        Color32[] maskedPixels = (Color32[])pixels.Clone();
        Color32 maskColor = new Color32(30, 144, 255, 255);  // Dodger blue overlay
        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * width;
            for (int x = 0; x < width; x++)
            {
                int pixelIndex = rowOffset + x;
                if (mask[y, x])
                {
                    Color32 orig = maskedPixels[pixelIndex];
                    // Apply 40% blend (matching CreateMaskedImage logic)
                    maskedPixels[pixelIndex] = new Color32(
                        (byte)(orig.r * 0.6f + maskColor.r * 0.4f),
                        (byte)(orig.g * 0.6f + maskColor.g * 0.4f),
                        (byte)(orig.b * 0.6f + maskColor.b * 0.4f),
                        maskColor.a
                    );
                }
            }
        }

        // Verify: only top-left quadrant pixels should be modified
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                if (y < height / 2 && x < width / 2)
                {
                    // Should be blended (red + dodger blue overlay)
                    Assert.That(maskedPixels[idx].r, Is.Not.EqualTo(pixels[idx].r),
                        $"Masked pixel [{y},{x}] R should be modified");
                }
                else
                {
                    // Should be unchanged
                    Assert.That(maskedPixels[idx].r, Is.EqualTo(pixels[idx].r),
                        $"Unmasked pixel [{y},{x}] R should be unchanged");
                    Assert.That(maskedPixels[idx].g, Is.EqualTo(pixels[idx].g),
                        $"Unmasked pixel [{y},{x}] G should be unchanged");
                    Assert.That(maskedPixels[idx].b, Is.EqualTo(pixels[idx].b),
                        $"Unmasked pixel [{y},{x}] B should be unchanged");
                }
            }
    }

    // =======================================================
    // 9. SetPixels32 Y-flip issue: if top-to-bottom pixels are
    //    passed to SetPixels32 (which expects bottom-to-top),
    //    the resulting texture will be vertically flipped.
    //    This verifies the mask and image stay in sync.
    // =======================================================
    [Test]
    public void SetPixels32_TopToBottom_CausesVerticalFlip()
    {
        // Simulate: mask overlay is done in top-to-bottom order,
        // then pixels are passed to SetPixels32.
        // Unity's SetPixels32 treats y=0 as bottom row.
        // So pixel[0] (our top row) becomes Unity's bottom row -> flipped display.
        int width = 4, height = 4;

        // Top-to-bottom pixels: row0=red(top), row3=blue(bottom)
        Color32[] top2bottom = new Color32[width * height];
        for (int x = 0; x < width; x++)
        {
            top2bottom[0 * width + x] = new Color32(255, 0, 0, 255);  // top row = red
            top2bottom[1 * width + x] = new Color32(0, 255, 0, 255);
            top2bottom[2 * width + x] = new Color32(0, 255, 0, 255);
            top2bottom[3 * width + x] = new Color32(0, 0, 255, 255);  // bottom row = blue
        }

        // What SetPixels32 would display (bottom-to-top reading):
        // Unity row 0 (bottom of screen) = our top2bottom[0] = RED
        // Unity row 3 (top of screen) = our top2bottom[3] = BLUE
        // This means: red appears at bottom, blue at top = FLIPPED

        // The mask is applied in top-to-bottom space to the same pixel array.
        // If mask says "top area is the object", it modifies top2bottom[0..1].
        // After SetPixels32, those modified pixels appear at the BOTTOM of screen.
        // -> Mask overlay appears flipped relative to the displayed image.

        // To verify: check that the data passed to SetPixels32 would need
        // another VerticalFlip to display correctly.
        Color32[] whatUnityDisplaysAsTop = new Color32[width];
        for (int x = 0; x < width; x++)
            whatUnityDisplaysAsTop[x] = top2bottom[(height - 1) * width + x]; // last row

        // Unity displays last element as top -> should be blue (our bottom)
        Assert.That(whatUnityDisplaysAsTop[0].b, Is.EqualTo(255),
            "Without flip correction, Unity would show bottom-row content at top of screen");
        Assert.That(whatUnityDisplaysAsTop[0].r, Is.EqualTo(0),
            "Red (our top row) would appear at bottom instead of top");
    }

    // =======================================================
    // 10. Click point Y-coordinate: verify that click points
    //     in top-to-bottom space are used correctly by SAM2
    // =======================================================
    [Test]
    public void ClickPoint_TopToBottom_CoordinateSystem()
    {
        // In Unity, mouse click Y is in screen space (bottom=0).
        // The sample code uses click points in top-to-bottom image space.
        // SAM2 expects top-to-bottom coordinates.
        int imageHeight = 1200;
        int imageWidth = 1800;

        // Click on truck at roughly (500, 375) in top-to-bottom space
        int clickX = 500;
        int clickY = 375;

        logic.AddClickPoint(clickX, clickY);
        float[,] points = logic.GetClickPoints(imageHeight);

        // Verify the point is passed through without Y-flip
        Assert.That(points[0, 0], Is.EqualTo((float)clickX).Within(Tolerance), "X should be unchanged");
        Assert.That(points[0, 1], Is.EqualTo((float)clickY).Within(Tolerance), "Y should be unchanged (top-to-bottom)");

        // After coordinate scaling to 1024x1024
        float[,] scaled = logic.ApplyCoordinateScaling(points, imageHeight, imageWidth);
        float expectedScaledX = clickX * 1024f / imageWidth;
        float expectedScaledY = clickY * 1024f / imageHeight;

        Assert.That(scaled[0, 0], Is.EqualTo(expectedScaledX).Within(Tolerance), "Scaled X");
        Assert.That(scaled[0, 1], Is.EqualTo(expectedScaledY).Within(Tolerance), "Scaled Y");

        // The scaled point should be in the upper-left area of the 1024x1024 image
        Assert.That(scaled[0, 0], Is.LessThan(512f), "Click X should be in left half");
        Assert.That(scaled[0, 1], Is.LessThan(512f), "Click Y should be in upper half");
    }

    // =======================================================
    // 11. Unity screen space to SAM2 image space conversion
    //     Unity screen: y=0 at bottom
    //     SAM2 image: y=0 at top
    //     Conversion: sam2_y = imageHeight - 1 - unity_screen_y
    // =======================================================
    [Test]
    public void ScreenToImageSpace_YFlip()
    {
        int imageHeight = 1200;

        // Unity screen click at bottom of image (y=0 in screen space)
        int unityScreenY = 0;
        int sam2Y = imageHeight - 1 - unityScreenY; // Convert to top-to-bottom
        Assert.That(sam2Y, Is.EqualTo(1199), "Screen bottom (y=0) -> image bottom (y=1199)");

        // Unity screen click at top of image (y=1199 in screen space)
        unityScreenY = 1199;
        sam2Y = imageHeight - 1 - unityScreenY;
        Assert.That(sam2Y, Is.EqualTo(0), "Screen top (y=1199) -> image top (y=0)");

        // Unity screen click at center
        unityScreenY = 600;
        sam2Y = imageHeight - 1 - unityScreenY;
        Assert.That(sam2Y, Is.EqualTo(599), "Screen center -> near image center");
    }

    // =======================================================
    // 12. E2E test with truck.png: Unity pipeline vs direct pipeline
    //     Loads real image, simulates Unity's GetPixels32 (bottom-to-top),
    //     applies VerticalFlip, preprocesses, and compares with direct path
    // =======================================================
    [Test]
    public void EndToEnd_UnityPipeline_MatchesDirectPipeline_TruckPng()
    {
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found at " + PNG_IMAGE_PATH);

        var (top2bottom, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);

        // Direct path: top-to-bottom -> PreprocessImage
        float[,,,] directResult = logic.PreprocessImage(top2bottom, imgW, imgH, 1024);

        // Unity path: top-to-bottom -> simulate GetPixels32 (bottom-to-top)
        //           -> VerticalFlip (back to top-to-bottom) -> PreprocessImage
        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, imgW, imgH);
        Color32[] unityFlipped = VerticalFlip(bottom2top, imgW, imgH);
        float[,,,] unityResult = logic.PreprocessImage(unityFlipped, imgW, imgH, 1024);

        // Compare all values
        float maxAbsDiff = 0f;
        int diffCount = 0;
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 1024; h++)
                for (int w = 0; w < 1024; w++)
                {
                    float diff = Math.Abs(unityResult[0, c, h, w] - directResult[0, c, h, w]);
                    if (diff > maxAbsDiff) maxAbsDiff = diff;
                    if (diff > Tolerance) diffCount++;
                }

        Console.WriteLine($"Unity vs Direct pipeline comparison:");
        Console.WriteLine($"  Max absolute difference: {maxAbsDiff}");
        Console.WriteLine($"  Pixels with diff > {Tolerance}: {diffCount}/{3 * 1024 * 1024}");

        Assert.That(maxAbsDiff, Is.LessThan(Tolerance),
            $"Unity pipeline should match direct pipeline exactly, max diff = {maxAbsDiff}");
    }

    // =======================================================
    // 13. E2E: Without VerticalFlip, truck.png preprocessing is wrong
    //     This proves that skipping the flip corrupts the input
    // =======================================================
    [Test]
    public void EndToEnd_WithoutFlip_TruckPng_ProducesWrongResult()
    {
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found at " + PNG_IMAGE_PATH);

        var (top2bottom, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);

        // Correct preprocessing
        float[,,,] correctResult = logic.PreprocessImage(top2bottom, imgW, imgH, 1024);

        // Wrong: feed bottom-to-top directly without VerticalFlip
        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, imgW, imgH);
        float[,,,] wrongResult = logic.PreprocessImage(bottom2top, imgW, imgH, 1024);

        // Compute difference
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
        Console.WriteLine($"Without VerticalFlip vs correct:");
        Console.WriteLine($"  Max absolute difference: {maxAbsDiff:F6}");
        Console.WriteLine($"  Mean absolute difference: {meanAbsDiff:F6}");

        // The difference should be significant (not just floating-point noise)
        Assert.That(maxAbsDiff, Is.GreaterThan(0.1f),
            "Skipping VerticalFlip should cause significant preprocessing error");
        Assert.That(meanAbsDiff, Is.GreaterThan(0.01),
            "Mean difference should be substantial when flip is skipped");
    }

    // =======================================================
    // 14. E2E with full inference: Unity pipeline produces same
    //     mask as direct pipeline (requires ONNX models)
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

        // Unity pipeline
        Color32[] bottom2top = SimulateUnityGetPixels32(top2bottom, imgW, imgH);
        Color32[] unityFlipped = VerticalFlip(bottom2top, imgW, imgH);
        float[,,,] unityPreprocessed = logic.PreprocessImage(unityFlipped, imgW, imgH, 1024);
        float[] unityNchw = logic.Flatten4D(unityPreprocessed);

        // Direct pipeline
        float[,,,] directPreprocessed = logic.PreprocessImage(top2bottom, imgW, imgH, 1024);
        float[] directNchw = logic.Flatten4D(directPreprocessed);

        // Run inference with both inputs
        using var backend = new OrtSam2Backend();
        backend.LoadModels(encoderPath, decoderPath, promptPath);

        var unityMask = RunInferencePipeline(backend, unityNchw, imgW, imgH);
        var directMask = RunInferencePipeline(backend, directNchw, imgW, imgH);

        // Compare masks
        int matchCount = 0, totalPixels = imgW * imgH;
        for (int y = 0; y < imgH; y++)
            for (int x = 0; x < imgW; x++)
                if (unityMask[y, x] == directMask[y, x])
                    matchCount++;

        double matchRate = 100.0 * matchCount / totalPixels;
        Console.WriteLine($"Unity vs Direct mask match rate: {matchRate:F4}%");

        Assert.That(matchRate, Is.EqualTo(100.0).Within(0.01),
            "Unity pipeline should produce identical mask to direct pipeline");
    }

    // Helper: run full inference pipeline and return best mask
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
