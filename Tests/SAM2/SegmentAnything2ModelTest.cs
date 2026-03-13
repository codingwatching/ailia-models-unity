/* SAM2 Sam2Processor Tests */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * Tests Sam2Processor directly — the class that owns all SAM2 inference logic.
 * Sam2Processor is used by SegmentAnything2Model (Unity wrapper) and can be
 * tested standalone with any ISam2Backend (ORT or ailia).
 *
 * Covers:
 *   - Click point / box management
 *   - ProcessEmbedding (B2T -> T2B flip + encode)
 *   - ProcessMask (inference -> best mask -> T2B -> B2T flip)
 *   - ApplyMaskOverlay (B2T mask on B2T pixels)
 *   - ConvertClickCoordsToB2T
 *   - DrawClickPoints
 *   - Multi-point and box prompt flows
 *   - Python reference comparison
 */

using NUnit.Framework;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.IO;
using System.Linq;
using UnityEngine;

[TestFixture]
public class SegmentAnything2ModelTest
{
    private const string MODEL_DIR = "/tmp/sam2_models";
    private const string PNG_IMAGE_PATH = "/tmp/sam2_test_output/truck.png";
    private const string PYTHON_OUTPUT_DIR = "/tmp/sam2_test_output";
    private const string CSHARP_OUTPUT_DIR = "/tmp/sam2_csharp_output";

    private const int IMG_W = 1800;
    private const int IMG_H = 1200;

    [SetUp]
    public void SetUp()
    {
        Directory.CreateDirectory(CSHARP_OUTPUT_DIR);
    }

    private string EncoderPath => Path.Combine(MODEL_DIR, "image_encoder_hiera_l.onnx");
    private string DecoderPath => Path.Combine(MODEL_DIR, "mask_decoder_hiera_l.onnx");
    private string PromptPath => Path.Combine(MODEL_DIR, "prompt_encoder_hiera_l.onnx");

    private bool ModelsExist() =>
        File.Exists(EncoderPath) && File.Exists(DecoderPath) && File.Exists(PromptPath);

    // =======================================================
    // Image loading helpers
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

    private Color32[] SimulateUnityGetPixels32(Color32[] top2bottom, int width, int height)
    {
        return Sam2InferenceEngine.VerticalFlip(top2bottom, width, height);
    }

    // =======================================================
    // Mask comparison / save helpers
    // =======================================================
    private bool[,] LoadNpyBool(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        int headerLen = BitConverter.ToUInt16(raw, 8);
        string header = System.Text.Encoding.ASCII.GetString(raw, 10, headerLen);
        int shapeStart = header.IndexOf("'shape': (") + 10;
        int shapeEnd = header.IndexOf(")", shapeStart);
        var dims = header.Substring(shapeStart, shapeEnd - shapeStart)
            .Split(',').Select(s => int.Parse(s.Trim())).ToArray();
        int h = dims[0], w = dims[1];
        int dataOffset = 10 + headerLen;
        bool[,] result = new bool[h, w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                result[y, x] = raw[dataOffset + y * w + x] != 0;
        return result;
    }

    private void SaveMaskPng(bool[,] mask, string path)
    {
        int h = mask.GetLength(0), w = mask.GetLength(1);
        using var img = new Image<L8>(w, h);
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                    row[x] = new L8(mask[y, x] ? (byte)255 : (byte)0);
            }
        });
        img.Save(path);
    }

    private void SaveDiffPng(bool[,] csMask, bool[,] pyMask, string path)
    {
        int h = csMask.GetLength(0), w = csMask.GetLength(1);
        using var img = new Image<Rgb24>(w, h);
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    bool cs = csMask[y, x], py = pyMask[y, x];
                    if (cs && py) row[x] = new Rgb24(255, 255, 255);
                    else if (cs) row[x] = new Rgb24(255, 0, 0);
                    else if (py) row[x] = new Rgb24(0, 0, 255);
                    else row[x] = new Rgb24(0, 0, 0);
                }
            }
        });
        img.Save(path);
    }

    private (double matchRate, int diffCount, int onlyCsharp, int onlyPython)
        CompareMasks(bool[,] csMask, bool[,] pyMask)
    {
        int h = csMask.GetLength(0), w = csMask.GetLength(1);
        int matchCount = 0, diffCount = 0, onlyCsharp = 0, onlyPython = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (csMask[y, x] == pyMask[y, x]) matchCount++;
                else { diffCount++; if (csMask[y, x]) onlyCsharp++; else onlyPython++; }
            }
        return (100.0 * matchCount / (h * w), diffCount, onlyCsharp, onlyPython);
    }

    // =======================================================
    // Create a Sam2Processor with ORT backend
    // =======================================================
    private (Sam2Processor processor, OrtSam2Backend backend) CreateOrtProcessor()
    {
        var backend = new OrtSam2Backend();
        backend.LoadModels(EncoderPath, DecoderPath, PromptPath);
        var processor = new Sam2Processor(backend);
        return (processor, backend);
    }

    // =======================================================
    // 1. Click point management
    // =======================================================
    [Test]
    public void ClickPointManagement()
    {
        // Use a dummy backend (not needed for click point tests)
        // We can't create a processor without a backend, so use ORT if available
        // But click points are pure engine logic — test engine directly
        var engine = new Sam2InferenceEngine();
        // Sam2Processor delegates to engine, but we test via processor to verify delegation

        if (!ModelsExist())
            Assert.Ignore("Models needed to construct processor");

        var (processor, backend) = CreateOrtProcessor();
        using var _ = backend;

        processor.AddClickPoint(500, 375);
        float[,] coords = processor.GetClickPoints(IMG_H);
        float[] labels = processor.GetPointLabels();

        Assert.That(coords.GetLength(0), Is.EqualTo(1));
        Assert.That(coords[0, 0], Is.EqualTo(500f));
        Assert.That(coords[0, 1], Is.EqualTo(375f));
        Assert.That(labels[0], Is.EqualTo(1f));

        // Negative point
        processor.AddClickPoint(100, 200, negativePoint: true);
        coords = processor.GetClickPoints(IMG_H);
        labels = processor.GetPointLabels();

        Assert.That(coords.GetLength(0), Is.EqualTo(2));
        Assert.That(labels[1], Is.EqualTo(0f));

        // Reset
        processor.ResetClickPoint();
        coords = processor.GetClickPoints(IMG_H);
        Assert.That(coords.GetLength(0), Is.EqualTo(0));
    }

    // =======================================================
    // 2. Box coordinates
    // =======================================================
    [Test]
    public void BoxCoords()
    {
        if (!ModelsExist())
            Assert.Ignore("Models needed to construct processor");

        var (processor, backend) = CreateOrtProcessor();
        using var _ = backend;

        processor.AddClickPoint(100, 100);
        processor.SetBoxCoords(new Rect(50, 50, 200, 150));

        float[,] coords = processor.GetClickPoints(IMG_H);
        float[] labels = processor.GetPointLabels();

        Assert.That(coords.GetLength(0), Is.EqualTo(3));
        Assert.That(labels[1], Is.EqualTo(2f));
        Assert.That(labels[2], Is.EqualTo(3f));
        Assert.That(coords[1, 0], Is.EqualTo(50f));
        Assert.That(coords[2, 0], Is.EqualTo(250f));
    }

    // =======================================================
    // 3. ProcessEmbedding + EmbeddingExist
    // =======================================================
    [Test]
    public void ProcessEmbedding_CreatesEmbedding()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (processor, backend) = CreateOrtProcessor();
        using var _ = backend;

        Assert.That(processor.EmbeddingExist(), Is.False);

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);

        processor.ProcessEmbedding(b2tPixels, imgW, imgH);

        Assert.That(processor.EmbeddingExist(), Is.True);
    }

    // =======================================================
    // 4. ProcessMask returns no mask without click points
    // =======================================================
    [Test]
    public void ProcessMask_NoClickPoints_ReturnsNoMask()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (processor, backend) = CreateOrtProcessor();
        using var _ = backend;

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);
        processor.ProcessEmbedding(b2tPixels, imgW, imgH);

        var result = processor.ProcessMask(imgW, imgH);
        Assert.That(result.HasMask, Is.False);
    }

    // =======================================================
    // 5. Full pipeline: ProcessEmbedding + ProcessMask
    //    Compare with Python reference
    // =======================================================
    [Test]
    public void FullPipeline_MatchesPython()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (processor, backend) = CreateOrtProcessor();
        using var _ = backend;

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);

        // Step 1: Embedding (B2T input, like Unity)
        processor.ProcessEmbedding(b2tPixels, imgW, imgH);

        // Step 2: Add click point (T2B coordinates)
        processor.AddClickPoint(500, 375);

        // Step 3: Mask
        var result = processor.ProcessMask(imgW, imgH);

        Assert.That(result.HasMask, Is.True);
        Assert.That(result.B2TMask.GetLength(0), Is.EqualTo(imgH));
        Assert.That(result.B2TMask.GetLength(1), Is.EqualTo(imgW));
        Assert.That(result.BestScore, Is.GreaterThan(0));

        Console.WriteLine($"Best mask: index={result.BestMaskIndex}, score={result.BestScore:F4}");

        int trueCount = 0;
        for (int y = 0; y < imgH; y++)
            for (int x = 0; x < imgW; x++)
                if (result.B2TMask[y, x]) trueCount++;
        Console.WriteLine($"B2T mask true pixels: {trueCount}/{imgW * imgH} ({100.0 * trueCount / (imgW * imgH):F1}%)");

        // The B2T mask is flipped from T2B. To compare with Python (T2B), flip back.
        bool[,] t2bMask = Sam2InferenceEngine.VerticalFlipMask(result.B2TMask);

        SaveMaskPng(t2bMask, Path.Combine(CSHARP_OUTPUT_DIR, "mask_processor_t2b.png"));

        string pyMaskPath = Path.Combine(PYTHON_OUTPUT_DIR, "best_mask_bool_png.npy");
        if (File.Exists(pyMaskPath))
        {
            bool[,] pyMask = LoadNpyBool(pyMaskPath);
            var (matchRate, diffCount, onlyCs, onlyPy) = CompareMasks(t2bMask, pyMask);
            Console.WriteLine($"Python comparison: match={matchRate:F4}%, diff={diffCount}");

            SaveDiffPng(t2bMask, pyMask, Path.Combine(CSHARP_OUTPUT_DIR, "diff_processor_vs_python.png"));

            Assert.That(matchRate, Is.GreaterThan(99.0),
                $"Match rate should be > 99%, got {matchRate:F4}%");
        }
        else
        {
            Console.WriteLine("Python reference mask not found, skipping comparison.");
        }
    }

    // =======================================================
    // 6. ApplyMaskOverlay: correct pixel modification
    // =======================================================
    [Test]
    public void ApplyMaskOverlay_ModifiesCorrectPixels()
    {
        int w = 4, h = 4;
        Color32[] pixels = new Color32[w * h];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = new Color32(0, 0, 255, 255);  // all blue

        bool[,] mask = new bool[h, w];
        // Mask: only top-left quadrant
        mask[0, 0] = true; mask[0, 1] = true;
        mask[1, 0] = true; mask[1, 1] = true;

        Color32 red = new Color32(255, 0, 0, 255);
        Color32[] result = Sam2Processor.ApplyMaskOverlay(mask, pixels, w, h, red);

        // Masked pixels should be modified (blue + red overlay)
        Assert.That(result[0].r, Is.GreaterThan(0), "Masked pixel should have red");
        Assert.That(result[0].b, Is.LessThan(255), "Masked pixel blue should be reduced");

        // Unmasked pixels should be unchanged
        Assert.That(result[2 * w].r, Is.EqualTo(0), "Unmasked pixel R unchanged");
        Assert.That(result[2 * w].b, Is.EqualTo(255), "Unmasked pixel B unchanged");

        // Original array should not be modified
        Assert.That(pixels[0].r, Is.EqualTo(0), "Original pixels should not be modified");
    }

    // =======================================================
    // 7. ConvertClickCoordsToB2T
    // =======================================================
    [Test]
    public void ConvertClickCoordsToB2T_FlipsY()
    {
        float[,] t2bCoords = new float[,] { { 500, 375 }, { 100, 0 } };
        float[,] b2tCoords = Sam2Processor.ConvertClickCoordsToB2T(t2bCoords, 1200);

        Assert.That(b2tCoords[0, 0], Is.EqualTo(500f), "X unchanged");
        Assert.That(b2tCoords[0, 1], Is.EqualTo(824f), "Y = 1200-1-375 = 824");
        Assert.That(b2tCoords[1, 0], Is.EqualTo(100f));
        Assert.That(b2tCoords[1, 1], Is.EqualTo(1199f), "Y = 1200-1-0 = 1199");
    }

    // =======================================================
    // 8. DrawClickPoints: markers drawn at correct positions
    // =======================================================
    [Test]
    public void DrawClickPoints_DrawsMarkers()
    {
        int w = 100, h = 100;
        Color32[] pixels = new Color32[w * h];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = new Color32(0, 0, 0, 255);  // all black

        float[,] coords = new float[,] { { 50, 50 } };
        float[] labels = new float[] { 1f };  // positive = green

        Color32[] result = Sam2Processor.DrawClickPoints(coords, labels, pixels, w, h);

        // Center pixel should be green (marker at 50,50)
        Assert.That(result[50 * w + 50].g, Is.EqualTo(255), "Center marker should be green");

        // Original should not be modified
        Assert.That(pixels[50 * w + 50].g, Is.EqualTo(0), "Original should not be modified");

        // Far-away pixel should still be black
        Assert.That(result[0].g, Is.EqualTo(0), "Far pixel should be unmodified");
    }

    // =======================================================
    // 9. DrawClickPoints: box labels (>=2) are skipped
    // =======================================================
    [Test]
    public void DrawClickPoints_SkipsBoxLabels()
    {
        int w = 100, h = 100;
        Color32[] pixels = new Color32[w * h];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = new Color32(0, 0, 0, 255);

        float[,] coords = new float[,] { { 50, 50 }, { 80, 80 } };
        float[] labels = new float[] { 2f, 3f };  // box labels

        Color32[] result = Sam2Processor.DrawClickPoints(coords, labels, pixels, w, h);

        // No markers should be drawn for box labels
        Assert.That(result[50 * w + 50].g, Is.EqualTo(0));
        Assert.That(result[80 * w + 80].g, Is.EqualTo(0));
    }

    // =======================================================
    // 10. Mask overlay on real inference result
    // =======================================================
    [Test]
    public void MaskOverlay_RealInference_CorrectAlignment()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (processor, backend) = CreateOrtProcessor();
        using var _ = backend;

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);

        processor.ProcessEmbedding(b2tPixels, imgW, imgH);
        processor.AddClickPoint(500, 375);
        var result = processor.ProcessMask(imgW, imgH);

        Assert.That(result.HasMask, Is.True);

        Color32[] overlayPixels = Sam2Processor.ApplyMaskOverlay(
            result.B2TMask, b2tPixels, imgW, imgH);

        // Verify alignment: unmasked pixels unchanged, masked pixels modified
        int unmaskedCorrect = 0, unmaskedTotal = 0;
        int maskedModified = 0, maskedTotal = 0;

        for (int y = 0; y < imgH; y++)
        {
            int rowOffset = y * imgW;
            for (int x = 0; x < imgW; x++)
            {
                int idx = rowOffset + x;
                if (result.B2TMask[y, x])
                {
                    maskedTotal++;
                    if (overlayPixels[idx].r != b2tPixels[idx].r ||
                        overlayPixels[idx].g != b2tPixels[idx].g ||
                        overlayPixels[idx].b != b2tPixels[idx].b)
                        maskedModified++;
                }
                else
                {
                    unmaskedTotal++;
                    if (overlayPixels[idx].r == b2tPixels[idx].r &&
                        overlayPixels[idx].g == b2tPixels[idx].g &&
                        overlayPixels[idx].b == b2tPixels[idx].b)
                        unmaskedCorrect++;
                }
            }
        }

        Console.WriteLine($"Masked: {maskedTotal} ({100.0 * maskedModified / Math.Max(1, maskedTotal):F1}% modified)");
        Console.WriteLine($"Unmasked: {unmaskedTotal} ({100.0 * unmaskedCorrect / Math.Max(1, unmaskedTotal):F1}% unchanged)");

        Assert.That(unmaskedCorrect, Is.EqualTo(unmaskedTotal), "All unmasked pixels unchanged");
        Assert.That(maskedTotal, Is.GreaterThan(0));
        Assert.That((double)maskedModified / maskedTotal, Is.GreaterThan(0.95));
    }

    // =======================================================
    // 11. Multiple click points
    // =======================================================
    [Test]
    public void MultipleClickPoints()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (processor, backend) = CreateOrtProcessor();
        using var _ = backend;

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);

        processor.ProcessEmbedding(b2tPixels, imgW, imgH);
        processor.AddClickPoint(500, 375);           // positive (truck)
        processor.AddClickPoint(1200, 100, true);     // negative (background)

        var result = processor.ProcessMask(imgW, imgH);
        Assert.That(result.HasMask, Is.True);

        // Flip B2T mask back to T2B to check in image coordinates
        bool[,] t2bMask = Sam2InferenceEngine.VerticalFlipMask(result.B2TMask);

        Assert.That(t2bMask[375, 500], Is.True, "Positive point should be inside mask");
        Assert.That(t2bMask[100, 1200], Is.False, "Negative point should be outside mask");

        SaveMaskPng(t2bMask, Path.Combine(CSHARP_OUTPUT_DIR, "mask_processor_multipoint.png"));
    }

    // =======================================================
    // 12. Box prompt
    // =======================================================
    [Test]
    public void BoxPrompt()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (processor, backend) = CreateOrtProcessor();
        using var _ = backend;

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);

        processor.ProcessEmbedding(b2tPixels, imgW, imgH);
        processor.SetBoxCoords(new Rect(150, 70, 1200, 900));

        var result = processor.ProcessMask(imgW, imgH);
        Assert.That(result.HasMask, Is.True);

        bool[,] t2bMask = Sam2InferenceEngine.VerticalFlipMask(result.B2TMask);
        int boxCenterX = 150 + 1200 / 2;
        int boxCenterY = 70 + 900 / 2;
        Assert.That(t2bMask[boxCenterY, boxCenterX], Is.True, "Box center should be inside mask");

        SaveMaskPng(t2bMask, Path.Combine(CSHARP_OUTPUT_DIR, "mask_processor_box.png"));
    }

    // =======================================================
    // 13. Full visualization pipeline (overlay + click points)
    // =======================================================
    [Test]
    public void FullVisualizationPipeline()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (processor, backend) = CreateOrtProcessor();
        using var _ = backend;

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);

        processor.ProcessEmbedding(b2tPixels, imgW, imgH);
        processor.AddClickPoint(500, 375);

        var result = processor.ProcessMask(imgW, imgH);
        Assert.That(result.HasMask, Is.True);

        // Full visualization: overlay + click points
        Color32[] overlayPixels = Sam2Processor.ApplyMaskOverlay(
            result.B2TMask, b2tPixels, imgW, imgH);

        float[,] t2bCoords = processor.GetClickPoints(imgH);
        float[] labels = processor.GetPointLabels();
        float[,] b2tCoords = Sam2Processor.ConvertClickCoordsToB2T(t2bCoords, imgH);

        Color32[] finalPixels = Sam2Processor.DrawClickPoints(
            b2tCoords, labels, overlayPixels, imgW, imgH);

        // The click point at B2T coords should have a green marker
        int b2tClickY = imgH - 1 - 375;  // = 824
        int markerIdx = b2tClickY * imgW + 500;
        Assert.That(finalPixels[markerIdx].g, Is.EqualTo(255),
            "Click point marker should be green");

        Console.WriteLine($"Visualization pipeline complete. Output size: {finalPixels.Length}");
    }
}
