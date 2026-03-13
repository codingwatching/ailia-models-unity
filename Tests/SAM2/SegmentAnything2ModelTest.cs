/* SAM2 SegmentAnything2Model Logic Test */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * Tests the data flow of SegmentAnything2Model without Unity dependencies.
 * SegmentAnything2Model owns the following logic beyond Sam2InferenceEngine:
 *
 *   ProcessEmbedding(B2T image):
 *     1. VerticalFlip B2T -> T2B
 *     2. RunEmbedding(T2B image) -> encoderOutput, highResFeats
 *
 *   ProcessMask(B2T image):
 *     1. GetClickPoints(imageHeight) -> T2B coords
 *     2. RunInference(T2B coords) -> T2B masks + scores
 *     3. FindBestMaskIndex(scores)
 *     4. VerticalFlipMask T2B -> B2T
 *     5. CreateMaskedImage(B2T mask, B2T pixels)
 *     6. Convert T2B click coords -> B2T for DrawClickPoints
 *
 * These tests replicate the exact same flow using ISam2Backend,
 * ensuring correctness of the B2T/T2B conversions and mask overlay.
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
    private Sam2InferenceEngine engine = null!;

    private const string MODEL_DIR = "/tmp/sam2_models";
    private const string PNG_IMAGE_PATH = "/tmp/sam2_test_output/truck.png";
    private const string PYTHON_OUTPUT_DIR = "/tmp/sam2_test_output";
    private const string CSHARP_OUTPUT_DIR = "/tmp/sam2_csharp_output";

    private const int IMG_W = 1800;
    private const int IMG_H = 1200;
    private const int TARGET_SIZE = 1024;

    [SetUp]
    public void SetUp()
    {
        engine = new Sam2InferenceEngine();
        Directory.CreateDirectory(CSHARP_OUTPUT_DIR);
    }

    private string EncoderPath => Path.Combine(MODEL_DIR, "image_encoder_hiera_l.onnx");
    private string DecoderPath => Path.Combine(MODEL_DIR, "mask_decoder_hiera_l.onnx");
    private string PromptPath => Path.Combine(MODEL_DIR, "prompt_encoder_hiera_l.onnx");

    private bool ModelsExist() =>
        File.Exists(EncoderPath) && File.Exists(DecoderPath) && File.Exists(PromptPath);

    // =======================================================
    // Load PNG as T2B Color32 array (standard image format)
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
    // Simulate Unity GetPixels32: T2B -> B2T
    // =======================================================
    private Color32[] SimulateUnityGetPixels32(Color32[] top2bottom, int width, int height)
    {
        return Sam2InferenceEngine.VerticalFlip(top2bottom, width, height);
    }

    // =======================================================
    // Load Python .npy bool mask
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

    // =======================================================
    // Save bool mask as PNG
    // =======================================================
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

    // =======================================================
    // Save diff image (white=both, red=only C#, blue=only Python)
    // =======================================================
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

    // =======================================================
    // Compare masks helper
    // =======================================================
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
    // Replicate SegmentAnything2Model.RunEmbedding
    //   - Takes T2B image (already flipped from B2T)
    //   - Preprocesses and encodes
    //   - Returns encoderOutput + highResFeats
    // =======================================================
    private (float[] encoderOutput, float[][] highResFeats)
        ReplicateRunEmbedding(ISam2Backend backend, Color32[] t2bImage, int imgWidth, int imgHeight)
    {
        float[,,,] inputTensor = engine.PreprocessImage(t2bImage, imgWidth, imgHeight, TARGET_SIZE);
        float[] nchwInput = engine.Flatten4D(inputTensor);

        var encOut = backend.RunEncoder(nchwInput);
        return engine.PrepareEncoderFeatures(encOut);
    }

    // =======================================================
    // Replicate SegmentAnything2Model.RunInference
    //   - Takes encoderOutput, highResFeats, T2B coords, labels
    //   - Runs prompt encoder + decoder
    //   - Returns T2B masks + IoU scores
    // =======================================================
    private (bool[][,] masks, float[] iouPred)
        ReplicateRunInference(
            ISam2Backend backend,
            float[] encoderOutput, float[][] highResFeats,
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

    // =======================================================
    // Replicate SegmentAnything2Model.CreateMaskedImage logic
    //   - Both mask and pixels are B2T
    //   - mask[y,x] maps to pixels[y*width+x]
    // =======================================================
    private Color32[] ReplicateCreateMaskedImage(
        bool[,] mask, Color32[] pixels, int imageWidth, int imageHeight)
    {
        Color32[] result = (Color32[])pixels.Clone();
        Color32 maskColor = new Color32(255, 0, 0, 255);
        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        for (int y = 0; y < maskHeight; y++)
        {
            int rowOffset = y * imageWidth;
            for (int x = 0; x < maskWidth; x++)
            {
                int pixelIndex = rowOffset + x;
                if (pixelIndex >= 0 && pixelIndex < pixels.Length && mask[y, x])
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

    // =======================================================
    // 1. Click point management via engine: T2B coordinate system
    // =======================================================
    [Test]
    public void ClickPointManagement_T2B_Coordinates()
    {
        // AiliaImageSegmentationSample computes click coords as:
        //   x = pixel x in texture
        //   y = textureHeight - 1 - mouseY  (screen B2T -> T2B)
        // Then calls sam2Model.AddClickPoint(x, y)

        int clickX = 500;
        int clickY = 375;  // T2B coordinate

        engine.AddClickPoint(clickX, clickY);
        float[,] coords = engine.GetClickPoints(IMG_H);
        float[] labels = engine.GetPointLabels();

        Assert.That(coords.GetLength(0), Is.EqualTo(1));
        Assert.That(coords[0, 0], Is.EqualTo((float)clickX));
        Assert.That(coords[0, 1], Is.EqualTo((float)clickY));
        Assert.That(labels[0], Is.EqualTo(1f));  // positive point

        // Negative point
        engine.AddClickPoint(100, 200, negativePoint: true);
        coords = engine.GetClickPoints(IMG_H);
        labels = engine.GetPointLabels();

        Assert.That(coords.GetLength(0), Is.EqualTo(2));
        Assert.That(labels[1], Is.EqualTo(0f));  // negative point
    }

    // =======================================================
    // 2. Box coordinates via engine
    // =======================================================
    [Test]
    public void BoxCoords_AreIncludedInPoints()
    {
        engine.AddClickPoint(100, 100);
        engine.SetBoxCoords(new Rect(50, 50, 200, 150));  // x,y,w,h

        float[,] coords = engine.GetClickPoints(IMG_H);
        float[] labels = engine.GetPointLabels();

        // 1 click point + 2 box corners
        Assert.That(coords.GetLength(0), Is.EqualTo(3));
        Assert.That(labels[0], Is.EqualTo(1f));   // click point
        Assert.That(labels[1], Is.EqualTo(2f));   // box top-left
        Assert.That(labels[2], Is.EqualTo(3f));   // box bottom-right

        // Box corners: xMin,yMin and xMax,yMax
        Assert.That(coords[1, 0], Is.EqualTo(50f));   // xMin
        Assert.That(coords[1, 1], Is.EqualTo(50f));   // yMin
        Assert.That(coords[2, 0], Is.EqualTo(250f));  // xMax = x + width
        Assert.That(coords[2, 1], Is.EqualTo(200f));  // yMax = y + height
    }

    // =======================================================
    // 3. ResetClickPoint clears all state
    // =======================================================
    [Test]
    public void ResetClickPoint_ClearsAll()
    {
        engine.AddClickPoint(100, 100);
        engine.SetBoxCoords(new Rect(50, 50, 200, 150));
        engine.ResetClickPoint();

        float[,] coords = engine.GetClickPoints(IMG_H);
        Assert.That(coords.GetLength(0), Is.EqualTo(0));
    }

    // =======================================================
    // 4. ProcessEmbedding flow: B2T -> flip -> T2B -> encode
    //    Verify B2T flip produces same encoding as direct T2B
    // =======================================================
    [Test]
    public void ProcessEmbedding_B2TFlip_ProducesSameEncoding()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found in " + MODEL_DIR);
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);

        // Simulate Unity: T2B -> GetPixels32 (B2T) -> ProcessEmbedding
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);

        // ProcessEmbedding internally flips B2T -> T2B
        Color32[] flippedToT2B = Sam2InferenceEngine.VerticalFlip(b2tPixels, imgW, imgH);

        using var backend = new OrtSam2Backend();
        backend.LoadModels(EncoderPath, DecoderPath, PromptPath);

        // B2T path (what SegmentAnything2Model does)
        var (encOutputB2T, highResB2T) = ReplicateRunEmbedding(backend, flippedToT2B, imgW, imgH);

        // Direct T2B path (reference)
        var (encOutputDirect, highResDirect) = ReplicateRunEmbedding(backend, t2bPixels, imgW, imgH);

        // Should be identical
        Assert.That(encOutputB2T.Length, Is.EqualTo(encOutputDirect.Length));
        float maxDiff = 0;
        for (int i = 0; i < encOutputB2T.Length; i++)
        {
            float diff = Math.Abs(encOutputB2T[i] - encOutputDirect[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        Console.WriteLine($"Encoder output max diff: {maxDiff}");
        Assert.That(maxDiff, Is.LessThan(1e-5f), "B2T path should produce identical encoding");
    }

    // =======================================================
    // 5. Full ProcessEmbedding + ProcessMask flow:
    //    B2T input -> T2B inference -> B2T mask output
    //    Compare with Python reference
    // =======================================================
    [Test]
    public void FullFlow_ProcessEmbeddingAndMask_MatchesPython()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found in " + MODEL_DIR);
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);

        // === Simulate SegmentAnything2Model flow ===

        // Step 1: Simulate Unity GetPixels32 (B2T)
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);

        // Step 2: ProcessEmbedding - flip B2T -> T2B
        Color32[] t2bForInference = Sam2InferenceEngine.VerticalFlip(b2tPixels, imgW, imgH);

        using var backend = new OrtSam2Backend();
        backend.LoadModels(EncoderPath, DecoderPath, PromptPath);

        var (encoderOutput, highResFeats) = ReplicateRunEmbedding(backend, t2bForInference, imgW, imgH);

        // Step 3: Add click point (T2B coordinates, same as AiliaImageSegmentationSample)
        int clickX = 500;
        int clickY = 375;
        engine.AddClickPoint(clickX, clickY);

        // Step 4: ProcessMask - get T2B coords from engine
        float[,] coords = engine.GetClickPoints(imgH);
        float[] labels = engine.GetPointLabels();

        Console.WriteLine($"Click point: ({coords[0, 0]}, {coords[0, 1]}) label={labels[0]}");

        // Step 5: RunInference with T2B coords
        var (masks, iouPred) = ReplicateRunInference(
            backend, encoderOutput, highResFeats,
            imgW, imgH, coords, labels);

        Assert.That(masks.Length, Is.GreaterThan(0), "Should produce masks");
        Assert.That(iouPred.Length, Is.GreaterThan(0), "Should produce IoU predictions");

        // Step 6: Find best mask (T2B)
        int bestIdx = engine.FindBestMaskIndex(iouPred);
        bool[,] t2bBestMask = masks[bestIdx];

        Console.WriteLine($"Best mask: index={bestIdx}, score={iouPred[bestIdx]:F4}");
        Console.WriteLine($"Mask shape: {t2bBestMask.GetLength(0)} x {t2bBestMask.GetLength(1)}");

        int trueCount = 0;
        for (int y = 0; y < imgH; y++)
            for (int x = 0; x < imgW; x++)
                if (t2bBestMask[y, x]) trueCount++;
        Console.WriteLine($"T2B mask true pixels: {trueCount}/{imgW * imgH} ({100.0 * trueCount / (imgW * imgH):F1}%)");

        // Step 7: Flip mask T2B -> B2T (what SegmentAnything2Model does)
        bool[,] b2tBestMask = Sam2InferenceEngine.VerticalFlipMask(t2bBestMask);

        // Save T2B mask for comparison
        SaveMaskPng(t2bBestMask, Path.Combine(CSHARP_OUTPUT_DIR, "mask_sam2model_t2b.png"));
        SaveMaskPng(b2tBestMask, Path.Combine(CSHARP_OUTPUT_DIR, "mask_sam2model_b2t.png"));

        // Step 8: Compare T2B mask with Python reference
        string pyMaskPath = Path.Combine(PYTHON_OUTPUT_DIR, "best_mask_bool_png.npy");
        if (File.Exists(pyMaskPath))
        {
            bool[,] pyMask = LoadNpyBool(pyMaskPath);
            Assert.That(t2bBestMask.GetLength(0), Is.EqualTo(pyMask.GetLength(0)));
            Assert.That(t2bBestMask.GetLength(1), Is.EqualTo(pyMask.GetLength(1)));

            var (matchRate, diffCount, onlyCs, onlyPy) = CompareMasks(t2bBestMask, pyMask);
            Console.WriteLine($"\n=== Python Comparison (SegmentAnything2Model flow) ===");
            Console.WriteLine($"Match rate: {matchRate:F4}%");
            Console.WriteLine($"Different: {diffCount}");

            SaveDiffPng(t2bBestMask, pyMask, Path.Combine(CSHARP_OUTPUT_DIR, "diff_sam2model_vs_python.png"));

            Assert.That(matchRate, Is.GreaterThan(99.0),
                $"Mask match rate should be > 99%, got {matchRate:F4}%");
        }
        else
        {
            Console.WriteLine("Python reference mask not found, skipping comparison.");
        }
    }

    // =======================================================
    // 6. Mask overlay: B2T mask on B2T pixels
    //    Verifies CreateMaskedImage logic alignment
    // =======================================================
    [Test]
    public void MaskOverlay_B2T_CorrectAlignment()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found in " + MODEL_DIR);
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);

        // Run inference (same as SegmentAnything2Model)
        Color32[] t2bForInference = Sam2InferenceEngine.VerticalFlip(b2tPixels, imgW, imgH);

        using var backend = new OrtSam2Backend();
        backend.LoadModels(EncoderPath, DecoderPath, PromptPath);

        var (encoderOutput, highResFeats) = ReplicateRunEmbedding(backend, t2bForInference, imgW, imgH);

        engine.AddClickPoint(500, 375);
        float[,] coords = engine.GetClickPoints(imgH);
        float[] labels = engine.GetPointLabels();

        var (masks, iouPred) = ReplicateRunInference(
            backend, encoderOutput, highResFeats,
            imgW, imgH, coords, labels);

        int bestIdx = engine.FindBestMaskIndex(iouPred);
        bool[,] b2tMask = Sam2InferenceEngine.VerticalFlipMask(masks[bestIdx]);

        // Apply overlay (CreateMaskedImage logic)
        Color32[] overlayPixels = ReplicateCreateMaskedImage(b2tMask, b2tPixels, imgW, imgH);

        // Verify: masked pixels differ from original, unmasked are identical
        int maskedCount = 0, unmaskedCount = 0;
        int maskedCorrect = 0, unmaskedCorrect = 0;

        for (int y = 0; y < imgH; y++)
        {
            int rowOffset = y * imgW;
            for (int x = 0; x < imgW; x++)
            {
                int idx = rowOffset + x;
                if (b2tMask[y, x])
                {
                    maskedCount++;
                    // Should be modified (blended with red)
                    if (overlayPixels[idx].r != b2tPixels[idx].r ||
                        overlayPixels[idx].g != b2tPixels[idx].g ||
                        overlayPixels[idx].b != b2tPixels[idx].b)
                        maskedCorrect++;
                }
                else
                {
                    unmaskedCount++;
                    // Should be unchanged
                    if (overlayPixels[idx].r == b2tPixels[idx].r &&
                        overlayPixels[idx].g == b2tPixels[idx].g &&
                        overlayPixels[idx].b == b2tPixels[idx].b)
                        unmaskedCorrect++;
                }
            }
        }

        Console.WriteLine($"Masked pixels: {maskedCount} ({100.0 * maskedCorrect / Math.Max(1, maskedCount):F1}% correctly modified)");
        Console.WriteLine($"Unmasked pixels: {unmaskedCount} ({100.0 * unmaskedCorrect / Math.Max(1, unmaskedCount):F1}% correctly unchanged)");

        Assert.That(maskedCount, Is.GreaterThan(0), "Should have masked pixels");
        Assert.That(unmaskedCount, Is.GreaterThan(0), "Should have unmasked pixels");
        Assert.That(unmaskedCorrect, Is.EqualTo(unmaskedCount),
            "All unmasked pixels should be unchanged");
        // Some pixels might have pure black/white that blend to same value,
        // but the vast majority should be modified
        Assert.That((double)maskedCorrect / maskedCount, Is.GreaterThan(0.95),
            "Most masked pixels should be modified");
    }

    // =======================================================
    // 7. Click coord T2B -> B2T conversion for DrawClickPoints
    //    SegmentAnything2Model.ProcessMask converts coords for visualization
    // =======================================================
    [Test]
    public void ClickCoords_T2B_To_B2T_ForVisualization()
    {
        int imageHeight = 1200;

        // Add click at T2B (500, 375) - near top of image
        engine.AddClickPoint(500, 375);
        float[,] t2bCoords = engine.GetClickPoints(imageHeight);

        // ProcessMask converts T2B -> B2T for DrawClickPoints:
        //   b2tCoords[i, 1] = imageHeight - 1 - coords[i, 1]
        float[,] b2tCoords = new float[t2bCoords.GetLength(0), 2];
        for (int i = 0; i < t2bCoords.GetLength(0); i++)
        {
            b2tCoords[i, 0] = t2bCoords[i, 0];
            b2tCoords[i, 1] = imageHeight - 1 - t2bCoords[i, 1];
        }

        Assert.That(b2tCoords[0, 0], Is.EqualTo(500f), "X unchanged");
        Assert.That(b2tCoords[0, 1], Is.EqualTo(824f), "Y flipped: 1200-1-375=824");

        // In B2T, y=824 is near top of image (high row index = top)
        Assert.That(b2tCoords[0, 1], Is.GreaterThan(imageHeight / 2),
            "B2T click near top should have high Y value");
    }

    // =======================================================
    // 8. Multiple click points: the full ProcessMask flow
    // =======================================================
    [Test]
    public void MultipleClickPoints_ProcessMaskFlow()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found in " + MODEL_DIR);
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);
        Color32[] t2bForInference = Sam2InferenceEngine.VerticalFlip(b2tPixels, imgW, imgH);

        using var backend = new OrtSam2Backend();
        backend.LoadModels(EncoderPath, DecoderPath, PromptPath);

        var (encoderOutput, highResFeats) = ReplicateRunEmbedding(backend, t2bForInference, imgW, imgH);

        // Add positive point on truck + negative point on background
        engine.AddClickPoint(500, 375);           // positive (on truck)
        engine.AddClickPoint(1200, 100, true);     // negative (background, top-right)

        float[,] coords = engine.GetClickPoints(imgH);
        float[] labels = engine.GetPointLabels();

        Assert.That(coords.GetLength(0), Is.EqualTo(2));
        Assert.That(labels[0], Is.EqualTo(1f), "First point positive");
        Assert.That(labels[1], Is.EqualTo(0f), "Second point negative");

        var (masks, iouPred) = ReplicateRunInference(
            backend, encoderOutput, highResFeats,
            imgW, imgH, coords, labels);

        Assert.That(masks.Length, Is.GreaterThan(0));
        int bestIdx = engine.FindBestMaskIndex(iouPred);
        bool[,] bestMask = masks[bestIdx];

        // The positive point (500, 375) should be inside the mask
        Assert.That(bestMask[375, 500], Is.True,
            "Positive click point should be inside mask");

        // The negative point (1200, 100) should be outside the mask
        Assert.That(bestMask[100, 1200], Is.False,
            "Negative click point should be outside mask");

        Console.WriteLine($"Multi-point: best mask index={bestIdx}, score={iouPred[bestIdx]:F4}");

        SaveMaskPng(bestMask, Path.Combine(CSHARP_OUTPUT_DIR, "mask_sam2model_multipoint.png"));
    }

    // =======================================================
    // 9. Box prompt: ProcessMask flow with box coordinates
    // =======================================================
    [Test]
    public void BoxPrompt_ProcessMaskFlow()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found in " + MODEL_DIR);
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);
        Color32[] b2tPixels = SimulateUnityGetPixels32(t2bPixels, imgW, imgH);
        Color32[] t2bForInference = Sam2InferenceEngine.VerticalFlip(b2tPixels, imgW, imgH);

        using var backend = new OrtSam2Backend();
        backend.LoadModels(EncoderPath, DecoderPath, PromptPath);

        var (encoderOutput, highResFeats) = ReplicateRunEmbedding(backend, t2bForInference, imgW, imgH);

        // Set box around the truck area (T2B coordinates)
        // AiliaImageSegmentationSample sets box as Rect(xMin, yMin, width, height)
        engine.SetBoxCoords(new Rect(150, 70, 1200, 900));

        float[,] coords = engine.GetClickPoints(imgH);
        float[] labels = engine.GetPointLabels();

        // Box adds 2 points with labels 2 and 3
        Assert.That(coords.GetLength(0), Is.EqualTo(2));
        Assert.That(labels[0], Is.EqualTo(2f), "Box top-left label");
        Assert.That(labels[1], Is.EqualTo(3f), "Box bottom-right label");

        var (masks, iouPred) = ReplicateRunInference(
            backend, encoderOutput, highResFeats,
            imgW, imgH, coords, labels);

        Assert.That(masks.Length, Is.GreaterThan(0));
        int bestIdx = engine.FindBestMaskIndex(iouPred);
        bool[,] bestMask = masks[bestIdx];

        // Center of box should be inside mask
        int boxCenterX = 150 + 1200 / 2;
        int boxCenterY = 70 + 900 / 2;
        Assert.That(bestMask[boxCenterY, boxCenterX], Is.True,
            "Box center should be inside mask");

        Console.WriteLine($"Box prompt: best mask index={bestIdx}, score={iouPred[bestIdx]:F4}");

        SaveMaskPng(bestMask, Path.Combine(CSHARP_OUTPUT_DIR, "mask_sam2model_box.png"));
    }

    // =======================================================
    // 10. Verify RunFullPipeline produces same result as
    //     step-by-step SegmentAnything2Model flow
    // =======================================================
    [Test]
    public void RunFullPipeline_MatchesStepByStep()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found in " + MODEL_DIR);
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("truck.png not found");

        var (t2bPixels, imgW, imgH) = LoadPngTopToBottom(PNG_IMAGE_PATH);

        using var backend = new OrtSam2Backend();
        backend.LoadModels(EncoderPath, DecoderPath, PromptPath);

        int clickX = 500, clickY = 375;
        float[,] pointCoords = new float[,] { { clickX, clickY } };
        float[] pointLabels = new float[] { 1f };

        // Method 1: RunFullPipeline (one-shot)
        var (pipelineMasks, pipelineIou) = engine.RunFullPipeline(
            backend, t2bPixels, imgW, imgH, pointCoords, pointLabels);

        // Method 2: Step-by-step (replicating SegmentAnything2Model)
        var (encoderOutput, highResFeats) = ReplicateRunEmbedding(backend, t2bPixels, imgW, imgH);
        var (stepMasks, stepIou) = ReplicateRunInference(
            backend, encoderOutput, highResFeats,
            imgW, imgH, pointCoords, pointLabels);

        // Both should produce the same results
        Assert.That(pipelineMasks.Length, Is.EqualTo(stepMasks.Length), "Same number of masks");
        Assert.That(pipelineIou.Length, Is.EqualTo(stepIou.Length), "Same number of IoU scores");

        for (int i = 0; i < pipelineIou.Length; i++)
            Assert.That(pipelineIou[i], Is.EqualTo(stepIou[i]).Within(1e-5f),
                $"IoU[{i}] should match");

        int pipelineBest = engine.FindBestMaskIndex(pipelineIou);
        int stepBest = engine.FindBestMaskIndex(stepIou);
        Assert.That(pipelineBest, Is.EqualTo(stepBest), "Same best mask index");

        // Masks should be identical
        var (matchRate, diffCount, _, _) = CompareMasks(
            pipelineMasks[pipelineBest], stepMasks[stepBest]);
        Console.WriteLine($"RunFullPipeline vs step-by-step: match rate = {matchRate:F4}%");
        Assert.That(matchRate, Is.EqualTo(100.0),
            "RunFullPipeline should produce identical masks to step-by-step");
    }
}
