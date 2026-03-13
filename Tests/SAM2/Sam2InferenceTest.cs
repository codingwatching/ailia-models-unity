/* SAM2 Backend-Agnostic Inference Test */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * End-to-end inference test using ISam2Backend interface.
 * Runs the same test logic with both ORT and ailia backends.
 *
 * Test image: truck.png (1800x1200, lossless PNG)
 * Click point: (500, 375), label=1 (foreground)
 * Models: segment-anything-2 (Hiera-L)
 *
 * The inference pipeline matches SegmentAnything2Model.cs:
 *   1. PreprocessImage (bilinear resize, normalize)
 *   2. RunEncoder -> PrepareBackboneFeatures -> feats
 *   3. ApplyCoordinateScaling -> RunPromptEncoder
 *   4. RunDecoder -> PostprocessMasks -> ConvertToBoolMasks
 *
 * Note: no_mem_embed is NOT added (single-image mode).
 */

using NUnit.Framework;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.IO;
using System.Linq;
using UnityEngine;

[TestFixture]
public class Sam2InferenceTest
{
    private Sam2InferenceEngine logic = null!;

    private const string MODEL_DIR = "/tmp/sam2_models";
    private const string PNG_IMAGE_PATH = "/tmp/sam2_test_output/truck.png";
    private const string PYTHON_OUTPUT_DIR = "/tmp/sam2_test_output";
    private const string CSHARP_OUTPUT_DIR = "/tmp/sam2_csharp_output";

    private const int CLICK_X = 500;
    private const int CLICK_Y = 375;
    private const int POINT_LABEL = 1;
    private const int IMG_W = 1800;
    private const int IMG_H = 1200;

    [SetUp]
    public void SetUp()
    {
        logic = new Sam2InferenceEngine();
        Directory.CreateDirectory(CSHARP_OUTPUT_DIR);
    }

    private string EncoderPath => Path.Combine(MODEL_DIR, "image_encoder_hiera_l.onnx");
    private string DecoderPath => Path.Combine(MODEL_DIR, "mask_decoder_hiera_l.onnx");
    private string PromptPath => Path.Combine(MODEL_DIR, "prompt_encoder_hiera_l.onnx");

    private bool ModelsExist()
    {
        return File.Exists(EncoderPath)
            && File.Exists(DecoderPath)
            && File.Exists(PromptPath);
    }

    private bool AiliaModelsExist()
    {
        return ModelsExist()
            && File.Exists(EncoderPath + ".prototxt")
            && File.Exists(DecoderPath + ".prototxt")
            && File.Exists(PromptPath + ".prototxt");
    }

    // =======================================================
    // Load PNG image as Color32 array (top-to-bottom)
    // =======================================================
    private (Color32[] pixels, int width, int height) LoadPngImage(string path)
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
    // Full inference pipeline (shared between ORT and ailia)
    // Matches SegmentAnything2Model.cs RunEmbedding + RunInference
    // =======================================================
    private (bool[,] bestMask, int bestIdx, float bestScore, float[] iouPred)
        RunFullPipeline(ISam2Backend backend, float[] nchwInput)
    {
        // Step 1: Encoder
        var encOut = backend.RunEncoder(nchwInput);

        Console.WriteLine($"Encoder outputs:");
        Console.WriteLine($"  backbone_fpn_0: [{encOut.Fpn0Shape.N},{encOut.Fpn0Shape.C},{encOut.Fpn0Shape.H},{encOut.Fpn0Shape.W}]");
        Console.WriteLine($"  backbone_fpn_1: [{encOut.Fpn1Shape.N},{encOut.Fpn1Shape.C},{encOut.Fpn1Shape.H},{encOut.Fpn1Shape.W}]");
        Console.WriteLine($"  backbone_fpn_2: [{encOut.Fpn2Shape.N},{encOut.Fpn2Shape.C},{encOut.Fpn2Shape.H},{encOut.Fpn2Shape.W}]");

        // Step 2: Prepare backbone features
        // (same logic as SegmentAnything2Model.RunEmbedding)
        float[,,,] fpn0_4d = logic.ReshapeTo4D(encOut.BackboneFpn0,
            encOut.Fpn0Shape.N, encOut.Fpn0Shape.C, encOut.Fpn0Shape.H, encOut.Fpn0Shape.W);
        float[,,,] fpn1_4d = logic.ReshapeTo4D(encOut.BackboneFpn1,
            encOut.Fpn1Shape.N, encOut.Fpn1Shape.C, encOut.Fpn1Shape.H, encOut.Fpn1Shape.W);
        float[,,,] fpn2_4d = logic.ReshapeTo4D(encOut.BackboneFpn2,
            encOut.Fpn2Shape.N, encOut.Fpn2Shape.C, encOut.Fpn2Shape.H, encOut.Fpn2Shape.W);

        float[][,,,] backboneFpn = new[] { fpn0_4d, fpn1_4d, fpn2_4d };
        float[][,,] visionFeats = logic.PrepareBackboneFeatures(backboneFpn);

        // Note: no_mem_embed is NOT added for single-image inference
        // (fixed bug: previously TruncNormal random noise was added here)

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

        // Step 3: Coordinate scaling + Prompt encoder
        float[,] rawCoords = new float[,] { { CLICK_X, CLICK_Y } };
        float[,] scaledCoords = logic.ApplyCoordinateScaling(rawCoords, IMG_H, IMG_W);
        Console.WriteLine($"Scaled coords: ({scaledCoords[0,0]:F4}, {scaledCoords[0,1]:F4})");

        float[] flatCoords = { scaledCoords[0, 0], scaledCoords[0, 1] };
        float[] labels = { POINT_LABEL };
        float[] maskInputDummy = new float[256 * 256];

        var promptOut = backend.RunPromptEncoder(flatCoords, 1, labels, maskInputDummy, 0f);

        // Step 4: Mask decoder
        var decOut = backend.RunDecoder(
            imageEmbFlat,
            promptOut.DensePe, promptOut.DensePeShape,
            promptOut.SparseEmbeddings, promptOut.SparseShape,
            promptOut.DenseEmbeddings, promptOut.DenseShape,
            highResFeat0,
            highResFeat1
        );

        Console.WriteLine($"Raw masks shape: [{string.Join(",", decOut.MasksShape)}]");
        Console.WriteLine($"IoU predictions: [{string.Join(", ", decOut.IouPred.Select(v => $"{v:F4}"))}]");

        // Step 5: Postprocess masks
        float[,,,] masks4d = logic.ReshapeTo4D(decOut.Masks,
            decOut.MasksShape[0], decOut.MasksShape[1],
            decOut.MasksShape[2], decOut.MasksShape[3]);
        float[,,,] resizedMasks = logic.PostprocessMasks(masks4d, IMG_H, IMG_W);
        bool[][,] boolMasks = logic.ConvertToBoolMasks(resizedMasks, 0.0f);

        // Find best mask
        int bestIdx = 0;
        float maxScore = float.MinValue;
        for (int i = 0; i < decOut.IouPred.Length; i++)
            if (decOut.IouPred[i] > maxScore) { maxScore = decOut.IouPred[i]; bestIdx = i; }

        int trueCount = 0;
        var bestMask = boolMasks[bestIdx];
        for (int y = 0; y < IMG_H; y++)
            for (int x = 0; x < IMG_W; x++)
                if (bestMask[y, x]) trueCount++;

        Console.WriteLine($"Best mask: index={bestIdx}, score={maxScore:F4}");
        Console.WriteLine($"True pixels: {trueCount}/{IMG_W * IMG_H} ({100.0 * trueCount / (IMG_W * IMG_H):F1}%)");

        return (bestMask, bestIdx, maxScore, decOut.IouPred);
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
    // Run full pipeline with a given backend and compare to Python
    // =======================================================
    private void RunEndToEndTest(ISam2Backend backend, string backendName)
    {
        Console.WriteLine($"=== SAM2 End-to-End ({backendName}) ===");

        // Preprocess
        var (pixels, imgW, imgH) = LoadPngImage(PNG_IMAGE_PATH);
        Assert.That(imgW, Is.EqualTo(IMG_W));
        Assert.That(imgH, Is.EqualTo(IMG_H));

        float[,,,] preprocessed = logic.PreprocessImage(pixels, imgW, imgH, 1024);
        float[] nchwInput = logic.Flatten4D(preprocessed);
        Console.WriteLine($"Preprocessed: {nchwInput.Length} values, range=[{nchwInput.Min():F4}, {nchwInput.Max():F4}]");

        // Run inference
        var (bestMask, bestIdx, bestScore, iouPred) = RunFullPipeline(backend, nchwInput);

        SaveMaskPng(bestMask, Path.Combine(CSHARP_OUTPUT_DIR,
            $"mask_{backendName.ToLower()}.png"));

        // Compare with Python reference
        string pyMaskPath = Path.Combine(PYTHON_OUTPUT_DIR, "best_mask_bool_png.npy");
        if (!File.Exists(pyMaskPath))
        {
            Console.WriteLine("Python reference mask not found, skipping comparison.");
            return;
        }

        bool[,] pyMask = LoadNpyBool(pyMaskPath);
        Assert.That(bestMask.GetLength(0), Is.EqualTo(pyMask.GetLength(0)));
        Assert.That(bestMask.GetLength(1), Is.EqualTo(pyMask.GetLength(1)));

        var (matchRate, diffCount, onlyCs, onlyPy) = CompareMasks(bestMask, pyMask);
        Console.WriteLine($"\n=== Mask Comparison ({backendName}) ===");
        Console.WriteLine($"Total pixels: {IMG_W * IMG_H}");
        Console.WriteLine($"Matching: {IMG_W * IMG_H - diffCount} ({matchRate:F4}%)");
        Console.WriteLine($"Different: {diffCount} ({100.0 * diffCount / (IMG_W * IMG_H):F4}%)");
        Console.WriteLine($"  Only in C#: {onlyCs}");
        Console.WriteLine($"  Only in Python: {onlyPy}");

        SaveDiffPng(bestMask, pyMask, Path.Combine(CSHARP_OUTPUT_DIR,
            $"diff_{backendName.ToLower()}.png"));

        Assert.That(matchRate, Is.GreaterThan(99.0),
            $"{backendName} mask match rate should be > 99%, got {matchRate:F4}%");
    }

    // =======================================================
    // Test: ORT Backend
    // =======================================================
    [Test]
    public void FullInference_ORT()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found in " + MODEL_DIR);
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("PNG test image not found.");

        using var backend = new OrtSam2Backend();
        backend.LoadModels(EncoderPath, DecoderPath, PromptPath);
        RunEndToEndTest(backend, "ORT");
    }

    // =======================================================
    // Test: ailia Backend
    // Requires: AILIA_SDK define + ailia-csharp wrapper + native library
    // Build with: dotnet test -p:DefineConstants=AILIA_SDK
    // =======================================================
#if AILIA_SDK
    [Test]
    public void FullInference_Ailia()
    {
        if (!AiliaModelsExist())
            Assert.Ignore("ailia models (ONNX + prototxt) not found in " + MODEL_DIR);
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("PNG test image not found.");

        // Verify ailia native library is available
        try
        {
            using var backend = new AiliaSam2Backend();
            backend.LoadModels(EncoderPath, DecoderPath, PromptPath);
            RunEndToEndTest(backend, "Ailia");
        }
        catch (DllNotFoundException ex)
        {
            Assert.Ignore($"ailia native library not found: {ex.Message}");
        }
        catch (TypeLoadException ex)
        {
            Assert.Ignore($"ailia SDK types not available: {ex.Message}");
        }
    }

    // =======================================================
    // Test: Compare ORT vs ailia outputs
    // =======================================================
    [Test]
    public void CompareBackends_ORT_vs_Ailia()
    {
        if (!AiliaModelsExist())
            Assert.Ignore("ailia models not found");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("PNG test image not found.");

        var (pixels, imgW, imgH) = LoadPngImage(PNG_IMAGE_PATH);
        float[,,,] preprocessed = logic.PreprocessImage(pixels, imgW, imgH, 1024);
        float[] nchwInput = logic.Flatten4D(preprocessed);

        bool[,] ortMask, ailiaMask;

        try
        {
            using var ortBackend = new OrtSam2Backend();
            ortBackend.LoadModels(EncoderPath, DecoderPath, PromptPath);
            (ortMask, _, _, _) = RunFullPipeline(ortBackend, nchwInput);
        }
        catch (Exception ex)
        {
            Assert.Ignore($"ORT backend failed: {ex.Message}");
            return;
        }

        try
        {
            using var ailiaBackend = new AiliaSam2Backend();
            ailiaBackend.LoadModels(EncoderPath, DecoderPath, PromptPath);
            (ailiaMask, _, _, _) = RunFullPipeline(ailiaBackend, nchwInput);
        }
        catch (Exception ex)
        {
            Assert.Ignore($"ailia backend failed: {ex.Message}");
            return;
        }

        var (matchRate, diffCount, onlyOrt, onlyAilia) = CompareMasks(ortMask, ailiaMask);
        Console.WriteLine($"\n=== ORT vs ailia Comparison ===");
        Console.WriteLine($"Match rate: {matchRate:F4}%");
        Console.WriteLine($"Different: {diffCount}");
        Console.WriteLine($"  Only ORT: {onlyOrt}");
        Console.WriteLine($"  Only ailia: {onlyAilia}");

        SaveDiffPng(ortMask, ailiaMask, Path.Combine(CSHARP_OUTPUT_DIR, "diff_ort_vs_ailia.png"));

        Assert.That(matchRate, Is.GreaterThan(99.0),
            $"ORT vs ailia should match > 99%, got {matchRate:F4}%");
    }
#endif
}
