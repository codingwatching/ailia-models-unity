/* SAM2 ONNX Inference Comparison Test */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * All tests use PNG input (lossless) to ensure zero decoder difference
 * between PIL (Python) and ImageSharp (C#).
 *
 * Test image: truck.png (1800x1200, lossless PNG)
 * Click point: (500, 375), label=1 (foreground)
 * Models: segment-anything-2 (Hiera-L)
 *
 * Tests:
 *   1. Preprocess_PngZeroError: C# preprocessing matches Python exactly (error=0)
 *   2. FullInference_PngEndToEnd: C# preprocess + ONNX inference matches Python mask
 */

using NUnit.Framework;
using Sam2MaskTests;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.IO;
using System.Linq;

[TestFixture]
public class Sam2OnnxInferenceTest
{
    private Sam2ImageMaskLogic logic = null!;

    private const string MODEL_DIR = "/tmp/sam2_models";
    private const string PNG_IMAGE_PATH = "/tmp/sam2_test_output/truck.png";
    private const string PYTHON_OUTPUT_DIR = "/tmp/sam2_test_output";
    private const string CSHARP_OUTPUT_DIR = "/tmp/sam2_csharp_output";

    // Same point as Python test
    private const int CLICK_X = 500;
    private const int CLICK_Y = 375;
    private const int POINT_LABEL = 1;

    // Image dimensions (truck.png)
    private const int IMG_W = 1800;
    private const int IMG_H = 1200;

    [SetUp]
    public void SetUp()
    {
        logic = new Sam2ImageMaskLogic();
        Directory.CreateDirectory(CSHARP_OUTPUT_DIR);
    }

    private bool ModelsExist()
    {
        return File.Exists(Path.Combine(MODEL_DIR, "image_encoder_hiera_l.onnx"))
            && File.Exists(Path.Combine(MODEL_DIR, "mask_decoder_hiera_l.onnx"))
            && File.Exists(Path.Combine(MODEL_DIR, "prompt_encoder_hiera_l.onnx"));
    }

    // =======================================================
    // Load PNG image as Color32 array
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
    // Load Python .npy file (bool array)
    // =======================================================
    private bool[,] LoadNpyBool(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        int headerLen = BitConverter.ToUInt16(raw, 8);
        string header = System.Text.Encoding.ASCII.GetString(raw, 10, headerLen);

        int shapeStart = header.IndexOf("'shape': (") + 10;
        int shapeEnd = header.IndexOf(")", shapeStart);
        string shapeStr = header.Substring(shapeStart, shapeEnd - shapeStart);
        var dims = shapeStr.Split(',').Select(s => int.Parse(s.Trim())).ToArray();

        int h = dims[0], w = dims[1];
        int dataOffset = 10 + headerLen;
        bool[,] result = new bool[h, w];

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                result[y, x] = raw[dataOffset + y * w + x] != 0;

        return result;
    }

    // =======================================================
    // Load Python .npy file (float32 array, 1D)
    // =======================================================
    private float[] LoadNpyFloat1D(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        int headerLen = BitConverter.ToUInt16(raw, 8);
        int dataOffset = 10 + headerLen;
        int count = (raw.Length - dataOffset) / 4;
        float[] result = new float[count];
        Buffer.BlockCopy(raw, dataOffset, result, 0, count * 4);
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
    // Save diff image as PNG
    // =======================================================
    private void SaveDiffPng(bool[,] csharpMask, bool[,] pythonMask, string path)
    {
        int h = csharpMask.GetLength(0), w = csharpMask.GetLength(1);
        using var img = new Image<Rgb24>(w, h);
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    bool cs = csharpMask[y, x];
                    bool py = pythonMask[y, x];
                    if (cs && py)
                        row[x] = new Rgb24(255, 255, 255);
                    else if (cs && !py)
                        row[x] = new Rgb24(255, 0, 0);
                    else if (!cs && py)
                        row[x] = new Rgb24(0, 0, 255);
                    else
                        row[x] = new Rgb24(0, 0, 0);
                }
            }
        });
        img.Save(path);
    }

    // =======================================================
    // Run encoder + backbone feature preparation
    // =======================================================
    private (float[] imageEmbFlat, float[] highResFeat0, float[] highResFeat1)
        RunEncoder(float[] nchwInput)
    {
        var encoderSession = new InferenceSession(
            Path.Combine(MODEL_DIR, "image_encoder_hiera_l.onnx"));

        var encInputMeta = encoderSession.InputMetadata;
        var encTensor = new DenseTensor<float>(nchwInput, new[] { 1, 3, 1024, 1024 });
        var encInputs = new[] { NamedOnnxValue.CreateFromTensor(
            encInputMeta.Keys.First(), encTensor) };

        using var encResults = encoderSession.Run(encInputs);

        var encMap = encResults.ToDictionary(r => r.Name, r => r.AsTensor<float>().ToArray());
        var encShapes = encResults.ToDictionary(r => r.Name,
            r => r.AsTensor<float>().Dimensions.ToArray());

        Console.WriteLine("Encoder outputs:");
        foreach (var kv in encShapes)
            Console.WriteLine($"  {kv.Key}: [{string.Join(",", kv.Value)}]");

        var fpn0Shape = encShapes["backbone_fpn_0"];
        var fpn1Shape = encShapes["backbone_fpn_1"];
        var fpn2Shape = encShapes["backbone_fpn_2"];

        float[,,,] fpn0_4d = logic.ReshapeTo4D(encMap["backbone_fpn_0"], fpn0Shape[0], fpn0Shape[1], fpn0Shape[2], fpn0Shape[3]);
        float[,,,] fpn1_4d = logic.ReshapeTo4D(encMap["backbone_fpn_1"], fpn1Shape[0], fpn1Shape[1], fpn1Shape[2], fpn1Shape[3]);
        float[,,,] fpn2_4d = logic.ReshapeTo4D(encMap["backbone_fpn_2"], fpn2Shape[0], fpn2Shape[1], fpn2Shape[2], fpn2Shape[3]);

        float[][,,,] backboneFpn = new[] { fpn0_4d, fpn1_4d, fpn2_4d };
        float[][,,] visionFeats = logic.PrepareBackboneFeatures(backboneFpn);

        int hidden_dim = 256;
        float[,,] noMemEmbed = new float[1, 1, hidden_dim];
        float[,,] lastFeat = visionFeats[^1];
        float[,,] updatedFeat = logic.BroadcastAdd3D(lastFeat, noMemEmbed);
        visionFeats[^1] = updatedFeat;

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

        return (imageEmbFlat, highResFeat0, highResFeat1);
    }

    // =======================================================
    // Run prompt encoder + mask decoder
    // =======================================================
    private (bool[,] bestMask, int bestIdx, float bestScore, bool[][,] allMasks, float[] iouPred)
        RunDecoder(float[] imageEmbFlat, float[] highResFeat0, float[] highResFeat1)
    {
        var promptSession = new InferenceSession(
            Path.Combine(MODEL_DIR, "prompt_encoder_hiera_l.onnx"));

        float[,] rawCoords = new float[,] { { CLICK_X, CLICK_Y } };
        float[,] scaledCoords = logic.ApplyCoordinateScaling(rawCoords, IMG_H, IMG_W);
        Console.WriteLine($"Scaled coords: ({scaledCoords[0,0]:F4}, {scaledCoords[0,1]:F4})");

        float[] flatCoords = { scaledCoords[0, 0], scaledCoords[0, 1] };
        int[] labels = { POINT_LABEL };
        float[] maskInputDummy = new float[1 * 256 * 256];
        int[] masksEnable = { 0 };

        var promptInputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("coords",
                new DenseTensor<float>(flatCoords, new[] { 1, 1, 2 })),
            NamedOnnxValue.CreateFromTensor("labels",
                new DenseTensor<int>(labels, new[] { 1, 1 })),
            NamedOnnxValue.CreateFromTensor("masks",
                new DenseTensor<float>(maskInputDummy, new[] { 1, 256, 256 })),
            NamedOnnxValue.CreateFromTensor("masks_enable",
                new DenseTensor<int>(masksEnable, new[] { 1 })),
        };

        using var promptResults = promptSession.Run(promptInputs);
        var promptMap = promptResults.ToDictionary(r => r.Name, r => r);

        float[] sparseEmb = promptMap["sparse_embeddings"].AsTensor<float>().ToArray();
        float[] denseEmb = promptMap["dense_embeddings"].AsTensor<float>().ToArray();
        float[] densePe = promptMap["dense_pe"].AsTensor<float>().ToArray();

        var sparseShape = promptMap["sparse_embeddings"].AsTensor<float>().Dimensions.ToArray();
        var denseEmbShape = promptMap["dense_embeddings"].AsTensor<float>().Dimensions.ToArray();
        var densePeShape = promptMap["dense_pe"].AsTensor<float>().Dimensions.ToArray();

        var decoderSession = new InferenceSession(
            Path.Combine(MODEL_DIR, "mask_decoder_hiera_l.onnx"));

        var decoderInputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("image_embeddings",
                new DenseTensor<float>(imageEmbFlat, new[] { 1, 256, 64, 64 })),
            NamedOnnxValue.CreateFromTensor("image_pe",
                new DenseTensor<float>(densePe, densePeShape)),
            NamedOnnxValue.CreateFromTensor("sparse_prompt_embeddings",
                new DenseTensor<float>(sparseEmb, sparseShape)),
            NamedOnnxValue.CreateFromTensor("dense_prompt_embeddings",
                new DenseTensor<float>(denseEmb, denseEmbShape)),
            NamedOnnxValue.CreateFromTensor("high_res_features1",
                new DenseTensor<float>(highResFeat0, new[] { 1, 32, 256, 256 })),
            NamedOnnxValue.CreateFromTensor("high_res_features2",
                new DenseTensor<float>(highResFeat1, new[] { 1, 64, 128, 128 })),
        };

        using var decResults = decoderSession.Run(decoderInputs);
        var decMap = decResults.ToDictionary(r => r.Name, r => r);

        float[] masksRaw = decMap["masks"].AsTensor<float>().ToArray();
        float[] iouPred = decMap["iou_pred"].AsTensor<float>().ToArray();
        var masksShape = decMap["masks"].AsTensor<float>().Dimensions.ToArray();

        Console.WriteLine($"Raw masks shape: [{string.Join(",", masksShape)}]");
        Console.WriteLine($"IoU predictions: [{string.Join(", ", iouPred.Select(v => $"{v:F4}"))}]");

        float[,,,] masks4d = logic.ReshapeTo4D(masksRaw,
            masksShape[0], masksShape[1], masksShape[2], masksShape[3]);
        float[,,,] resizedMasks = logic.PostprocessMasks(masks4d, IMG_H, IMG_W);
        bool[][,] boolMasks = logic.ConvertToBoolMasks(resizedMasks, 0.0f);

        int bestIdx = 0;
        float maxScore = float.MinValue;
        for (int i = 0; i < iouPred.Length; i++)
            if (iouPred[i] > maxScore) { maxScore = iouPred[i]; bestIdx = i; }

        return (boolMasks[bestIdx], bestIdx, maxScore, boolMasks, iouPred);
    }

    // =======================================================
    // Compare two masks and return match rate
    // =======================================================
    private (double matchRate, int diffCount, int onlyCsharp, int onlyPython)
        CompareMasks(bool[,] csharpMask, bool[,] pythonMask)
    {
        int h = csharpMask.GetLength(0), w = csharpMask.GetLength(1);
        int matchCount = 0, diffCount = 0, onlyCsharp = 0, onlyPython = 0;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (csharpMask[y, x] == pythonMask[y, x])
                    matchCount++;
                else
                {
                    diffCount++;
                    if (csharpMask[y, x]) onlyCsharp++;
                    else onlyPython++;
                }
            }
        }

        double matchRate = 100.0 * matchCount / (h * w);
        return (matchRate, diffCount, onlyCsharp, onlyPython);
    }

    // =======================================================
    // 1. Preprocess PNG: verify zero error vs Python
    // =======================================================
    [Test]
    public void Preprocess_PngZeroError()
    {
        string pngPreprocessPath = Path.Combine(PYTHON_OUTPUT_DIR, "preprocess_output_png.npy");
        if (!File.Exists(pngPreprocessPath))
            Assert.Ignore("Python PNG preprocess output not found.");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("PNG test image not found.");

        var (pixels, imgW, imgH) = LoadPngImage(PNG_IMAGE_PATH);
        Assert.That(imgW, Is.EqualTo(IMG_W), "Image width");
        Assert.That(imgH, Is.EqualTo(IMG_H), "Image height");

        float[,,,] csharpResult = logic.PreprocessImage(pixels, imgW, imgH, 1024);
        float[] csharpFlat = logic.Flatten4D(csharpResult);
        float[] pythonFlat = LoadNpyFloat1D(pngPreprocessPath);

        Assert.That(csharpFlat.Length, Is.EqualTo(pythonFlat.Length),
            $"Tensor size: C#={csharpFlat.Length} vs Python={pythonFlat.Length}");

        // PNG is lossless: both decoders produce identical pixels.
        // Only float32 precision differences expected (< 1e-6).
        double maxErr = 0, sumErr = 0;
        int mismatchCount = 0;
        for (int i = 0; i < csharpFlat.Length; i++)
        {
            double err = Math.Abs(csharpFlat[i] - pythonFlat[i]);
            maxErr = Math.Max(maxErr, err);
            sumErr += err;
            if (err > 1e-5) mismatchCount++;
        }
        double avgErr = sumErr / csharpFlat.Length;

        Console.WriteLine($"PNG Preprocess comparison ({csharpFlat.Length} values):");
        Console.WriteLine($"  Max error: {maxErr:E6}");
        Console.WriteLine($"  Avg error: {avgErr:E6}");
        Console.WriteLine($"  Values with error > 1e-5: {mismatchCount}");

        // With PNG input, only float32 rounding differences remain
        Assert.That(maxErr, Is.LessThan(1e-4),
            $"PNG preprocess max error should be near zero: {maxErr:E6}");
        Assert.That(avgErr, Is.LessThan(1e-6),
            $"PNG preprocess avg error should be near zero: {avgErr:E6}");
        Assert.That(mismatchCount, Is.EqualTo(0),
            $"No values should differ by more than 1e-5 with PNG input");
    }

    // =======================================================
    // 2. End-to-end: C# preprocess PNG + ONNX -> mask == Python
    // =======================================================
    [Test]
    public void FullInference_PngEndToEnd()
    {
        if (!ModelsExist())
            Assert.Ignore("ONNX models not found in " + MODEL_DIR);
        string pngMaskPath = Path.Combine(PYTHON_OUTPUT_DIR, "best_mask_bool_png.npy");
        string pngMetaPath = Path.Combine(PYTHON_OUTPUT_DIR, "metadata_png.json");
        if (!File.Exists(pngMaskPath) || !File.Exists(pngMetaPath))
            Assert.Ignore("Python PNG reference output not found.");
        if (!File.Exists(PNG_IMAGE_PATH))
            Assert.Ignore("PNG test image not found.");

        Console.WriteLine("=== C# SAM2 End-to-End (PNG preprocess + ONNX inference) ===");

        // Step 1: C# preprocesses the PNG image
        var (pixels, imgW, imgH) = LoadPngImage(PNG_IMAGE_PATH);
        float[,,,] preprocessed = logic.PreprocessImage(pixels, imgW, imgH, 1024);
        float[] nchwInput = logic.Flatten4D(preprocessed);
        Console.WriteLine($"C# preprocessed tensor: {nchwInput.Length} values, range=[{nchwInput.Min():F4}, {nchwInput.Max():F4}]");

        // Step 2: Run encoder
        var (imageEmbFlat, highResFeat0, highResFeat1) = RunEncoder(nchwInput);

        // Step 3: Run decoder
        var (bestMask, bestIdx, bestScore, allMasks, iouPred) =
            RunDecoder(imageEmbFlat, highResFeat0, highResFeat1);

        int trueCount = 0;
        for (int y = 0; y < IMG_H; y++)
            for (int x = 0; x < IMG_W; x++)
                if (bestMask[y, x]) trueCount++;

        Console.WriteLine($"Best mask: index={bestIdx}, score={bestScore:F4}");
        Console.WriteLine($"True pixels: {trueCount}/{IMG_W * IMG_H} ({100.0 * trueCount / (IMG_W * IMG_H):F1}%)");

        SaveMaskPng(bestMask, Path.Combine(CSHARP_OUTPUT_DIR, "csharp_mask_png_e2e.png"));

        // Step 4: Compare with Python reference
        bool[,] pythonMask = LoadNpyBool(pngMaskPath);
        Assert.That(bestMask.GetLength(0), Is.EqualTo(pythonMask.GetLength(0)), "Mask height");
        Assert.That(bestMask.GetLength(1), Is.EqualTo(pythonMask.GetLength(1)), "Mask width");

        var (matchRate, diffCount, onlyCsharp, onlyPython) = CompareMasks(bestMask, pythonMask);

        Console.WriteLine($"\n=== Mask Comparison (PNG End-to-End) ===");
        Console.WriteLine($"Total pixels: {IMG_W * IMG_H}");
        Console.WriteLine($"Matching: {IMG_W * IMG_H - diffCount} ({matchRate:F4}%)");
        Console.WriteLine($"Different: {diffCount} ({100.0 * diffCount / (IMG_W * IMG_H):F4}%)");
        Console.WriteLine($"  Only in C#: {onlyCsharp}");
        Console.WriteLine($"  Only in Python: {onlyPython}");

        SaveDiffPng(bestMask, pythonMask, Path.Combine(CSHARP_OUTPUT_DIR, "diff_mask_png_e2e.png"));

        // Check best mask index
        string pyMetaJson = File.ReadAllText(pngMetaPath);
        int pyBestIdx = -1;
        foreach (var line in pyMetaJson.Split('\n'))
        {
            if (line.Contains("\"best_mask_index\""))
            {
                var val = line.Split(':')[1].Trim().TrimEnd(',');
                pyBestIdx = int.Parse(val);
                break;
            }
        }
        Assert.That(bestIdx, Is.EqualTo(pyBestIdx),
            $"Best mask index: C#={bestIdx} vs Python={pyBestIdx}");

        // With identical PNG input, ONNX inference should produce identical masks
        // (same preprocessing -> same encoder input -> same output)
        Assert.That(matchRate, Is.GreaterThan(99.0),
            $"End-to-end mask match rate should be > 99%, got {matchRate:F4}%");
    }
}
