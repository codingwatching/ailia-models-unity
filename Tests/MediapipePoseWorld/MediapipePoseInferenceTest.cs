using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using ailiaSDK;

/// <summary>
/// End-to-end inference test for MediaPipe Pose World Landmarks.
/// Compares C# engine output with Python reference values.
/// </summary>
[TestFixture]
public class MediapipePoseInferenceTest
{
    private const string MODEL_DIR = "/tmp/ailia-models/pose_estimation_3d/mediapipe_pose_world_landmarks";
    private const string REFERENCE_DIR = "/tmp/mediapipe_pose_test_output";
    private const string NATIVE_LIB_DIR = "/tmp/ailia-csharp/ailia-csharp/ailia-csharp/ailia/ailia-sdk-unity/Runtime/Plugins/linux";

    private MediapipePoseWorldEngine engine;
    private float[,] anchors;

    [OneTimeSetUp]
    public void Setup()
    {
        // Set native library search path
        string currentPath = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "";
        if (!currentPath.Contains(NATIVE_LIB_DIR))
        {
            Environment.SetEnvironmentVariable("LD_LIBRARY_PATH",
                NATIVE_LIB_DIR + ":" + currentPath);
        }

        engine = new MediapipePoseWorldEngine();

        // Load anchors
        var anchorsHolder = new AiliaMediapipePoseWorldLandmarksAnchors();
        float[] anchorsFlat = ConvertDoubleArrayToFloatArray(anchorsHolder.anchors);
        anchors = new float[MediapipePoseWorldEngine.DETECTOR_TENSOR_COUNT, 4];
        for (int i = 0; i < anchorsFlat.Length; i++)
            anchors[i / 4, i % 4] = anchorsFlat[i];
    }

    private static float[] ConvertDoubleArrayToFloatArray(double[] arr)
    {
        float[] result = new float[arr.Length];
        for (int i = 0; i < arr.Length; i++)
            result[i] = (float)arr[i];
        return result;
    }

    // -------------------------------------------------------
    // Helper: Load PNG image as Color32[] (T2B order)
    // -------------------------------------------------------
    private (Color32[] pixels, int width, int height) LoadPngImage(string path)
    {
        using var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(path);
        int w = image.Width, h = image.Height;
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

    // -------------------------------------------------------
    // Helper: Load .npy file (float64 or float32)
    // -------------------------------------------------------
    private float[] LoadNpy(string path)
    {
        byte[] data = File.ReadAllBytes(path);
        // Parse numpy header
        // Magic: \x93NUMPY
        if (data[0] != 0x93 || data[1] != (byte)'N')
            throw new Exception("Invalid .npy file: " + path);

        int majorVersion = data[6];
        int headerLen;
        int headerStart;
        if (majorVersion == 1)
        {
            headerLen = BitConverter.ToUInt16(data, 8);
            headerStart = 10;
        }
        else
        {
            headerLen = (int)BitConverter.ToUInt32(data, 8);
            headerStart = 12;
        }

        string header = System.Text.Encoding.ASCII.GetString(data, headerStart, headerLen);
        int dataOffset = headerStart + headerLen;

        bool isFloat64 = header.Contains("<f8") || header.Contains("float64");
        bool isFloat32 = header.Contains("<f4") || header.Contains("float32");
        bool isInt64 = header.Contains("<i8") || header.Contains("int64");

        int remaining = data.Length - dataOffset;

        if (isFloat64)
        {
            int count = remaining / 8;
            float[] result = new float[count];
            for (int i = 0; i < count; i++)
                result[i] = (float)BitConverter.ToDouble(data, dataOffset + i * 8);
            return result;
        }
        else if (isFloat32)
        {
            int count = remaining / 4;
            float[] result = new float[count];
            Buffer.BlockCopy(data, dataOffset, result, 0, remaining);
            return result;
        }
        else if (isInt64)
        {
            int count = remaining / 8;
            float[] result = new float[count];
            for (int i = 0; i < count; i++)
                result[i] = (float)BitConverter.ToInt64(data, dataOffset + i * 8);
            return result;
        }
        else
        {
            throw new Exception($"Unsupported dtype in {path}: {header}");
        }
    }

    // -------------------------------------------------------
    // Check prerequisites
    // -------------------------------------------------------
    private void CheckModelsExist()
    {
        string detPath = Path.Combine(MODEL_DIR, "pose_detection.onnx");
        string estPath = Path.Combine(MODEL_DIR, "pose_landmark_heavy.onnx");

        if (!File.Exists(detPath) || !File.Exists(estPath))
            Assert.Ignore("Models not found. Download them first.");
    }

    private void CheckReferenceExists()
    {
        if (!File.Exists(Path.Combine(REFERENCE_DIR, "landmarks_image.npy")))
            Assert.Ignore("Reference data not found. Run generate_e2e_reference.py first.");
    }

    private void CheckNativeLibExists()
    {
        string libPath = Path.Combine(NATIVE_LIB_DIR, "libailia.so");
        if (!File.Exists(libPath))
            Assert.Ignore("ailia native library not found at " + libPath);
    }

    // -------------------------------------------------------
    // Test: Box decoding matches Python
    // Uses Python's raw detector output directly
    // -------------------------------------------------------
    [Test]
    public void TestBoxDecoding_MatchesPython()
    {
        CheckReferenceExists();

        // Load Python's raw detector output
        float[] rawBoxesFlat = LoadNpy(Path.Combine(REFERENCE_DIR, "det_raw_boxes.npy"));
        float[] rawScoresFlat = LoadNpy(Path.Combine(REFERENCE_DIR, "det_raw_scores.npy"));
        float[] padArr = LoadNpy(Path.Combine(REFERENCE_DIR, "det_pad.npy"));
        float[] refBox = LoadNpy(Path.Combine(REFERENCE_DIR, "detected_box.npy"));

        // rawBoxes shape: [1, 2254, 12] -> flatten to [2254*12]
        float[] rawBoxes = new float[MediapipePoseWorldEngine.DETECTOR_TENSOR_COUNT *
            MediapipePoseWorldEngine.DETECTOR_TENSOR_SIZE];
        Array.Copy(rawBoxesFlat, rawBoxes, rawBoxes.Length);

        // rawScores shape: [1, 2254, 1] -> flatten to [2254]
        float[] rawScores = new float[MediapipePoseWorldEngine.DETECTOR_TENSOR_COUNT];
        for (int i = 0; i < rawScores.Length; i++)
            rawScores[i] = rawScoresFlat[i];

        float padH = padArr[0];
        float padW = padArr[1];

        var boxes = engine.DecodeAndProcessBoxes(rawBoxes, rawScores, anchors, padH, padW);

        Assert.That(boxes.Count, Is.GreaterThan(0), "Should detect at least one box");

        var best = boxes[0];
        Console.WriteLine($"C# box: xMin={best.xMin:F6}, yMin={best.yMin:F6}, xMax={best.xMax:F6}, yMax={best.yMax:F6}");
        Console.WriteLine($"Python box: xMin={refBox[0]:F6}, yMin={refBox[1]:F6}, xMax={refBox[2]:F6}, yMax={refBox[3]:F6}");

        // Python box format: [xmin, ymin, xmax, ymax, kp0_x, kp0_y, kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y]
        float tol = 1e-3f;
        Assert.That(best.xMin, Is.EqualTo(refBox[0]).Within(tol), "xMin mismatch");
        Assert.That(best.yMin, Is.EqualTo(refBox[1]).Within(tol), "yMin mismatch");
        Assert.That(best.xMax, Is.EqualTo(refBox[2]).Within(tol), "xMax mismatch");
        Assert.That(best.yMax, Is.EqualTo(refBox[3]).Within(tol), "yMax mismatch");

        // Check keypoints
        for (int k = 0; k < 4; k++)
        {
            int refIdx = 4 + k * 2;
            Console.WriteLine($"  kp{k}: C#=({best.keypoints[k][0]:F6},{best.keypoints[k][1]:F6}) Python=({refBox[refIdx]:F6},{refBox[refIdx + 1]:F6})");
            Assert.That(best.keypoints[k][0], Is.EqualTo(refBox[refIdx]).Within(tol),
                $"keypoint {k} x mismatch");
            Assert.That(best.keypoints[k][1], Is.EqualTo(refBox[refIdx + 1]).Within(tol),
                $"keypoint {k} y mismatch");
        }
    }

    // -------------------------------------------------------
    // Test: ROI parameters match Python
    // -------------------------------------------------------
    [Test]
    public void TestROIParameters_MatchPython()
    {
        CheckReferenceExists();

        float[] refBox = LoadNpy(Path.Combine(REFERENCE_DIR, "detected_box.npy"));
        float[] roiParams = LoadNpy(Path.Combine(REFERENCE_DIR, "roi_params.npy"));
        float[] imgShape = LoadNpy(Path.Combine(REFERENCE_DIR, "image_shape.npy"));

        if (refBox.Length == 0)
        {
            Assert.Ignore("No pose detected in reference");
            return;
        }

        int imgHeight = (int)imgShape[0];
        int imgWidth = (int)imgShape[1];

        // Create a fake box from reference data
        var box = new MediapipePoseWorldEngine.DecodedBox
        {
            xMin = refBox[0],
            yMin = refBox[1],
            xMax = refBox[2],
            yMax = refBox[3],
            keypoints = new float[4][]
        };
        for (int k = 0; k < 4; k++)
            box.keypoints[k] = new float[] { refBox[4 + k * 2], refBox[5 + k * 2] };

        // Create minimal pixel array (we don't actually need the image data for ROI param test)
        var pixels = new Color32[imgWidth * imgHeight];

        engine.ExtractROI(pixels, imgWidth, imgHeight, box);

        float refXCenter = roiParams[0];
        float refYCenter = roiParams[1];
        float refBoxSize = roiParams[2];
        float refRotation = roiParams[3];

        Console.WriteLine($"C# ROI: center=({engine.RoiCenterX:F4},{engine.RoiCenterY:F4}), box_size={engine.RoiBoxSize:F4}, rotation={engine.RoiRotation:F6}");
        Console.WriteLine($"Python ROI: center=({refXCenter:F4},{refYCenter:F4}), box_size={refBoxSize:F4}, rotation={refRotation:F6}");

        float tol = 0.5f;
        Assert.That(engine.RoiCenterX, Is.EqualTo(refXCenter).Within(tol), "ROI center X mismatch");
        Assert.That(engine.RoiCenterY, Is.EqualTo(refYCenter).Within(tol), "ROI center Y mismatch");
        Assert.That(engine.RoiBoxSize, Is.EqualTo(refBoxSize).Within(tol), "ROI box size mismatch");
        Assert.That(engine.RoiRotation, Is.EqualTo(refRotation).Within(0.01f), "ROI rotation mismatch");
    }

    // -------------------------------------------------------
    // Test: Full E2E pipeline with ailia backend
    // -------------------------------------------------------
    [Test]
    public void TestFullPipeline_E2E()
    {
        CheckModelsExist();
        CheckReferenceExists();
        CheckNativeLibExists();

        string testImagePath = Path.Combine(REFERENCE_DIR, "test_image.png");
        if (!File.Exists(testImagePath))
            Assert.Ignore("Test image not found at " + testImagePath);

        // Load image
        var (pixels, imgW, imgH) = LoadPngImage(testImagePath);
        Console.WriteLine($"Loaded image: {imgW}x{imgH}");

        // Load reference landmarks
        float[] refImageLandmarks = LoadNpy(Path.Combine(REFERENCE_DIR, "landmarks_image.npy"));
        // Shape: [33, 5] -> 165 values

        string detPath = Path.Combine(MODEL_DIR, "pose_detection.onnx");
        string estPath = Path.Combine(MODEL_DIR, "pose_landmark_heavy.onnx");

        using (var backend = new AiliaMediapipePoseBackend())
        {
            try
            {
                backend.LoadModels(detPath, estPath);
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("Failed to open") || ex.Message.Contains("License"))
                    Assert.Ignore("ailia SDK license not available: " + ex.Message);
                throw;
            }

            var results = engine.RunFullPipeline(backend, pixels, imgW, imgH, anchors,
                worldCoordinate: false);

            Assert.That(results, Is.Not.Null, "Pipeline should detect a pose");
            Assert.That(results.Length, Is.EqualTo(33), "Should have 33 landmarks");

            Console.WriteLine("\n=== C# vs Python Image Landmarks ===");
            string[] names = { "Nose", "L_InnerEye", "L_Eye", "L_OuterEye", "R_InnerEye",
                "R_Eye", "R_OuterEye", "L_Ear", "R_Ear", "L_Mouth", "R_Mouth",
                "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist" };

            float totalDist = 0;
            int count = 0;
            for (int i = 0; i < Math.Min(results.Length, 17); i++)
            {
                float pyX = refImageLandmarks[i * 5];
                float pyY = refImageLandmarks[i * 5 + 1];
                float csX = results[i].X;
                float csY = results[i].Y;
                float dist = (float)Math.Sqrt(Math.Pow(csX - pyX, 2) + Math.Pow(csY - pyY, 2));
                totalDist += dist;
                count++;

                string name = i < names.Length ? names[i] : $"kp{i}";
                Console.WriteLine($"  {name}: C#=({csX:F6},{csY:F6}) Python=({pyX:F6},{pyY:F6}) dist={dist:F6}");
            }

            float avgDist = totalDist / count;
            Console.WriteLine($"\nAverage landmark distance: {avgDist:F6}");

            // Allow 5% tolerance for preprocessing differences (bilinear vs warpPerspective)
            Assert.That(avgDist, Is.LessThan(0.05f),
                $"Average landmark distance {avgDist:F6} exceeds tolerance");
        }
    }

    // -------------------------------------------------------
    // Test: Landmark decoding from Python's raw output
    // Bypasses preprocessing differences
    // -------------------------------------------------------
    [Test]
    public void TestLandmarkDecoding_MatchesPython()
    {
        CheckReferenceExists();

        float[] rawLandmarks = LoadNpy(Path.Combine(REFERENCE_DIR, "est_raw_landmarks.npy"));
        float[] refNormLandmarks = LoadNpy(Path.Combine(REFERENCE_DIR, "landmarks_normalized.npy"));

        if (rawLandmarks.Length == 0)
        {
            Assert.Ignore("No reference landmark data");
            return;
        }

        // Decode using engine (first 195 values = 33 landmarks * 5)
        var decoded = engine.DecodeLandmarks(rawLandmarks);

        Console.WriteLine("=== Landmark Decoding Comparison ===");
        float maxDiff = 0;
        for (int i = 0; i < 33; i++)
        {
            // Python normalized landmarks: x/256, y/256, z/256
            // Note: Python also applies sigmoid to visibility and presence separately,
            // but the normalized landmarks have visibility/presence as sigmoid values
            float pyX = refNormLandmarks[i * 5];
            float pyY = refNormLandmarks[i * 5 + 1];
            float csX = decoded[i].X;
            float csY = decoded[i].Y;

            float diffX = Math.Abs(csX - pyX);
            float diffY = Math.Abs(csY - pyY);
            maxDiff = Math.Max(maxDiff, Math.Max(diffX, diffY));

            if (i < 10)
                Console.WriteLine($"  lm{i}: C#=({csX:F6},{csY:F6}) Python=({pyX:F6},{pyY:F6})");
        }

        Console.WriteLine($"Max position diff: {maxDiff:F6}");

        // Note: Python applies heatmap refinement which changes x,y slightly.
        // Without heatmap refinement, the raw x/256 y/256 values should match closely.
        // With heatmap refinement, there will be small differences.
        // Use a generous tolerance to account for heatmap refinement.
        Assert.That(maxDiff, Is.LessThan(0.05f),
            "Landmark decoding differs too much (may need heatmap refinement)");
    }

    // -------------------------------------------------------
    // Test: Preprocessing normalization
    // -------------------------------------------------------
    [Test]
    public void TestPreprocessNormalization()
    {
        // Create a simple 2x2 test image
        Color32[] pixels = {
            new Color32(0, 0, 0, 255),
            new Color32(255, 255, 255, 255),
            new Color32(128, 128, 128, 255),
            new Color32(64, 192, 32, 255)
        };

        float padH, padW;
        float[] result = engine.PreprocessDetection(pixels, 2, 2, out padH, out padW);

        // For 2x2 input, boxSize = 2, so no padding needed
        Assert.That(padH, Is.EqualTo(0).Within(1e-6f));
        Assert.That(padW, Is.EqualTo(0).Within(1e-6f));

        // Check normalization range: should be [-1, 1]
        bool hasNegative = false;
        for (int i = 0; i < result.Length; i++)
        {
            Assert.That(result[i], Is.GreaterThanOrEqualTo(-1.01f), $"Value below -1: {result[i]}");
            Assert.That(result[i], Is.LessThanOrEqualTo(1.01f), $"Value above 1: {result[i]}");
            if (result[i] < 0) hasNegative = true;
        }
        Assert.That(hasNegative, Is.True, "Should have negative values for [-1,1] normalization");
    }

    // -------------------------------------------------------
    // Test: Sigmoid correctness
    // -------------------------------------------------------
    [Test]
    public void TestSigmoid()
    {
        Assert.That(engine.Sigmoid(0), Is.EqualTo(0.5f).Within(1e-6f));
        Assert.That(engine.Sigmoid(100), Is.EqualTo(1.0f).Within(1e-6f));
        Assert.That(engine.Sigmoid(-100), Is.EqualTo(0.0f).Within(1e-6f));

        // sigmoid(1) = 1 / (1 + exp(-1)) ≈ 0.7310586
        Assert.That(engine.Sigmoid(1), Is.EqualTo(0.7310586f).Within(1e-5f));
    }

    // -------------------------------------------------------
    // Helper: Convert Python NCHW tensor to C# HWC for comparison
    // -------------------------------------------------------
    private float[] NchwToHwc(float[] nchw, int channels, int height, int width)
    {
        float[] hwc = new float[nchw.Length];
        int offset = 0; // skip batch dim if present
        if (nchw.Length > channels * height * width)
            offset = 0; // npy already flattened from [1,C,H,W]

        for (int c = 0; c < channels; c++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    hwc[(y * width + x) * channels + c] = nchw[c * height * width + y * width + x];
        return hwc;
    }

    // -------------------------------------------------------
    // Test: Detection preprocessing image matches Python
    // -------------------------------------------------------
    [Test]
    public void TestDetectionPreprocessing_MatchesPython()
    {
        CheckReferenceExists();

        string detInputPath = Path.Combine(REFERENCE_DIR, "det_input.npy");
        if (!File.Exists(detInputPath))
            Assert.Ignore("det_input.npy not found");

        string testImagePath = Path.Combine(REFERENCE_DIR, "test_image.png");
        if (!File.Exists(testImagePath))
            Assert.Ignore("test_image.png not found");

        // Load Python's NCHW detection input [1,3,224,224]
        float[] pyDetInput = LoadNpy(detInputPath);
        float[] pyHwc = NchwToHwc(pyDetInput, 3, 224, 224);

        // Run C# preprocessing
        var (pixels, imgW, imgH) = LoadPngImage(testImagePath);
        float padH, padW;
        float[] csDetInput = engine.PreprocessDetection(pixels, imgW, imgH, out padH, out padW);

        Assert.That(csDetInput.Length, Is.EqualTo(pyHwc.Length),
            $"Length mismatch: C#={csDetInput.Length}, Python={pyHwc.Length}");

        // Compare pixel values
        float maxDiff = 0;
        double sumDiff = 0;
        int totalPixels = 224 * 224;
        for (int i = 0; i < csDetInput.Length; i++)
        {
            float diff = Math.Abs(csDetInput[i] - pyHwc[i]);
            maxDiff = Math.Max(maxDiff, diff);
            sumDiff += diff;
        }

        float avgDiff = (float)(sumDiff / csDetInput.Length);
        Console.WriteLine($"Detection preprocessing comparison:");
        Console.WriteLine($"  Max pixel diff: {maxDiff:F6}");
        Console.WriteLine($"  Avg pixel diff: {avgDiff:F6}");
        Console.WriteLine($"  Total pixels: {totalPixels} ({csDetInput.Length} values)");

        // Print some sample values
        for (int i = 0; i < 15; i++)
        {
            Console.WriteLine($"  [{i}] C#={csDetInput[i]:F6} Python={pyHwc[i]:F6} diff={Math.Abs(csDetInput[i] - pyHwc[i]):F6}");
        }

        // Tolerance: bilinear interpolation differences should be small
        Assert.That(maxDiff, Is.LessThan(0.1f),
            $"Max pixel diff {maxDiff:F6} exceeds tolerance");
        Assert.That(avgDiff, Is.LessThan(0.02f),
            $"Avg pixel diff {avgDiff:F6} exceeds tolerance");
    }

    // -------------------------------------------------------
    // Test: ROI extraction image matches Python
    // -------------------------------------------------------
    [Test]
    public void TestROIExtraction_MatchesPython()
    {
        CheckReferenceExists();

        string lmkInputPath = Path.Combine(REFERENCE_DIR, "lmk_input.npy");
        if (!File.Exists(lmkInputPath))
            Assert.Ignore("lmk_input.npy not found");

        string testImagePath = Path.Combine(REFERENCE_DIR, "test_image.png");
        if (!File.Exists(testImagePath))
            Assert.Ignore("test_image.png not found");

        float[] refBox = LoadNpy(Path.Combine(REFERENCE_DIR, "detected_box.npy"));
        if (refBox.Length == 0)
        {
            Assert.Ignore("No detected box in reference");
            return;
        }

        // Load Python's NCHW landmark input [1,3,256,256]
        float[] pyLmkInput = LoadNpy(lmkInputPath);
        float[] pyHwc = NchwToHwc(pyLmkInput, 3, 256, 256);

        // Run C# ROI extraction with the same detected box
        var (pixels, imgW, imgH) = LoadPngImage(testImagePath);

        var box = new MediapipePoseWorldEngine.DecodedBox
        {
            xMin = refBox[0],
            yMin = refBox[1],
            xMax = refBox[2],
            yMax = refBox[3],
            keypoints = new float[4][]
        };
        for (int k = 0; k < 4; k++)
            box.keypoints[k] = new float[] { refBox[4 + k * 2], refBox[5 + k * 2] };

        var (csLmkInput, roiW, roiH) = engine.ExtractROI(pixels, imgW, imgH, box);

        Assert.That(csLmkInput.Length, Is.EqualTo(pyHwc.Length),
            $"Length mismatch: C#={csLmkInput.Length}, Python={pyHwc.Length}");

        // Compare pixel values
        float maxDiff = 0;
        double sumDiff = 0;
        int totalPixels = 256 * 256;
        for (int i = 0; i < csLmkInput.Length; i++)
        {
            float diff = Math.Abs(csLmkInput[i] - pyHwc[i]);
            maxDiff = Math.Max(maxDiff, diff);
            sumDiff += diff;
        }

        float avgDiff = (float)(sumDiff / csLmkInput.Length);
        Console.WriteLine($"ROI extraction comparison:");
        Console.WriteLine($"  Max pixel diff: {maxDiff:F6}");
        Console.WriteLine($"  Avg pixel diff: {avgDiff:F6}");
        Console.WriteLine($"  Total pixels: {totalPixels} ({csLmkInput.Length} values)");

        // Print sample values at center region
        int cx = 128, cy = 128;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int idx = ((cy + dy) * 256 + (cx + dx)) * 3;
                Console.WriteLine($"  [{cx+dx},{cy+dy}] C#=({csLmkInput[idx]:F4},{csLmkInput[idx+1]:F4},{csLmkInput[idx+2]:F4}) Python=({pyHwc[idx]:F4},{pyHwc[idx+1]:F4},{pyHwc[idx+2]:F4})");
            }
        }

        Assert.That(maxDiff, Is.LessThan(0.1f),
            $"Max pixel diff {maxDiff:F6} exceeds tolerance");
        Assert.That(avgDiff, Is.LessThan(0.02f),
            $"Avg pixel diff {avgDiff:F6} exceeds tolerance");
    }

    // -------------------------------------------------------
    // Test: World landmark decoding from Python's raw output
    // Verifies DecodeWorldLandmarks + GetWorldResult match Python
    // -------------------------------------------------------
    [Test]
    public void TestWorldLandmarkDecoding_MatchesPython()
    {
        CheckReferenceExists();

        float[] rawWorldLandmarks = LoadNpy(Path.Combine(REFERENCE_DIR, "est_world_landmarks.npy"));
        float[] refWorldTransformed = LoadNpy(Path.Combine(REFERENCE_DIR, "world_landmarks_transformed.npy"));
        float[] roiParams = LoadNpy(Path.Combine(REFERENCE_DIR, "roi_params.npy"));

        if (rawWorldLandmarks.Length == 0 || refWorldTransformed.Length == 0)
        {
            Assert.Ignore("No world landmark reference data");
            return;
        }

        Assert.That(rawWorldLandmarks.Length, Is.EqualTo(MediapipePoseWorldEngine.WORLD_LANDMARK_TENSOR_SIZE),
            $"World landmark tensor should be {MediapipePoseWorldEngine.WORLD_LANDMARK_TENSOR_SIZE} but got {rawWorldLandmarks.Length}");

        // Set ROI parameters from Python reference (needed for rotation transform)
        // roiParams: [x_center, y_center, box_size, rotation, ...]
        // We need to call ExtractROI or set them manually. Use a dummy image to set ROI state.
        float[] rawLandmarks = LoadNpy(Path.Combine(REFERENCE_DIR, "est_raw_landmarks.npy"));
        engine.DecodeLandmarks(rawLandmarks);

        // Set ROI rotation by calling ExtractROI with the reference detection box
        float[] detBox = LoadNpy(Path.Combine(REFERENCE_DIR, "detected_box.npy"));
        float[] imageShape = LoadNpy(Path.Combine(REFERENCE_DIR, "image_shape.npy"));
        int imgH = (int)imageShape[0];
        int imgW = (int)imageShape[1];

        // Reconstruct detection box
        var box = new MediapipePoseWorldEngine.DecodedBox
        {
            yMin = detBox[0], xMin = detBox[1], yMax = detBox[2], xMax = detBox[3],
            score = 1.0f,
            keypoints = new float[][] {
                new float[] { detBox[4], detBox[5] },
                new float[] { detBox[6], detBox[7] },
                new float[] { detBox[8], detBox[9] },
                new float[] { detBox[10], detBox[11] }
            }
        };

        // Call ExtractROI with dummy pixels to set ROI parameters
        var dummyPixels = new Color32[1];
        engine.ExtractROI(dummyPixels, imgW, imgH, box);

        // Verify ROI rotation matches Python
        float pyRotation = roiParams[3];
        Console.WriteLine($"ROI rotation: C#={engine.RoiRotation:F6} Python={pyRotation:F6}");
        Assert.That(engine.RoiRotation, Is.EqualTo(pyRotation).Within(1e-4f), "ROI rotation should match Python");

        // Decode world landmarks
        engine.DecodeWorldLandmarks(rawWorldLandmarks);

        // Verify raw world landmarks (before rotation)
        Console.WriteLine("\n=== Raw World Landmarks (first 5) ===");
        for (int i = 0; i < 5; i++)
        {
            float pyX = rawWorldLandmarks[i * 3];
            float pyY = rawWorldLandmarks[i * 3 + 1];
            float pyZ = rawWorldLandmarks[i * 3 + 2];
            Console.WriteLine($"  wl{i}: decoded=({engine.WorldLandmarks[i].X:F6},{engine.WorldLandmarks[i].Y:F6},{engine.WorldLandmarks[i].Z:F6}) raw=({pyX:F6},{pyY:F6},{pyZ:F6})");
            Assert.That(engine.WorldLandmarks[i].X, Is.EqualTo(pyX).Within(1e-6f), $"World landmark {i} X");
            Assert.That(engine.WorldLandmarks[i].Y, Is.EqualTo(pyY).Within(1e-6f), $"World landmark {i} Y");
            Assert.That(engine.WorldLandmarks[i].Z, Is.EqualTo(pyZ).Within(1e-6f), $"World landmark {i} Z");
        }

        // Get world result (applies rotation transform + maps to 19 keypoints)
        var worldResult = engine.GetWorldResult();
        Assert.That(worldResult, Is.Not.Null, "GetWorldResult should not return null");
        Assert.That(worldResult.Length, Is.EqualTo(19), "Should have 19 keypoints");

        // Compare first 17 keypoints with Python's transformed world landmarks
        // C# negates Y for Unity (Y-up), so compare with negated Python Y
        Console.WriteLine("\n=== World Landmarks Transformed (17 body keypoints) ===");
        float maxDiff = 0;
        string[] kpNames = { "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
            "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
            "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle" };
        for (int i = 0; i < 17; i++)
        {
            int mpIdx = MediapipePoseWorldEngine.KEYPOINT_MAPPING[i];
            float pyX = refWorldTransformed[mpIdx * 3];
            float pyY = -refWorldTransformed[mpIdx * 3 + 1]; // Negated: Python Y-down -> Unity Y-up
            float pyZ = refWorldTransformed[mpIdx * 3 + 2];
            float csX = worldResult[i].X;
            float csY = worldResult[i].Y;
            float csZ = worldResult[i].Z;
            float diff = Math.Max(Math.Abs(csX - pyX), Math.Max(Math.Abs(csY - pyY), Math.Abs(csZ - pyZ)));
            maxDiff = Math.Max(maxDiff, diff);

            Console.WriteLine($"  {kpNames[i]}: C#=({csX:F6},{csY:F6},{csZ:F6}) Python=({pyX:F6},{pyY:F6},{pyZ:F6}) diff={diff:F6}");
        }

        Console.WriteLine($"\nMax world landmark diff: {maxDiff:F6}");
        Assert.That(maxDiff, Is.LessThan(1e-4f),
            $"World landmark max diff {maxDiff:F6} exceeds tolerance");
    }
}
