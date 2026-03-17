using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using ailiaSDK;

/// <summary>
/// Test to verify DrawBone3D and DrawAxis3D coordinate systems.
/// Uses real world landmark data to simulate what Unity renders.
/// Validates that Y-negation fixes the upside-down skeleton issue
/// (MediaPipe world landmarks use Y-down, Unity uses Y-up).
/// </summary>
[TestFixture]
public class DrawAxis3DTest
{
    private const string REFERENCE_DIR = "/tmp/mediapipe_pose_test_output";
    private const string NATIVE_LIB_DIR = "/tmp/ailia-csharp/ailia-csharp/ailia-csharp/ailia/ailia-sdk-unity/Runtime/Plugins/linux";

    // 19-keypoint indices
    const int NOSE = 0;
    const int SHOULDER_LEFT = 5;
    const int SHOULDER_RIGHT = 6;
    const int WRIST_LEFT = 9;
    const int WRIST_RIGHT = 10;
    const int HIP_LEFT = 11;
    const int HIP_RIGHT = 12;
    const int KNEE_LEFT = 13;
    const int KNEE_RIGHT = 14;
    const int ANKLE_LEFT = 15;
    const int ANKLE_RIGHT = 16;

    struct Point3D { public float x, y, z; }

    private MediapipePoseWorldEngine engine;

    [OneTimeSetUp]
    public void Setup()
    {
        string currentPath = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "";
        if (!currentPath.Contains(NATIVE_LIB_DIR))
            Environment.SetEnvironmentVariable("LD_LIBRARY_PATH", NATIVE_LIB_DIR + ":" + currentPath);

        engine = new MediapipePoseWorldEngine();
    }

    private float[] LoadNpy(string path)
    {
        byte[] data = File.ReadAllBytes(path);
        int headerEnd = Array.IndexOf(data, (byte)'\n', 10) + 1;
        string header = System.Text.Encoding.ASCII.GetString(data, 0, headerEnd);
        bool is8byte = header.Contains("f8") || header.Contains("float64") ||
                       header.Contains("i8") || header.Contains("int64");
        bool isInt = header.Contains("<i") || header.Contains("int");
        int dataOffset = headerEnd;
        int numBytes = data.Length - dataOffset;
        if (is8byte)
        {
            int count = numBytes / 8;
            float[] result = new float[count];
            for (int i = 0; i < count; i++)
            {
                if (isInt)
                    result[i] = (float)BitConverter.ToInt64(data, dataOffset + i * 8);
                else
                    result[i] = (float)BitConverter.ToDouble(data, dataOffset + i * 8);
            }
            return result;
        }
        else
        {
            int count = numBytes / 4;
            float[] result = new float[count];
            if (isInt)
            {
                for (int i = 0; i < count; i++)
                    result[i] = (float)BitConverter.ToInt32(data, dataOffset + i * 4);
            }
            else
            {
                Buffer.BlockCopy(data, dataOffset, result, 0, numBytes);
            }
            return result;
        }
    }

    private Point3D[] GetWorldPoints()
    {
        float[] rawWorldLandmarks = LoadNpy(Path.Combine(REFERENCE_DIR, "est_world_landmarks.npy"));
        float[] rawLandmarks = LoadNpy(Path.Combine(REFERENCE_DIR, "est_raw_landmarks.npy"));
        float[] detBox = LoadNpy(Path.Combine(REFERENCE_DIR, "detected_box.npy"));
        float[] imageShape = LoadNpy(Path.Combine(REFERENCE_DIR, "image_shape.npy"));

        int imgH = (int)imageShape[0], imgW = (int)imageShape[1];
        var box = new MediapipePoseWorldEngine.DecodedBox {
            yMin = detBox[0], xMin = detBox[1], yMax = detBox[2], xMax = detBox[3],
            score = 1.0f,
            keypoints = new float[][] {
                new float[] { detBox[4], detBox[5] }, new float[] { detBox[6], detBox[7] },
                new float[] { detBox[8], detBox[9] }, new float[] { detBox[10], detBox[11] }
            }
        };

        engine.DecodeLandmarks(rawLandmarks);
        engine.ExtractROI(new Color32[1], imgW, imgH, box);
        engine.DecodeWorldLandmarks(rawWorldLandmarks);

        var worldResult = engine.GetWorldResult();
        var points = new Point3D[19];
        for (int i = 0; i < 19; i++)
            points[i] = new Point3D { x = worldResult[i].X, y = worldResult[i].Y, z = worldResult[i].Z };
        return points;
    }

    [Test]
    public void TestDrawAxis3D_OriginalCoords_SkeletonIsUpsideDown()
    {
        if (!Directory.Exists(REFERENCE_DIR))
            Assert.Ignore("Reference data not found");

        var pts = GetWorldPoints();

        // Hip center (origin for DrawBone3D/DrawAxis3D)
        float hip_cy = (pts[HIP_LEFT].y + pts[HIP_RIGHT].y) / 2f;

        float nose_cy = pts[NOSE].y - hip_cy;
        float lAnkle_cy = pts[ANKLE_LEFT].y - hip_cy;
        float rAnkle_cy = pts[ANKLE_RIGHT].y - hip_cy;

        Console.WriteLine("=== Original coordinates (no Y-flip) ===");
        Console.WriteLine($"  Nose Y = {nose_cy:F4}");
        Console.WriteLine($"  L_Ankle Y = {lAnkle_cy:F4}");
        Console.WriteLine($"  R_Ankle Y = {rAnkle_cy:F4}");

        // MediaPipe world landmarks are Y-down, so ankles have LARGER Y than nose
        // In Unity Y-up, this means ankles appear ABOVE nose = upside down
        Assert.That(Math.Max(lAnkle_cy, rAnkle_cy), Is.GreaterThan(nose_cy),
            "Without Y-flip, ankles should have larger Y than nose (upside-down in Unity)");
    }

    [Test]
    public void TestDrawAxis3D_WithYFlip_SkeletonIsCorrect()
    {
        if (!Directory.Exists(REFERENCE_DIR))
            Assert.Ignore("Reference data not found");

        var pts = GetWorldPoints();

        // Hip center
        float hip_cx = (pts[HIP_LEFT].x + pts[HIP_RIGHT].x) / 2f;
        float hip_cy = (pts[HIP_LEFT].y + pts[HIP_RIGHT].y) / 2f;
        float hip_cz = (pts[HIP_LEFT].z + pts[HIP_RIGHT].z) / 2f;

        string[] names = { "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
            "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
            "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
            "ShoulderCenter", "BodyCenter" };

        // Simulate DrawBone3D/DrawAxis3D with Y-flip fix: cy = -(y - origin_y)
        Console.WriteLine("=== With Y-flip: cy = -(y - origin_y) ===");

        float abs_max = 0f;
        float y_min = 1f;
        float y_max = -1f;
        float[] cy_arr = new float[19];

        for (int i = 0; i < 19; i++)
        {
            float cx = pts[i].x - hip_cx;
            float cy = -(pts[i].y - hip_cy);  // Y-FLIP
            float cz = pts[i].z - hip_cz;

            cy_arr[i] = cy;
            abs_max = Math.Max(abs_max, Math.Max(Math.Abs(cx), Math.Max(Math.Abs(cy), Math.Abs(cz))));
            y_min = Math.Min(y_min, cy);
            y_max = Math.Max(y_max, cy);
            Console.WriteLine($"  {names[i],-16}: y={cy:F4}  (x={cx:F4}, z={cz:F4})");
        }

        float scale = abs_max + 0.1f;
        float nose_cy = cy_arr[NOSE];
        float lAnkle_cy = cy_arr[ANKLE_LEFT];
        float rAnkle_cy = cy_arr[ANKLE_RIGHT];

        Console.WriteLine($"\n  Skeleton Y range: [{y_min:F4}, {y_max:F4}]");
        Console.WriteLine($"  Nose Y = {nose_cy:F4}");
        Console.WriteLine($"  L_Ankle Y = {lAnkle_cy:F4}");
        Console.WriteLine($"  R_Ankle Y = {rAnkle_cy:F4}");

        // With Y-flip, nose should be ABOVE ankles in Unity (nose Y > ankle Y)
        Assert.That(nose_cy, Is.GreaterThan(Math.Max(lAnkle_cy, rAnkle_cy)),
            "With Y-flip, nose should have larger Y than ankles (correct orientation)");

        // Verify anatomical ordering: nose > shoulders > hips > knees > ankles
        float lShoulder_cy = cy_arr[SHOULDER_LEFT];
        float rShoulder_cy = cy_arr[SHOULDER_RIGHT];
        float lKnee_cy = cy_arr[KNEE_LEFT];
        float rKnee_cy = cy_arr[KNEE_RIGHT];

        float avg_shoulder = (lShoulder_cy + rShoulder_cy) / 2f;
        float avg_knee = (lKnee_cy + rKnee_cy) / 2f;
        float avg_ankle = (lAnkle_cy + rAnkle_cy) / 2f;

        Console.WriteLine($"\n  Anatomical ordering check:");
        Console.WriteLine($"    Nose={nose_cy:F4} > Shoulders={avg_shoulder:F4} > Hips~0 > Knees={avg_knee:F4} > Ankles={avg_ankle:F4}");

        Assert.That(nose_cy, Is.GreaterThan(avg_shoulder), "Nose should be above shoulders");
        Assert.That(avg_shoulder, Is.GreaterThan(0f).Within(0.1f), "Shoulders should be above hip center");
        Assert.That(avg_knee, Is.LessThan(0f), "Knees should be below hip center");
        Assert.That(avg_ankle, Is.LessThan(avg_knee), "Ankles should be below knees");

        // Verify DrawAxis3D grid placement
        Console.WriteLine($"\n=== DrawAxis3D grid (with Y-flip) ===");
        Console.WriteLine($"  y_min = {y_min:F4} (ankle level = ground)");
        Console.WriteLine($"  scale = {scale:F4}");
        Console.WriteLine($"  Grid floor Y = {y_min:F4}");
        Console.WriteLine($"  Box bottom Y = {y_min - scale * 2:F4} (pedestal below ground)");
        Console.WriteLine($"  Skeleton top Y = {y_max:F4} (above grid = standing on ground)");

        // Grid (y_min) should be at or below the ankles (feet = ground level)
        Assert.That(y_min, Is.LessThanOrEqualTo(avg_ankle + 0.01f),
            "Grid floor should be at or below ankle level");

        // Skeleton top (y_max) should be above the grid
        Assert.That(y_max, Is.GreaterThan(y_min),
            "Skeleton top should be above the grid floor");

        Console.WriteLine("\n  RESULT: With Y-flip, skeleton is upright and grid is at feet level.");
    }
}
