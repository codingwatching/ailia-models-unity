using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using ailiaSDK;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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
    private const string OUTPUT_DIR = "/tmp/mediapipe_pose_test_output";

    // 19-keypoint indices
    const int NOSE = 0;
    const int EYE_LEFT = 1;
    const int EYE_RIGHT = 2;
    const int EAR_LEFT = 3;
    const int EAR_RIGHT = 4;
    const int SHOULDER_LEFT = 5;
    const int SHOULDER_RIGHT = 6;
    const int ELBOW_LEFT = 7;
    const int ELBOW_RIGHT = 8;
    const int WRIST_LEFT = 9;
    const int WRIST_RIGHT = 10;
    const int HIP_LEFT = 11;
    const int HIP_RIGHT = 12;
    const int KNEE_LEFT = 13;
    const int KNEE_RIGHT = 14;
    const int ANKLE_LEFT = 15;
    const int ANKLE_RIGHT = 16;
    const int SHOULDER_CENTER = 17;
    const int BODY_CENTER = 18;

    // Skeleton bone connections (same as AiliaPoseEstimatorsSample)
    static readonly int[][] BONES = {
        new[] { NOSE, SHOULDER_CENTER },
        new[] { SHOULDER_LEFT, SHOULDER_CENTER },
        new[] { SHOULDER_RIGHT, SHOULDER_CENTER },
        new[] { EYE_LEFT, NOSE },
        new[] { EYE_RIGHT, NOSE },
        new[] { EAR_LEFT, EYE_LEFT },
        new[] { EAR_RIGHT, EYE_RIGHT },
        new[] { ELBOW_LEFT, SHOULDER_LEFT },
        new[] { ELBOW_RIGHT, SHOULDER_RIGHT },
        new[] { WRIST_LEFT, ELBOW_LEFT },
        new[] { WRIST_RIGHT, ELBOW_RIGHT },
        new[] { HIP_LEFT, SHOULDER_LEFT },
        new[] { HIP_RIGHT, SHOULDER_RIGHT },
        new[] { HIP_LEFT, HIP_RIGHT },
        new[] { KNEE_LEFT, HIP_LEFT },
        new[] { ANKLE_LEFT, KNEE_LEFT },
        new[] { KNEE_RIGHT, HIP_RIGHT },
        new[] { ANKLE_RIGHT, KNEE_RIGHT },
    };

    // Left-side keypoints (drawn in cyan)
    static readonly HashSet<int> LEFT_INDICES = new HashSet<int> {
        EYE_LEFT, EAR_LEFT, SHOULDER_LEFT, ELBOW_LEFT, WRIST_LEFT,
        HIP_LEFT, KNEE_LEFT, ANKLE_LEFT
    };
    // Right-side keypoints (drawn in orange)
    static readonly HashSet<int> RIGHT_INDICES = new HashSet<int> {
        EYE_RIGHT, EAR_RIGHT, SHOULDER_RIGHT, ELBOW_RIGHT, WRIST_RIGHT,
        HIP_RIGHT, KNEE_RIGHT, ANKLE_RIGHT
    };

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

    // ========== Image rendering helpers ==========

    private static void DrawLine(Image<Rgba32> img, int x0, int y0, int x1, int y1, Rgba32 color, int thickness = 1)
    {
        // Bresenham's line algorithm with thickness
        for (int t = -thickness / 2; t <= thickness / 2; t++)
        {
            int dx = Math.Abs(x1 - x0), dy = Math.Abs(y1 - y0);
            bool steep = dy > dx;
            int ax0 = x0, ay0 = y0, ax1 = x1, ay1 = y1;
            if (steep) { ax0 += t; ax1 += t; } else { ay0 += t; ay1 += t; }
            DrawLineThin(img, ax0, ay0, ax1, ay1, color);
        }
    }

    private static void DrawLineThin(Image<Rgba32> img, int x0, int y0, int x1, int y1, Rgba32 color)
    {
        int dx = Math.Abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -Math.Abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        while (true)
        {
            if (x0 >= 0 && x0 < img.Width && y0 >= 0 && y0 < img.Height)
                img[x0, y0] = AlphaBlend(img[x0, y0], color);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }

    private static Rgba32 AlphaBlend(Rgba32 bg, Rgba32 fg)
    {
        if (fg.A == 255) return fg;
        if (fg.A == 0) return bg;
        float a = fg.A / 255f;
        return new Rgba32(
            (byte)(fg.R * a + bg.R * (1 - a)),
            (byte)(fg.G * a + bg.G * (1 - a)),
            (byte)(fg.B * a + bg.B * (1 - a)),
            255);
    }

    private static void DrawCircle(Image<Rgba32> img, int cx, int cy, int r, Rgba32 color)
    {
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++)
                if (dx * dx + dy * dy <= r * r)
                {
                    int px = cx + dx, py = cy + dy;
                    if (px >= 0 && px < img.Width && py >= 0 && py < img.Height)
                        img[px, py] = color;
                }
    }

    private static void DrawText(Image<Rgba32> img, int x, int y, string text, Rgba32 color)
    {
        // Simple 3x5 bitmap font for digits, letters, and basic symbols
        var glyphs = new System.Collections.Generic.Dictionary<char, string[]> {
            {'0', new[]{"###","# #","# #","# #","###"}},
            {'1', new[]{" # "," # "," # "," # "," # "}},
            {'2', new[]{"###","  #","###","#  ","###"}},
            {'3', new[]{"###","  #","###","  #","###"}},
            {'4', new[]{"# #","# #","###","  #","  #"}},
            {'5', new[]{"###","#  ","###","  #","###"}},
            {'6', new[]{"###","#  ","###","# #","###"}},
            {'7', new[]{"###","  #","  #","  #","  #"}},
            {'8', new[]{"###","# #","###","# #","###"}},
            {'9', new[]{"###","# #","###","  #","###"}},
            {'N', new[]{"# #","###","###","# #","# #"}},
            {'o', new[]{"   ","###","# #","# #","###"}},
            {'s', new[]{"   ","###","#  ","  #","###"}},
            {'e', new[]{"   ","###","###","#  ","###"}},
            {'A', new[]{"###","# #","###","# #","# #"}},
            {'n', new[]{"   ","###","# #","# #","# #"}},
            {'k', new[]{"# #","# #","## ","# #","# #"}},
            {'l', new[]{"#  ","#  ","#  ","#  ","###"}},
            {'H', new[]{"# #","# #","###","# #","# #"}},
            {'i', new[]{" # ","   "," # "," # "," # "}},
            {'p', new[]{"   ","###","# #","###","#  "}},
            {'K', new[]{"# #","## ","#  ","## ","# #"}},
            {'W', new[]{"# #","# #","###","###","# #"}},
            {'r', new[]{"   ","###","# #","#  ","#  "}},
            {'S', new[]{"###","#  ","###","  #","###"}},
            {'h', new[]{"#  ","#  ","###","# #","# #"}},
            {'d', new[]{"  #","  #","###","# #","###"}},
            {'G', new[]{"###","#  ","# #","# #","###"}},
            {'R', new[]{"###","# #","###","## ","# #"}},
            {'I', new[]{"###"," # "," # "," # ","###"}},
            {'D', new[]{"## ","# #","# #","# #","## "}},
            {'-', new[]{"   ","   ","###","   ","   "}},
            {'.', new[]{"   ","   ","   ","   "," # "}},
            {' ', new[]{"   ","   ","   ","   ","   "}},
            {'(', new[]{" # ","#  ","#  ","#  "," # "}},
            {')', new[]{" # ","  #","  #","  #"," # "}},
            {'Y', new[]{"# #","# #","###"," # "," # "}},
            {':', new[]{"   "," # ","   "," # ","   "}},
            {'=', new[]{"   ","###","   ","###","   "}},
            {'>', new[]{"#  "," # ","  #"," # ","#  "}},
            {'<', new[]{"  #"," # ","#  "," # ","  #"}},
            {'~', new[]{"   ","   "," ##","## ","   "}},
            {'_', new[]{"   ","   ","   ","   ","###"}},
        };
        int cx = x;
        foreach (char c in text)
        {
            if (glyphs.TryGetValue(c, out var g))
            {
                for (int row = 0; row < 5; row++)
                    for (int col = 0; col < g[row].Length; col++)
                        if (g[row][col] == '#')
                        {
                            int px = cx + col, py = y + row;
                            if (px >= 0 && px < img.Width && py >= 0 && py < img.Height)
                                img[px, py] = color;
                        }
            }
            cx += 4; // 3 wide + 1 spacing
        }
    }

    /// <summary>
    /// Renders DrawBone3D + DrawAxis3D simulation to an image.
    /// Reproduces the same coordinate transforms as AiliaRenderer.
    /// </summary>
    private void RenderSkeletonImage(Point3D[] pts, bool flipY, float rotationAngle, string outputPath, string title)
    {
        int W = 600, H = 600;
        var img = new Image<Rgba32>(W, H);

        // Background
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img[x, y] = new Rgba32(32, 32, 32, 255);

        // Hip center
        float hip_cx = (pts[HIP_LEFT].x + pts[HIP_RIGHT].x) / 2f;
        float hip_cy = (pts[HIP_LEFT].y + pts[HIP_RIGHT].y) / 2f;
        float hip_cz = (pts[HIP_LEFT].z + pts[HIP_RIGHT].z) / 2f;

        // Hip-center and optionally flip Y (same as DrawBone3D/DrawAxis3D)
        float[] cx_arr = new float[19], cy_arr = new float[19], cz_arr = new float[19];
        for (int i = 0; i < 19; i++)
        {
            cx_arr[i] = pts[i].x - hip_cx;
            cy_arr[i] = flipY ? -(pts[i].y - hip_cy) : (pts[i].y - hip_cy);
            cz_arr[i] = pts[i].z - hip_cz;
        }

        // Compute DrawAxis3D parameters
        float abs_max = 0f, y_min = 1f;
        for (int i = 0; i < 19; i++)
        {
            abs_max = Math.Max(abs_max, Math.Max(Math.Abs(cx_arr[i]),
                Math.Max(Math.Abs(cy_arr[i]), Math.Abs(cz_arr[i]))));
            y_min = Math.Min(y_min, cy_arr[i]);
        }
        float scale = abs_max + 0.1f;

        // Apply Y-axis rotation (same as DrawBone3D)
        float cosR = (float)Math.Cos(rotationAngle);
        float sinR = (float)Math.Sin(rotationAngle);
        float[] rx_arr = new float[19], rz_arr = new float[19];
        for (int i = 0; i < 19; i++)
        {
            rx_arr[i] = cx_arr[i] * cosR + cz_arr[i] * sinR;
            rz_arr[i] = -cx_arr[i] * sinR + cz_arr[i] * cosR;
        }

        // Orthographic projection: screen X = rotated X, screen Y = centered Y
        // Find the total bounding box including grid/box
        float boxBottom = y_min - scale * 2;
        float viewYMin = Math.Min(boxBottom, y_min);
        float viewYMax = 0f;
        for (int i = 0; i < 19; i++)
            viewYMax = Math.Max(viewYMax, cy_arr[i]);
        viewYMax = Math.Max(viewYMax, y_min); // grid is also visible

        float viewXMin = -scale, viewXMax = scale;
        // Add margin
        float marginX = (viewXMax - viewXMin) * 0.1f;
        float marginY = (viewYMax - viewYMin) * 0.1f;
        viewXMin -= marginX; viewXMax += marginX;
        viewYMin -= marginY; viewYMax += marginY;

        // Scale to fit image
        float scaleX = (W - 40) / (viewXMax - viewXMin);
        float scaleY = (H - 40) / (viewYMax - viewYMin);
        float viewScale = Math.Min(scaleX, scaleY);

        // Center in image
        float offsetX = W / 2f - (viewXMin + viewXMax) / 2f * viewScale;
        float offsetY = H / 2f + (viewYMin + viewYMax) / 2f * viewScale; // Y is flipped for screen

        // Project 3D to screen (Unity Y-up -> screen Y-down)
        Func<float, float, (int, int)> toScreen = (wx, wy) => {
            int sx = (int)(wx * viewScale + offsetX);
            int sy = (int)(-wy * viewScale + offsetY);
            return (sx, sy);
        };

        // ======== Draw DrawAxis3D grid and box ========
        // Rotate grid corners around Y axis
        Func<float, float, (float, float)> rotXZ = (gx, gz) => {
            return (gx * cosR + gz * sinR, -gx * sinR + gz * cosR);
        };

        // Grid corners (top face at y_min)
        var gridCorners = new (float x, float z)[] {
            rotXZ(-scale, -scale), rotXZ(scale, -scale),
            rotXZ(scale, scale), rotXZ(-scale, scale)
        };
        // Box bottom corners (at y_min - scale*2)
        var boxCorners = new (float x, float z)[] {
            rotXZ(-scale, -scale), rotXZ(scale, -scale),
            rotXZ(scale, scale), rotXZ(-scale, scale)
        };

        var gridColor = new Rgba32(0, 92, 0, 255);
        var gridLineColor = new Rgba32(255, 255, 255, 180);
        var boxLineColor = new Rgba32(255, 255, 255, 100);
        var innerGridColor = new Rgba32(255, 255, 255, 40);

        // Draw box vertical edges
        for (int i = 0; i < 4; i++)
        {
            var (sx1, sy1) = toScreen(gridCorners[i].x, y_min);
            var (sx2, sy2) = toScreen(boxCorners[i].x, boxBottom);
            DrawLine(img, sx1, sy1, sx2, sy2, boxLineColor, 1);
        }

        // Draw box bottom edges
        for (int i = 0; i < 4; i++)
        {
            int j = (i + 1) % 4;
            var (sx1, sy1) = toScreen(boxCorners[i].x, boxBottom);
            var (sx2, sy2) = toScreen(boxCorners[j].x, boxBottom);
            DrawLine(img, sx1, sy1, sx2, sy2, boxLineColor, 1);
        }

        // Draw grid top edges
        for (int i = 0; i < 4; i++)
        {
            int j = (i + 1) % 4;
            var (sx1, sy1) = toScreen(gridCorners[i].x, y_min);
            var (sx2, sy2) = toScreen(gridCorners[j].x, y_min);
            DrawLine(img, sx1, sy1, sx2, sy2, gridLineColor, 2);
        }

        // Draw inner grid lines on top face
        float[] gridFracs = { -0.75f, -0.5f, -0.25f, 0f, 0.25f, 0.5f, 0.75f };
        foreach (float f in gridFracs)
        {
            // Lines parallel to Z
            var (gx1, gz1) = rotXZ(scale * f, -scale);
            var (gx2, gz2) = rotXZ(scale * f, scale);
            var (s1x, s1y) = toScreen(gx1, y_min);
            var (s2x, s2y) = toScreen(gx2, y_min);
            DrawLine(img, s1x, s1y, s2x, s2y, innerGridColor, 1);

            // Lines parallel to X
            (gx1, gz1) = rotXZ(-scale, scale * f);
            (gx2, gz2) = rotXZ(scale, scale * f);
            (s1x, s1y) = toScreen(gx1, y_min);
            (s2x, s2y) = toScreen(gx2, y_min);
            DrawLine(img, s1x, s1y, s2x, s2y, innerGridColor, 1);
        }

        // Draw grid corner spheres
        for (int i = 0; i < 4; i++)
        {
            var (sx, sy) = toScreen(gridCorners[i].x, y_min);
            DrawCircle(img, sx, sy, 4, gridColor);
            (sx, sy) = toScreen(boxCorners[i].x, boxBottom);
            DrawCircle(img, sx, sy, 4, gridColor);
        }

        // ======== Draw skeleton (DrawBone3D) ========
        var boneColor = new Rgba32(255, 255, 255, 255);
        var leftColor = new Rgba32(0, 179, 255, 255);
        var rightColor = new Rgba32(248, 123, 0, 255);

        foreach (var bone in BONES)
        {
            int from = bone[0], to = bone[1];
            var (sx1, sy1) = toScreen(rx_arr[from], cy_arr[from]);
            var (sx2, sy2) = toScreen(rx_arr[to], cy_arr[to]);

            // Use same color logic as AiliaPoseEstimatorsSample
            Rgba32 lineColor = boneColor;
            if (LEFT_INDICES.Contains(from) || LEFT_INDICES.Contains(to))
                lineColor = leftColor;
            if (RIGHT_INDICES.Contains(from) || RIGHT_INDICES.Contains(to))
                lineColor = rightColor;
            if (bone[0] == NOSE || bone[0] == SHOULDER_CENTER || bone[0] == HIP_LEFT && bone[1] == HIP_RIGHT)
                lineColor = boneColor;

            DrawLine(img, sx1, sy1, sx2, sy2, lineColor, 2);
        }

        // Draw keypoint spheres
        for (int i = 0; i < 19; i++)
        {
            var (sx, sy) = toScreen(rx_arr[i], cy_arr[i]);
            Rgba32 sphereColor = boneColor;
            if (LEFT_INDICES.Contains(i)) sphereColor = leftColor;
            else if (RIGHT_INDICES.Contains(i)) sphereColor = rightColor;
            DrawCircle(img, sx, sy, 4, sphereColor);
        }

        // ======== Draw labels ========
        var labelColor = new Rgba32(200, 200, 200, 255);

        // Label nose
        {
            var (sx, sy) = toScreen(rx_arr[NOSE], cy_arr[NOSE]);
            DrawText(img, sx + 6, sy - 3, "Nose", labelColor);
        }
        // Label ankles
        {
            var (sx, sy) = toScreen(rx_arr[ANKLE_LEFT], cy_arr[ANKLE_LEFT]);
            DrawText(img, sx + 6, sy - 3, "Ankle", labelColor);
        }
        // Label hip
        {
            var (sx, sy) = toScreen(rx_arr[BODY_CENTER], cy_arr[BODY_CENTER]);
            DrawText(img, sx + 6, sy - 3, "Hip", labelColor);
        }

        // Title
        DrawText(img, 10, 10, title, new Rgba32(255, 255, 100, 255));

        // Grid label
        {
            var (sx, sy) = toScreen(gridCorners[0].x, y_min);
            DrawText(img, sx + 8, sy - 3, "GRID", new Rgba32(0, 200, 0, 255));
        }

        // Y-axis direction indicator
        var arrowColor = new Rgba32(255, 80, 80, 255);
        int arrowX = W - 40;
        DrawLine(img, arrowX, H - 60, arrowX, H - 20, arrowColor, 2);
        DrawLine(img, arrowX, H - 60, arrowX - 5, H - 50, arrowColor, 2);
        DrawLine(img, arrowX, H - 60, arrowX + 5, H - 50, arrowColor, 2);
        DrawText(img, arrowX - 4, H - 72, "Y", arrowColor);

        img.SaveAsPng(outputPath);
        Console.WriteLine($"  Saved: {outputPath}");
    }

    // ========== Tests ==========

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

    [Test]
    public void TestDrawAxis3D_RenderImages()
    {
        if (!Directory.Exists(REFERENCE_DIR))
            Assert.Ignore("Reference data not found");

        var pts = GetWorldPoints();
        float rotAngle = (float)(Math.PI / 6); // 30 degrees for good 3D view

        // Render without Y-flip (broken)
        string brokenPath = Path.Combine(OUTPUT_DIR, "skeleton_no_yflip.png");
        RenderSkeletonImage(pts, flipY: false, rotAngle, brokenPath,
            "No Y-flip (BROKEN)");

        // Render with Y-flip (fixed)
        string fixedPath = Path.Combine(OUTPUT_DIR, "skeleton_with_yflip.png");
        RenderSkeletonImage(pts, flipY: true, rotAngle, fixedPath,
            "With Y-flip (FIXED)");

        Console.WriteLine($"\n=== Rendered images ===");
        Console.WriteLine($"  Broken (no Y-flip): {brokenPath}");
        Console.WriteLine($"  Fixed  (Y-flip):    {fixedPath}");
        Console.WriteLine($"\n  In the FIXED image:");
        Console.WriteLine($"    - Nose should be at the TOP");
        Console.WriteLine($"    - Ankles should be at the BOTTOM");
        Console.WriteLine($"    - Grid (green) should be below the ankles (ground plane)");
        Console.WriteLine($"    - Box/pedestal should extend BELOW the grid");

        Assert.That(File.Exists(brokenPath), Is.True, "Broken image should be saved");
        Assert.That(File.Exists(fixedPath), Is.True, "Fixed image should be saved");
    }
}
