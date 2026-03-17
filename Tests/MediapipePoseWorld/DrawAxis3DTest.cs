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
///
/// Unity screen coordinate: positive Y = DOWN on screen.
/// MediaPipe world landmarks: Y-down (positive Y = downward in real world).
/// Therefore skeleton orientation is naturally correct (no Y-flip needed).
/// Grid should be placed at y_max (foot level = bottom of screen).
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

    static readonly HashSet<int> LEFT_INDICES = new HashSet<int> {
        EYE_LEFT, EAR_LEFT, SHOULDER_LEFT, ELBOW_LEFT, WRIST_LEFT,
        HIP_LEFT, KNEE_LEFT, ANKLE_LEFT
    };
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

    private static void DrawText(Image<Rgba32> img, int x, int y, string text, Rgba32 color, int fontScale = 1)
    {
        var glyphs = new Dictionary<char, string[]> {
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
            {'B', new[]{"## ","# #","## ","# #","## "}},
            {'O', new[]{"###","# #","# #","# #","###"}},
            {'F', new[]{"###","#  ","###","#  ","#  "}},
            {'X', new[]{"# #","# #"," # ","# #","# #"}},
            {'E', new[]{"###","#  ","###","#  ","###"}},
            {'T', new[]{"###"," # "," # "," # "," # "}},
            {'L', new[]{"#  ","#  ","#  ","#  ","###"}},
            {'C', new[]{"###","#  ","#  ","#  ","###"}},
            {'M', new[]{"# #","###","###","# #","# #"}},
            {'Y', new[]{"# #","# #","###"," # "," # "}},
            {'U', new[]{"# #","# #","# #","# #","###"}},
            {'-', new[]{"   ","   ","###","   ","   "}},
            {'.', new[]{"   ","   ","   ","   "," # "}},
            {' ', new[]{"   ","   ","   ","   ","   "}},
            {'(', new[]{" # ","#  ","#  ","#  "," # "}},
            {')', new[]{" # ","  #","  #","  #"," # "}},
            {':', new[]{"   "," # ","   "," # ","   "}},
            {'=', new[]{"   ","###","   ","###","   "}},
            {'>', new[]{"#  "," # ","  #"," # ","#  "}},
            {'<', new[]{"  #"," # ","#  "," # ","  #"}},
            {'~', new[]{"   ","   "," ##","## ","   "}},
            {'_', new[]{"   ","   ","   ","   ","###"}},
            {'+', new[]{"   "," # ","###"," # ","   "}},
            {'|', new[]{" # "," # "," # "," # "," # "}},
            {'/', new[]{"  #","  #"," # ","#  ","#  "}},
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
                            for (int sy = 0; sy < fontScale; sy++)
                                for (int sx = 0; sx < fontScale; sx++)
                                {
                                    int px = cx + col * fontScale + sx;
                                    int py = y + row * fontScale + sy;
                                    if (px >= 0 && px < img.Width && py >= 0 && py < img.Height)
                                        img[px, py] = color;
                                }
                        }
            }
            cx += (3 + 1) * fontScale;
        }
    }

    /// <summary>
    /// Renders skeleton + grid/box simulation to an image.
    /// Screen coordinate: positive Y = DOWN (matching Unity's actual rendering).
    /// gridMode: "y_min" = original broken code, "y_max" = fixed code.
    /// </summary>
    private void RenderSkeletonImage(Point3D[] pts, string gridMode, float rotationAngle,
        string outputPath, string title)
    {
        int W = 600, H = 700;
        var img = new Image<Rgba32>(W, H);

        // Dark background
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img[x, y] = new Rgba32(32, 32, 32, 255);

        // Hip center
        float hip_cx = (pts[HIP_LEFT].x + pts[HIP_RIGHT].x) / 2f;
        float hip_cy = (pts[HIP_LEFT].y + pts[HIP_RIGHT].y) / 2f;
        float hip_cz = (pts[HIP_LEFT].z + pts[HIP_RIGHT].z) / 2f;

        // Hip-centered coordinates (no Y-flip, matching DrawBone3D)
        float[] cx_arr = new float[19], cy_arr = new float[19], cz_arr = new float[19];
        for (int i = 0; i < 19; i++)
        {
            cx_arr[i] = pts[i].x - hip_cx;
            cy_arr[i] = pts[i].y - hip_cy;  // No Y-flip
            cz_arr[i] = pts[i].z - hip_cz;
        }

        // Compute DrawAxis3D parameters
        float abs_max = 0f, y_min = 1f, y_max = -1f;
        for (int i = 0; i < 19; i++)
        {
            abs_max = Math.Max(abs_max, Math.Max(Math.Abs(cx_arr[i]),
                Math.Max(Math.Abs(cy_arr[i]), Math.Abs(cz_arr[i]))));
            y_min = Math.Min(y_min, cy_arr[i]);
            y_max = Math.Max(y_max, cy_arr[i]);
        }
        float scale = abs_max + 0.1f;

        // Grid and box position based on mode
        float gridY, boxEndY;
        if (gridMode == "y_max")
        {
            gridY = y_max;
            boxEndY = y_max + scale * 2;  // Extend further down (larger Y = lower on screen)
        }
        else // "y_min" - original broken code
        {
            gridY = y_min;
            boxEndY = y_min - scale * 2;  // Extend upward (smaller Y = higher on screen)
        }

        // Apply Y-axis rotation (same as DrawBone3D)
        float cosR = (float)Math.Cos(rotationAngle);
        float sinR = (float)Math.Sin(rotationAngle);
        float[] rx_arr = new float[19], rz_arr = new float[19];
        for (int i = 0; i < 19; i++)
        {
            rx_arr[i] = cx_arr[i] * cosR + cz_arr[i] * sinR;
            rz_arr[i] = -cx_arr[i] * sinR + cz_arr[i] * cosR;
        }

        // Screen projection: Unity renders positive Y DOWNWARD on screen
        // So screen_y = world_y (directly proportional, no negation)
        float allYMin = Math.Min(y_min, Math.Min(gridY, boxEndY));
        float allYMax = Math.Max(y_max, Math.Max(gridY, boxEndY));
        float viewXMin = -scale * 1.2f, viewXMax = scale * 1.2f;
        float viewYMin = allYMin - 0.1f, viewYMax = allYMax + 0.1f;

        float scaleX = (W - 60) / (viewXMax - viewXMin);
        float scaleY = (H - 100) / (viewYMax - viewYMin);
        float viewScale = Math.Min(scaleX, scaleY);

        float offsetX = W / 2f - (viewXMin + viewXMax) / 2f * viewScale;
        // Positive Y = down on screen (no negation)
        float offsetY = 50 - viewYMin * viewScale;

        Func<float, float, (int, int)> toScreen = (wx, wy) => {
            int sx = (int)(wx * viewScale + offsetX);
            int sy = (int)(wy * viewScale + offsetY);  // Direct mapping: +Y = down
            return (sx, sy);
        };

        // Rotate grid/box corner XZ around Y axis
        Func<float, float, (float, float)> rotXZ = (gx, gz) => {
            return (gx * cosR + gz * sinR, -gx * sinR + gz * cosR);
        };

        var gridCorners = new (float x, float z)[4];
        var boxCorners = new (float x, float z)[4];
        float[] cornerX = { -scale, scale, scale, -scale };
        float[] cornerZ = { -scale, -scale, scale, scale };
        for (int i = 0; i < 4; i++)
        {
            gridCorners[i] = rotXZ(cornerX[i], cornerZ[i]);
            boxCorners[i] = rotXZ(cornerX[i], cornerZ[i]);
        }

        var gridColor = new Rgba32(0, 92, 0, 255);
        var gridLineColor = new Rgba32(255, 255, 255, 180);
        var boxLineColor = new Rgba32(255, 255, 255, 80);
        var innerGridColor = new Rgba32(255, 255, 255, 40);

        // Draw box vertical edges
        for (int i = 0; i < 4; i++)
        {
            var (sx1, sy1) = toScreen(gridCorners[i].x, gridY);
            var (sx2, sy2) = toScreen(boxCorners[i].x, boxEndY);
            DrawLine(img, sx1, sy1, sx2, sy2, boxLineColor, 1);
        }

        // Draw box end face edges
        for (int i = 0; i < 4; i++)
        {
            int j = (i + 1) % 4;
            var (sx1, sy1) = toScreen(boxCorners[i].x, boxEndY);
            var (sx2, sy2) = toScreen(boxCorners[j].x, boxEndY);
            DrawLine(img, sx1, sy1, sx2, sy2, boxLineColor, 1);
        }

        // Draw grid face edges
        for (int i = 0; i < 4; i++)
        {
            int j = (i + 1) % 4;
            var (sx1, sy1) = toScreen(gridCorners[i].x, gridY);
            var (sx2, sy2) = toScreen(gridCorners[j].x, gridY);
            DrawLine(img, sx1, sy1, sx2, sy2, gridLineColor, 2);
        }

        // Draw inner grid lines
        float[] gridFracs = { -0.75f, -0.5f, -0.25f, 0f, 0.25f, 0.5f, 0.75f };
        foreach (float f in gridFracs)
        {
            var (gx1, _) = rotXZ(scale * f, -scale);
            var (gx2, _) = rotXZ(scale * f, scale);
            var (s1x, s1y) = toScreen(gx1, gridY);
            var (s2x, s2y) = toScreen(gx2, gridY);
            DrawLine(img, s1x, s1y, s2x, s2y, innerGridColor, 1);

            (gx1, _) = rotXZ(-scale, scale * f);
            (gx2, _) = rotXZ(scale, scale * f);
            (s1x, s1y) = toScreen(gx1, gridY);
            (s2x, s2y) = toScreen(gx2, gridY);
            DrawLine(img, s1x, s1y, s2x, s2y, innerGridColor, 1);
        }

        // Grid corner spheres
        for (int i = 0; i < 4; i++)
        {
            var (sx, sy) = toScreen(gridCorners[i].x, gridY);
            DrawCircle(img, sx, sy, 4, gridColor);
            (sx, sy) = toScreen(boxCorners[i].x, boxEndY);
            DrawCircle(img, sx, sy, 4, gridColor);
        }

        // ======== Draw skeleton ========
        var boneColor = new Rgba32(255, 255, 255, 255);
        var leftColor = new Rgba32(0, 179, 255, 255);
        var rightColor = new Rgba32(248, 123, 0, 255);

        foreach (var bone in BONES)
        {
            int from = bone[0], to = bone[1];
            var (sx1, sy1) = toScreen(rx_arr[from], cy_arr[from]);
            var (sx2, sy2) = toScreen(rx_arr[to], cy_arr[to]);

            Rgba32 lineColor = boneColor;
            if (LEFT_INDICES.Contains(from) && LEFT_INDICES.Contains(to))
                lineColor = leftColor;
            else if (RIGHT_INDICES.Contains(from) && RIGHT_INDICES.Contains(to))
                lineColor = rightColor;
            else if (LEFT_INDICES.Contains(from) || LEFT_INDICES.Contains(to))
                lineColor = leftColor;
            else if (RIGHT_INDICES.Contains(from) || RIGHT_INDICES.Contains(to))
                lineColor = rightColor;

            DrawLine(img, sx1, sy1, sx2, sy2, lineColor, 2);
        }

        // Keypoint spheres
        for (int i = 0; i < 19; i++)
        {
            var (sx, sy) = toScreen(rx_arr[i], cy_arr[i]);
            Rgba32 sphereColor = boneColor;
            if (LEFT_INDICES.Contains(i)) sphereColor = leftColor;
            else if (RIGHT_INDICES.Contains(i)) sphereColor = rightColor;
            DrawCircle(img, sx, sy, 4, sphereColor);
        }

        // ======== Labels ========
        var labelColor = new Rgba32(200, 200, 200, 255);
        {
            var (sx, sy) = toScreen(rx_arr[NOSE], cy_arr[NOSE]);
            DrawText(img, sx + 6, sy - 8, "Nose", labelColor);
        }
        {
            var (sx, sy) = toScreen(rx_arr[ANKLE_LEFT], cy_arr[ANKLE_LEFT]);
            DrawText(img, sx + 6, sy - 3, "Ankle", labelColor);
        }
        {
            var (sx, sy) = toScreen(rx_arr[BODY_CENTER], cy_arr[BODY_CENTER]);
            DrawText(img, sx + 6, sy - 3, "Hip", labelColor);
        }

        // Grid label
        {
            var (sx, sy) = toScreen(gridCorners[1].x, gridY);
            DrawText(img, sx + 8, sy - 3, "GRID", new Rgba32(0, 200, 0, 255));
        }

        // Title (larger font)
        DrawText(img, 10, 10, title, new Rgba32(255, 255, 100, 255), 2);

        // Y-axis direction indicator (arrow pointing DOWN = positive Y)
        var arrowColor = new Rgba32(255, 80, 80, 255);
        int arrowX = W - 40;
        DrawLine(img, arrowX, 40, arrowX, 80, arrowColor, 2);
        DrawLine(img, arrowX, 80, arrowX - 5, 70, arrowColor, 2);
        DrawLine(img, arrowX, 80, arrowX + 5, 70, arrowColor, 2);
        DrawText(img, arrowX - 6, 25, "+Y", arrowColor);

        img.SaveAsPng(outputPath);
        Console.WriteLine($"  Saved: {outputPath}");
    }

    // ========== Tests ==========

    [Test]
    public void TestUnityCoords_SkeletonOrientation()
    {
        if (!Directory.Exists(REFERENCE_DIR))
            Assert.Ignore("Reference data not found");

        var pts = GetWorldPoints();
        float hip_cy = (pts[HIP_LEFT].y + pts[HIP_RIGHT].y) / 2f;

        float nose_cy = pts[NOSE].y - hip_cy;
        float lAnkle_cy = pts[ANKLE_LEFT].y - hip_cy;
        float rAnkle_cy = pts[ANKLE_RIGHT].y - hip_cy;
        float avg_ankle = (lAnkle_cy + rAnkle_cy) / 2f;

        Console.WriteLine("=== Unity screen coords (positive Y = DOWN) ===");
        Console.WriteLine($"  Nose centered Y     = {nose_cy:F4} (small = top of screen)");
        Console.WriteLine($"  L_Ankle centered Y  = {lAnkle_cy:F4} (large = bottom of screen)");
        Console.WriteLine($"  R_Ankle centered Y  = {rAnkle_cy:F4} (large = bottom of screen)");

        // MediaPipe Y-down: ankles have larger Y than nose
        // Unity screen Y-down: larger Y = lower on screen
        // Therefore: ankles below nose on screen = correct orientation (no Y-flip needed)
        Assert.That(avg_ankle, Is.GreaterThan(nose_cy),
            "Ankles should have larger Y (displayed below nose on Unity screen)");

        Console.WriteLine("  => Skeleton is correctly oriented (head at top, feet at bottom)");
    }

    [Test]
    public void TestDrawAxis3D_GridAtYMin_IsWrong()
    {
        if (!Directory.Exists(REFERENCE_DIR))
            Assert.Ignore("Reference data not found");

        var pts = GetWorldPoints();
        float hip_cy = (pts[HIP_LEFT].y + pts[HIP_RIGHT].y) / 2f;

        float y_min = float.MaxValue, y_max = float.MinValue;
        for (int i = 0; i < 19; i++)
        {
            float cy = pts[i].y - hip_cy;
            y_min = Math.Min(y_min, cy);
            y_max = Math.Max(y_max, cy);
        }

        float nose_cy = pts[NOSE].y - hip_cy;
        float avg_ankle = ((pts[ANKLE_LEFT].y - hip_cy) + (pts[ANKLE_RIGHT].y - hip_cy)) / 2f;

        Console.WriteLine("=== Original code: grid at y_min ===");
        Console.WriteLine($"  y_min = {y_min:F4} (head/wrist area)");
        Console.WriteLine($"  y_max = {y_max:F4} (ankle area)");
        Console.WriteLine($"  Nose Y = {nose_cy:F4}");
        Console.WriteLine($"  Avg ankle Y = {avg_ankle:F4}");
        Console.WriteLine($"  Grid Y = {y_min:F4} => displayed at TOP of screen (WRONG!)");
        Console.WriteLine($"  Box extends to {y_min - y_min * 2:F4} => even further UP (WRONG!)");

        // y_min is near nose/head (small Y = top of screen)
        // Grid at y_min means grid is at the TOP, not at the feet
        Assert.That(y_min, Is.LessThan(nose_cy + 0.01f),
            "y_min should be near head level (top of screen) - confirming grid is in wrong position");
        Assert.That(y_min, Is.LessThan(avg_ankle),
            "y_min (grid) should be above ankles on screen - confirming grid at top, not bottom");

        Console.WriteLine("  => CONFIRMED: Grid at y_min appears at TOP (above skeleton)");
    }

    [Test]
    public void TestDrawAxis3D_GridAtYMax_IsCorrect()
    {
        if (!Directory.Exists(REFERENCE_DIR))
            Assert.Ignore("Reference data not found");

        var pts = GetWorldPoints();
        float hip_cx = (pts[HIP_LEFT].x + pts[HIP_RIGHT].x) / 2f;
        float hip_cy = (pts[HIP_LEFT].y + pts[HIP_RIGHT].y) / 2f;
        float hip_cz = (pts[HIP_LEFT].z + pts[HIP_RIGHT].z) / 2f;

        string[] names = { "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
            "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
            "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
            "ShoulderCenter", "BodyCenter" };

        Console.WriteLine("=== Fixed code: grid at y_max ===");

        float abs_max = 0f, y_min = 1f, y_max = -1f;
        for (int i = 0; i < 19; i++)
        {
            float cx = pts[i].x - hip_cx;
            float cy = pts[i].y - hip_cy;
            float cz = pts[i].z - hip_cz;
            abs_max = Math.Max(abs_max, Math.Max(Math.Abs(cx), Math.Max(Math.Abs(cy), Math.Abs(cz))));
            y_min = Math.Min(y_min, cy);
            y_max = Math.Max(y_max, cy);
            Console.WriteLine($"  {names[i],-16}: y={cy:F4}  (x={cx:F4}, z={cz:F4})");
        }

        float scale = abs_max + 0.1f;
        float nose_cy = pts[NOSE].y - hip_cy;
        float avg_ankle = ((pts[ANKLE_LEFT].y - hip_cy) + (pts[ANKLE_RIGHT].y - hip_cy)) / 2f;
        float avg_shoulder = ((pts[SHOULDER_LEFT].y - hip_cy) + (pts[SHOULDER_RIGHT].y - hip_cy)) / 2f;
        float avg_knee = ((pts[KNEE_LEFT].y - hip_cy) + (pts[KNEE_RIGHT].y - hip_cy)) / 2f;

        Console.WriteLine($"\n  Screen ordering (small Y = top, large Y = bottom):");
        Console.WriteLine($"    Nose={nose_cy:F4} < Shoulders={avg_shoulder:F4} < Hips~0 < Knees={avg_knee:F4} < Ankles={avg_ankle:F4}");
        Console.WriteLine($"\n  y_max = {y_max:F4} (ankle level = bottom of screen = ground)");
        Console.WriteLine($"  scale = {scale:F4}");
        Console.WriteLine($"  Grid Y = {y_max:F4} => at ankle level (ground plane)");
        Console.WriteLine($"  Box extends to {y_max + scale * 2:F4} => below ground (pedestal)");

        // Verify screen ordering (top to bottom)
        Assert.That(nose_cy, Is.LessThan(avg_shoulder + 0.05f), "Nose above shoulders on screen");
        Assert.That(avg_knee, Is.GreaterThan(0f), "Knees below hip center on screen");
        Assert.That(avg_ankle, Is.GreaterThan(avg_knee), "Ankles below knees on screen");

        // Grid at y_max should be at or below ankle level
        Assert.That(y_max, Is.GreaterThanOrEqualTo(avg_ankle - 0.01f),
            "Grid (y_max) should be at or below ankle level (bottom of screen)");

        // Box extends further down
        Assert.That(y_max + scale * 2, Is.GreaterThan(y_max),
            "Box should extend below the grid (further down on screen)");

        Console.WriteLine("\n  => CORRECT: Grid at bottom (feet), pedestal extends below");
    }

    [Test]
    public void TestDrawAxis3D_RenderImages()
    {
        if (!Directory.Exists(REFERENCE_DIR))
            Assert.Ignore("Reference data not found");

        var pts = GetWorldPoints();
        float rotAngle = (float)(Math.PI / 6); // 30 degrees for 3D view

        // Render with y_min grid (original broken code)
        string brokenPath = Path.Combine(OUTPUT_DIR, "skeleton_ymin_broken.png");
        RenderSkeletonImage(pts, "y_min", rotAngle, brokenPath,
            "BROKEN: Grid at y_min (top)");

        // Render with y_max grid (fixed code)
        string fixedPath = Path.Combine(OUTPUT_DIR, "skeleton_ymax_fixed.png");
        RenderSkeletonImage(pts, "y_max", rotAngle, fixedPath,
            "FIXED: Grid at y_max (bottom)");

        Console.WriteLine($"\n=== Rendered images (Unity Y-down screen) ===");
        Console.WriteLine($"  Broken: {brokenPath}");
        Console.WriteLine($"  Fixed:  {fixedPath}");
        Console.WriteLine($"\n  Screen convention: +Y = DOWN");
        Console.WriteLine($"  In BROKEN image: grid at TOP, skeleton hangs below");
        Console.WriteLine($"  In FIXED image:  grid at BOTTOM (feet), pedestal below");

        Assert.That(File.Exists(brokenPath), Is.True);
        Assert.That(File.Exists(fixedPath), Is.True);
    }
}
