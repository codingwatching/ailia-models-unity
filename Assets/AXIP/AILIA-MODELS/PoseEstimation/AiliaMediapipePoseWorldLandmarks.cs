using ailiaSDK;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

using ailia;

public class AiliaMediapipePoseWorldLandmarks : IDisposable
{
    private MediapipePoseWorldEngine engine = new MediapipePoseWorldEngine();
    private AiliaMediapipePoseBackend backend;
    private float[,] anchors;

    // GPU ROI extraction (Unity-specific)
    public ComputeShader computeShader = null;
    private int kernelIndex = -1;
    private int ID_InputTexture = Shader.PropertyToID("InputTexture");
    private int ID_InputWidth = Shader.PropertyToID("InputWidth");
    private int ID_InputHeight = Shader.PropertyToID("InputHeight");
    private int ID_OutputTexture = Shader.PropertyToID("OutputTexture");
    private int ID_OutputWidth = Shader.PropertyToID("OutputWidth");
    private int ID_OutputHeight = Shader.PropertyToID("OutputHeight");
    private int ID_Matrix = Shader.PropertyToID("Matrix");
    private int ID_BackgroundColor = Shader.PropertyToID("BackgroundColor");
    private RenderTexture computeTexture;
    private Texture2D roiTexture;
    private RenderTexture preprocessBuffer;
    private Texture2D input_texture;

    // Detection caching (skip re-detection after first frame)
    private MediapipePoseWorldEngine.DecodedBox? cachedDetectionBox;

    // Image dimensions for GetResult
    private int imageWidth;
    private int imageHeight;

    public AiliaMediapipePoseWorldLandmarks(bool gpuMode, string assetPath, string jsonPath)
    {
        backend = new AiliaMediapipePoseBackend(gpuMode);
        backend.LoadModels(
            $"{assetPath}/pose_detection.onnx",
            $"{assetPath}/pose_landmark_heavy.onnx"
        );

        // Load anchors
        var anchorsHolder = new AiliaMediapipePoseWorldLandmarksAnchors();
        float[] anchorsFlat = ConvertDoubleArrayToFloatArray(anchorsHolder.anchors);
        anchors = new float[MediapipePoseWorldEngine.DETECTOR_TENSOR_COUNT, 4];
        for (int i = 0; i < anchorsFlat.Length; ++i)
        {
            anchors[i / 4, i % 4] = anchorsFlat[i];
        }
    }

    public List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose> RunPoseEstimation(Color32[] camera, int tex_width, int tex_height)
    {
        imageWidth = tex_width;
        imageHeight = tex_height;

        // Convert B2T Color32[] to T2B for engine (Unity GetPixels32 returns B2T)
        Color32[] pixelsT2B = new Color32[camera.Length];
        for (int y = 0; y < tex_height; y++)
        {
            Array.Copy(camera, (tex_height - 1 - y) * tex_width, pixelsT2B, y * tex_width, tex_width);
        }

        // Run detection (or use cached box)
        MediapipePoseWorldEngine.DecodedBox? detectionBox = cachedDetectionBox;

        if (!detectionBox.HasValue)
        {
            float padH, padW;
            float[] detInput = engine.PreprocessDetection(pixelsT2B, tex_width, tex_height, out padH, out padW);
            var detOutput = backend.RunDetector(detInput,
                MediapipePoseWorldEngine.DETECTOR_INPUT_RESOLUTION,
                MediapipePoseWorldEngine.DETECTOR_INPUT_RESOLUTION,
                MediapipePoseWorldEngine.DETECTOR_INPUT_CHANNEL_COUNT);
            var boxes = engine.DecodeAndProcessBoxes(detOutput.RawBoxes, detOutput.RawScores, anchors, padH, padW);

            if (boxes.Count == 0)
            {
                Debug.Log("NO POSE");
                return new List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose>();
            }

            detectionBox = boxes[0];
            cachedDetectionBox = detectionBox;
        }

        // Extract ROI using GPU ComputeShader (Unity-specific, higher performance)
        Texture2D roiTex = ExtractROIFromBoxGpu(camera, tex_width, tex_height, detectionBox.Value);

        // Preprocess ROI texture for estimator ([0,1] normalization)
        int estRes = MediapipePoseWorldEngine.ESTIMATOR_INPUT_RESOLUTION;
        Color32[] roiPixels = roiTex.GetPixels32();
        float[] estInput = new float[estRes * estRes * MediapipePoseWorldEngine.DETECTOR_INPUT_CHANNEL_COUNT];
        const float factor = 1 / 255f;
        for (int y = 0; y < estRes; y++)
        {
            for (int x = 0; x < estRes; x++)
            {
                int idx = (y * estRes + x) * 3;
                Color32 c = roiPixels[y * estRes + x];
                estInput[idx + 0] = c.r * factor;
                estInput[idx + 1] = c.g * factor;
                estInput[idx + 2] = c.b * factor;
            }
        }

        // Run estimator
        var estOutput = backend.RunEstimator(estInput, estRes, estRes, MediapipePoseWorldEngine.DETECTOR_INPUT_CHANNEL_COUNT);

        // Decode landmarks
        engine.DecodeLandmarks(estOutput.Landmarks);

        return GetResult(false);
    }

    public List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose> GetResult(bool world_cordinate)
    {
        if (engine.Landmarks == null)
        {
            return new List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose>();
        }

        PoseLandmarkResult[] results;
        if (world_cordinate)
        {
            results = engine.GetWorldResult();
        }
        else
        {
            results = engine.GetImageResult(imageWidth, imageHeight);
        }

        if (results == null)
        {
            return new List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose>();
        }

        // Convert PoseLandmarkResult[19] to AILIAPoseEstimatorObjectPose
        var result_list = new List<AiliaPoseEstimator.AILIAPoseEstimatorObjectPose>();
        AiliaPoseEstimator.AILIAPoseEstimatorObjectPose one_pose = new AiliaPoseEstimator.AILIAPoseEstimatorObjectPose();
        one_pose.points = new AiliaPoseEstimator.AILIAPoseEstimatorKeypoint[19];

        for (int i = 0; i < results.Length; i++)
        {
            AiliaPoseEstimator.AILIAPoseEstimatorKeypoint keypoint = new AiliaPoseEstimator.AILIAPoseEstimatorKeypoint();
            keypoint.x = results[i].X;
            keypoint.y = results[i].Y;
            keypoint.z_local = results[i].Z;
            keypoint.score = results[i].Confidence;
            one_pose.points[i] = keypoint;
        }

        result_list.Add(one_pose);
        return result_list;
    }

    public string EnvironmentName()
    {
        return backend.EnvironmentName();
    }

    // -------------------------------------------------------
    // GPU ROI extraction using ComputeShader (Unity-specific)
    // -------------------------------------------------------
    private Texture2D ExtractROIFromBoxGpu(Color32[] camera, int tex_width, int tex_height,
        MediapipePoseWorldEngine.DecodedBox box)
    {
        // Create texture from camera data
        if (input_texture == null)
        {
            input_texture = new Texture2D(tex_width, tex_height);
        }
        input_texture.SetPixels32(camera);
        input_texture.Apply();

        int estRes = MediapipePoseWorldEngine.ESTIMATOR_INPUT_RESOLUTION;
        float dscale = MediapipePoseWorldEngine.ROI_SCALE_FACTOR;

        // Convert normalized box keypoints to pixel coordinates (B2T space for Unity)
        float finalSquareLength = Mathf.Max(tex_width, tex_height);
        int xOffset = (int)((finalSquareLength - tex_width) / 2);
        int yOffset = (int)((finalSquareLength - tex_height) / 2);

        float xc = finalSquareLength * box.keypoints[0][0] - xOffset;
        float yc = tex_height - 1 - finalSquareLength * box.keypoints[0][1] + yOffset;
        float x1 = finalSquareLength * box.keypoints[1][0] - xOffset;
        float y1 = tex_height - 1 - finalSquareLength * box.keypoints[1][1] + yOffset;

        float theta0 = Mathf.PI / 2f;
        float scale = dscale * Mathf.Sqrt(Mathf.Pow(xc - x1, 2) + Mathf.Pow(yc - y1, 2)) * 2;
        float angle = Mathf.Atan2(yc - y1, xc - x1) - theta0;

        // Set engine ROI parameters (for GetImageResult coordinate transform)
        // Use T2B keypoint coordinates for engine
        float xcT2B = box.keypoints[0][0] * tex_width;
        float ycT2B = box.keypoints[0][1] * tex_height;
        float x1T2B = box.keypoints[1][0] * tex_width;
        float y1T2B = box.keypoints[1][1] * tex_height;

        // Call ExtractROI with minimal pixel array just to set ROI parameters
        // (we use GPU for the actual extraction but need the engine's ROI state)
        var dummyPixels = new Color32[1];
        engine.ExtractROI(dummyPixels, tex_width, tex_height, box);

        // Compute affine transform matrix for GPU shader
        Vector2[] points = new Vector2[]
        {
            new Vector2(1, -1),
            new Vector2(1, 1),
            new Vector2(-1, -1)
        };
        for (int i = 0; i < points.Length; ++i)
            points[i] *= scale / 2;

        float cosAngle = Mathf.Cos(angle);
        float sinAngle = Mathf.Sin(angle);
        for (int i = 0; i < points.Length; ++i)
        {
            Vector2 p = points[i];
            points[i] = new Vector2(
                p.x * cosAngle - p.y * sinAngle + xc,
                p.x * sinAngle + p.y * cosAngle + yc
            );
        }

        int resolution = estRes - 1;
        Matrix4x4 before_m = new Matrix4x4()
        {
            m00 = 0, m01 = 0, m02 = resolution, m03 = 0,
            m10 = 0, m11 = resolution, m12 = 0, m13 = 0,
            m20 = 1, m21 = 1, m22 = 1, m23 = 0,
            m30 = 0, m31 = 0, m32 = 0, m33 = 1,
        };
        Matrix4x4 after_m = new Matrix4x4()
        {
            m00 = points[0].x, m01 = points[1].x, m02 = points[2].x, m03 = 0,
            m10 = points[0].y, m11 = points[1].y, m12 = points[2].y, m13 = 0,
            m20 = 1, m21 = 1, m22 = 1, m23 = 0,
            m30 = 0, m31 = 0, m32 = 0, m33 = 1,
        };
        Matrix4x4 transfrom_m = after_m * before_m.inverse;

        // GPU compute shader dispatch
        if (computeTexture == null)
        {
            computeTexture = new RenderTexture(estRes, estRes, 32);
            computeTexture.enableRandomWrite = true;
        }
        if (kernelIndex < 0)
        {
            kernelIndex = computeShader.FindKernel("AffineTransform");
        }

        computeShader.SetTexture(kernelIndex, ID_InputTexture, input_texture);
        computeShader.SetTexture(kernelIndex, ID_OutputTexture, computeTexture);
        computeShader.SetMatrix(ID_Matrix, transfrom_m);
        computeShader.SetInt(ID_InputWidth, tex_width);
        computeShader.SetInt(ID_InputHeight, tex_height);
        computeShader.SetInt(ID_OutputWidth, estRes);
        computeShader.SetInt(ID_OutputHeight, estRes);
        computeShader.SetVector(ID_BackgroundColor, new Vector4(0, 0, 0, 1));
        computeShader.Dispatch(kernelIndex, estRes / 32 + 1, estRes / 32 + 1, 1);

        roiTexture = ToTexture2D(computeTexture, roiTexture);
        return roiTexture;
    }

    private Texture2D ToTexture2D(RenderTexture rTex, Texture2D output = null)
    {
        var org = RenderTexture.active;
        output = output ?? new Texture2D(rTex.width, rTex.height, TextureFormat.RGBA32, false);
        RenderTexture.active = rTex;
        output.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        output.Apply();
        RenderTexture.active = org;
        return output;
    }

    public static float[] ConvertDoubleArrayToFloatArray(double[] doubleArray)
    {
        float[] floatArray = new float[doubleArray.Length];
        for (int i = 0; i < doubleArray.Length; i++)
        {
            floatArray[i] = (float)doubleArray[i];
        }
        return floatArray;
    }

    #region IDisposable Support
    private bool disposedValue = false;

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            backend?.Dispose();
            backend = null;
            computeTexture?.Release();
            preprocessBuffer?.Release();
            disposedValue = true;
        }
    }

    ~AiliaMediapipePoseWorldLandmarks()
    {
        Dispose(false);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    #endregion
}
