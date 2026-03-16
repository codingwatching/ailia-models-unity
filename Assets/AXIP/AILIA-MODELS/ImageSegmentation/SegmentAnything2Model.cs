/* AILIA Unity Plugin Segment Anything 2 */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * Unity wrapper for SAM2 image segmentation.
 * Thin shell: model download, ailia initialization, Texture2D visualization.
 * All inference logic lives in Sam2Processor.
 */

using ailia;
using ailiaSDK;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class SegmentAnything2Model
{
    private const string encoderWeightPath = "image_encoder_hiera_l.onnx";
    private const string encoderProtoPath = "image_encoder_hiera_l.onnx.prototxt";
    private const string decoderWeightPath = "mask_decoder_hiera_l.onnx";
    private const string decoderProtoPath = "mask_decoder_hiera_l.onnx.prototxt";
    private const string promptWeightPath = "prompt_encoder_hiera_l.onnx";
    private const string promptProtoPath = "prompt_encoder_hiera_l.onnx.prototxt";

    private AiliaSam2Backend backend;
    private Sam2Processor processor;

    public bool modelsInitialized { get; private set; } = false;
    public bool isProcessing { get; private set; } = false;
    public bool success { get; private set; } = false;
    public Texture2D visualizedResult { get; private set; }

    private bool showClickPoints = true;
    private bool saveFrame = false;
    private bool isQuitting = false;

    public void SetBoxCoords(Rect box) => processor?.SetBoxCoords(box);
    public void AddClickPoint(int x, int y, bool negativePoint = false) => processor?.AddClickPoint(x, y, negativePoint);
    public void ResetClickPoint() => processor?.ResetClickPoint();
    public float[,] GetClickPoints(int imageHeight) => processor?.GetClickPoints(imageHeight) ?? new float[0, 2];
    public bool EmbeddingExist() => processor?.EmbeddingExist() ?? false;

    public List<ModelDownloadURL> GetModelURLs(ImageSegmentaionModels modelType)
    {
        string folder = "segment-anything-2";
        return new List<ModelDownloadURL>
        {
            new ModelDownloadURL { folder_path = folder, file_name = encoderWeightPath },
            new ModelDownloadURL { folder_path = folder, file_name = encoderProtoPath },
            new ModelDownloadURL { folder_path = folder, file_name = decoderWeightPath },
            new ModelDownloadURL { folder_path = folder, file_name = decoderProtoPath },
            new ModelDownloadURL { folder_path = folder, file_name = promptWeightPath },
            new ModelDownloadURL { folder_path = folder, file_name = promptProtoPath },
        };
    }

    public bool InitializeModels(ImageSegmentaionModels modelType, bool gpuMode)
    {
        if (modelsInitialized)
            return true;

        try
        {
            string basePath = Application.temporaryCachePath;
            string encPath = Path.Combine(basePath, encoderWeightPath);
            string decPath = Path.Combine(basePath, decoderWeightPath);
            string pmtPath = Path.Combine(basePath, promptWeightPath);

            backend = new AiliaSam2Backend();
            if (gpuMode)
            {
                backend.SetGpuMode();
            }
            backend.LoadModels(encPath, decPath, pmtPath);
            processor = new Sam2Processor(backend);
            modelsInitialized = true;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error while loading SAM model: {e.Message}\n{e.StackTrace}");
            modelsInitialized = false;
        }

        return modelsInitialized;
    }

    // image : top-to-bottom format (SAM2 native)
    public void ProcessEmbedding(Color32[] image, int imageWidth, int imageHeight)
    {
        if (!modelsInitialized || isProcessing || isQuitting)
            return;
        processor.ProcessEmbedding(image, imageWidth, imageHeight);
    }

    // image : top-to-bottom format (SAM2 native)
    // Output visualization is also top-to-bottom
    public void ProcessMask(Color32[] image, int imageWidth, int imageHeight)
    {
        if (!modelsInitialized || isProcessing || isQuitting)
            return;

        try
        {
            isProcessing = true;

            var result = processor.ProcessMask(imageWidth, imageHeight);

            if (isQuitting)
                return;

            if (visualizedResult != null)
                GameObject.Destroy(visualizedResult);

            if (result.HasMask)
            {
                Color32[] overlayPixels = Sam2Processor.ApplyMaskOverlay(
                    result.Mask, image, imageWidth, imageHeight);

                if (showClickPoints)
                {
                    float[,] coords = processor.GetClickPoints(imageHeight);
                    float[] labels = processor.GetPointLabels();
                    overlayPixels = Sam2Processor.DrawClickPoints(
                        coords, labels, overlayPixels, imageWidth, imageHeight);
                }

                visualizedResult = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);
                visualizedResult.SetPixels32(overlayPixels);
                visualizedResult.Apply();

                if (saveFrame)
                    SaveFrameAsPNG(visualizedResult);

                success = true;
            }
            else
            {
                visualizedResult = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);
                visualizedResult.SetPixels32(image);
                visualizedResult.Apply();
                success = true;
            }
        }
        catch (OperationCanceledException)
        {
            success = false;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in ProcessMask: {e.Message}\n{e.StackTrace}");
            success = false;
        }
        finally
        {
            isProcessing = false;
        }
    }

    private void SaveFrameAsPNG(Texture2D texture)
    {
        try
        {
            string directory = Path.Combine(Application.persistentDataPath, "ailiaSAM1");
            if (!Directory.Exists(directory))
                Directory.CreateDirectory(directory);
            string filePath = Path.Combine(directory, "output.png");
            File.WriteAllBytes(filePath, texture.EncodeToPNG());
            Debug.Log($"Saved output to {filePath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving output: {e.Message}\n{e.StackTrace}");
        }
    }

    public void Destroy()
    {
        isQuitting = true;
        backend?.Dispose();
    }

    public string EnvironmentName()
    {
        return backend?.EnvironmentName() ?? "";
    }
}
