/* AILIA Unity Plugin Segment Anything Sample */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * Unity wrapper for SAM2 image segmentation.
 * Delegates all computational logic to Sam2InferenceEngine (shared with tests).
 * Keeps only Unity-specific code: model initialization, visualization, async control.
 */

using ailia;
using ailiaSDK;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class SegmentAnything2Model
{
    // Ailia models
    private const string encoderWeightPath = "image_encoder_hiera_l.onnx";
    private const string encoderProtoPath = "image_encoder_hiera_l.onnx.prototxt";
    private const string decoderWeightPath = "mask_decoder_hiera_l.onnx";
    private const string decoderProtoPath = "mask_decoder_hiera_l.onnx.prototxt";
    private const string promptWeightPath = "prompt_encoder_hiera_l.onnx";
    private const string promptProtoPath = "prompt_encoder_hiera_l.onnx.prototxt";

    private int targetSize = 1024;
    private ailia.AiliaModel encoder;
    private ailia.AiliaModel decoder;
    private ailia.AiliaModel memAttention;
    private ailia.AiliaModel encoderMem;
    private ailia.AiliaModel prompt;
    private ailia.AiliaModel mlp;

    public bool modelsInitialized { get; private set; } = false;
    public bool isProcessing { get; private set; } = false;
    public bool success { get; private set; } = false;
    private bool gpuMode = false;
    private bool showClickPoints = true;

    // For async cancellation
    private CancellationTokenSource cancellationTokenSource;
    private bool isQuitting = false;

    // Visualization-related fields
    public Texture2D visualizedResult { get; private set; }
    private readonly Color32 MaskColor = new Color32(255, 0, 0, 255);
    private bool saveFrame = false;
    private float[] encoderOutput = null;
    private float[][] highResFeats = null;
    private System.Random _rng = new System.Random();

    // Shared inference engine (all computational logic)
    private readonly Sam2InferenceEngine engine = new Sam2InferenceEngine();

    public void SetBoxCoords(Rect box)
    {
        engine.SetBoxCoords(box);
    }

    public void AddClickPoint(int x, int y, bool negativePoint = false)
    {
        engine.AddClickPoint(x, y, negativePoint);
    }

    public void ResetClickPoint()
    {
        engine.ResetClickPoint();
    }

    public List<ModelDownloadURL> GetModelURLs(ImageSegmentaionModels modelType)
    {
        List<ModelDownloadURL> modelDownloadURLs = new List<ModelDownloadURL>();
        string serverFolderName = "segment-anything-2";

        modelDownloadURLs.Add(
            new ModelDownloadURL() { folder_path = serverFolderName, file_name = encoderWeightPath }
        );
        modelDownloadURLs.Add(
            new ModelDownloadURL() { folder_path = serverFolderName, file_name = encoderProtoPath }
        );
        modelDownloadURLs.Add(
            new ModelDownloadURL() { folder_path = serverFolderName, file_name = decoderWeightPath }
        );
        modelDownloadURLs.Add(
            new ModelDownloadURL() { folder_path = serverFolderName, file_name = decoderProtoPath }
        );
        modelDownloadURLs.Add(
            new ModelDownloadURL() { folder_path = serverFolderName, file_name = promptWeightPath }
        );
        modelDownloadURLs.Add(
            new ModelDownloadURL() { folder_path = serverFolderName, file_name = promptProtoPath }
        );

        return modelDownloadURLs;
    }

    // Initialize Ailia models
    public bool InitializeModels(ImageSegmentaionModels modelType, bool gpuMode)
    {
        if (modelsInitialized)
            return true;

        try
        {
            string encPath = System.IO.Path.Combine(
                Application.temporaryCachePath,
                encoderWeightPath
            );
            string encProtoPath = System.IO.Path.Combine(
                Application.temporaryCachePath,
                encoderProtoPath
            );
            string decPath = System.IO.Path.Combine(
                Application.temporaryCachePath,
                decoderWeightPath
            );
            string decProtoPath = System.IO.Path.Combine(
                Application.temporaryCachePath,
                decoderProtoPath
            );
            string pmtPath = System.IO.Path.Combine(
                Application.temporaryCachePath,
                promptWeightPath
            );
            string pmtProtoPath = System.IO.Path.Combine(
                Application.temporaryCachePath,
                promptProtoPath
            );

            encoder = new ailia.AiliaModel();
            decoder = new ailia.AiliaModel();
            prompt = new ailia.AiliaModel();

            uint memory_mode =
                ailia.Ailia.AILIA_MEMORY_REDUCE_CONSTANT
                | ailia.Ailia.AILIA_MEMORY_REDUCE_CONSTANT_WITH_INPUT_INITIALIZER
                | ailia.Ailia.AILIA_MEMORY_REUSE_INTERSTAGE;
            memory_mode = ailia.Ailia.AILIA_MEMORY_REDUCE_INTERSTAGE;
            encoder.SetMemoryMode(memory_mode);
            decoder.SetMemoryMode(memory_mode);
            prompt.SetMemoryMode(memory_mode);

            this.gpuMode = gpuMode;
            if (gpuMode)
            {
                encoder.Environment(ailia.Ailia.AILIA_ENVIRONMENT_TYPE_GPU);
                decoder.Environment(ailia.Ailia.AILIA_ENVIRONMENT_TYPE_GPU);
                prompt.Environment(ailia.Ailia.AILIA_ENVIRONMENT_TYPE_GPU);
            }

            bool encOpened = false;
            bool decOpened = false;
            bool promptOpened = false;

            encOpened = encoder.OpenFile(encProtoPath, encPath);
            decOpened = decoder.OpenFile(decProtoPath, decPath);
            promptOpened = prompt.OpenFile(pmtProtoPath, pmtPath);

            if (!encOpened || !decOpened || !promptOpened)
            {
                throw new Exception("Failed to open SAM 2 model files");
            }

            modelsInitialized = true;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error while loading SAM model: {e.Message}\n{e.StackTrace}");
            modelsInitialized = false;
        }

        return modelsInitialized;
    }

    public float[,] GetClickPoints(int imageHeight)
    {
        return engine.GetClickPoints(imageHeight);
    }

    private float[] GetPointLabels()
    {
        return engine.GetPointLabels();
    }

    // Run Embedding
    private void RunEmbedding(Color32[] image, int imgWidth, int imgHeight)
    {
        if (isQuitting || !modelsInitialized || encoder == null)
        {
            return;
        }

        float[,,,] inputTensor = engine.PreprocessImage(image, imgWidth, imgHeight, targetSize);
        float[] nchwInput = engine.Flatten4D(inputTensor);

        try
        {
            int imgIndex = encoder.FindBlobIndexByName("input_image");

            // Set encoder input shape (1x3x1024x1024)
            ailia.Ailia.AILIAShape encInputShape = new ailia.Ailia.AILIAShape();
            encInputShape.dim = 4;
            encInputShape.w = 1;
            encInputShape.z = 3;
            encInputShape.y = (uint)targetSize;
            encInputShape.x = (uint)targetSize;

            if (isQuitting || encoder == null)
            {
                return;
            }

            bool shapeSetResult = encoder.SetInputBlobShape(encInputShape, imgIndex);
            if (isQuitting || !shapeSetResult)
            {
                Debug.LogError("Failed to set encoder input shape: " + encoder.Status);
                return;
            }

            bool dataSetResult = encoder.SetInputBlobData(nchwInput, imgIndex);

            bool encResult = encoder.Update();

            if (isQuitting || encoder == null)
            {
                return;
            }

            if (!encResult)
            {
                Debug.LogError("Encoder inference failed: " + encoder.Status);
                return;
            }

            int visionFeaturesBlobIndex = encoder.FindBlobIndexByName("vision_features");
            int visionPosEnc0BlobIndex = encoder.FindBlobIndexByName("vision_pos_enc_0");
            int visionPosEnc1BlobIndex = encoder.FindBlobIndexByName("vision_pos_enc_1");
            int visionPosEnc2BlobIndex = encoder.FindBlobIndexByName("vision_pos_enc_2");
            int backboneFpn0BlobIndex = encoder.FindBlobIndexByName("backbone_fpn_0");
            int backboneFpn1BlobIndex = encoder.FindBlobIndexByName("backbone_fpn_1");
            int backboneFpn2BlobIndex = encoder.FindBlobIndexByName("backbone_fpn_2");

            if (
                visionFeaturesBlobIndex < 0
                || visionPosEnc0BlobIndex < 0
                || visionPosEnc1BlobIndex < 0
                || visionPosEnc2BlobIndex < 0
                || backboneFpn0BlobIndex < 0
                || backboneFpn1BlobIndex < 0
                || backboneFpn2BlobIndex < 0
            )
            {
                Debug.LogError("Could not find required blobs");
                return;
            }

            if (isQuitting || encoder == null)
            {
                return;
            }

            // Read encoder outputs and build EncoderOutput struct for shared engine
            ailia.Ailia.AILIAShape vfShape = encoder.GetBlobShape((uint)visionFeaturesBlobIndex);
            float[] visionFeaturesOutput = new float[vfShape.w * vfShape.z * vfShape.y * vfShape.x];
            encoder.GetBlobData(visionFeaturesOutput, visionFeaturesBlobIndex);

            ailia.Ailia.AILIAShape fp0Shape = encoder.GetBlobShape((uint)backboneFpn0BlobIndex);
            float[] backboneFpn0Output = new float[fp0Shape.w * fp0Shape.z * fp0Shape.y * fp0Shape.x];
            encoder.GetBlobData(backboneFpn0Output, backboneFpn0BlobIndex);

            ailia.Ailia.AILIAShape fp1Shape = encoder.GetBlobShape((uint)backboneFpn1BlobIndex);
            float[] backboneFpn1Output = new float[fp1Shape.w * fp1Shape.z * fp1Shape.y * fp1Shape.x];
            encoder.GetBlobData(backboneFpn1Output, backboneFpn1BlobIndex);

            ailia.Ailia.AILIAShape fp2Shape = encoder.GetBlobShape((uint)backboneFpn2BlobIndex);
            float[] backboneFpn2Output = new float[fp2Shape.w * fp2Shape.z * fp2Shape.y * fp2Shape.x];
            encoder.GetBlobData(backboneFpn2Output, backboneFpn2BlobIndex);

            // Use shared engine to prepare features
            var encOut = new EncoderOutput
            {
                VisionFeatures = visionFeaturesOutput,
                VisionFeaturesShape = new TensorShape4D((int)vfShape.w, (int)vfShape.z, (int)vfShape.y, (int)vfShape.x),
                BackboneFpn0 = backboneFpn0Output,
                Fpn0Shape = new TensorShape4D((int)fp0Shape.w, (int)fp0Shape.z, (int)fp0Shape.y, (int)fp0Shape.x),
                BackboneFpn1 = backboneFpn1Output,
                Fpn1Shape = new TensorShape4D((int)fp1Shape.w, (int)fp1Shape.z, (int)fp1Shape.y, (int)fp1Shape.x),
                BackboneFpn2 = backboneFpn2Output,
                Fpn2Shape = new TensorShape4D((int)fp2Shape.w, (int)fp2Shape.z, (int)fp2Shape.y, (int)fp2Shape.x),
            };

            var (imageEmbFlat, highResFeatsResult) = engine.PrepareEncoderFeatures(encOut);
            encoderOutput = imageEmbFlat;
            highResFeats = highResFeatsResult;
        }
        catch (Exception e)
        {
            Debug.LogError("Inference error: " + e.Message + "\n" + e.StackTrace);
        }
    }

    // Run Inference
    private (bool[][,], float[]) RunInference(
        int imgWidth,
        int imgHeight,
        float[,] pointCoords,
        float[] pointLabels
    )
    {
        if (isQuitting || !modelsInitialized || decoder == null || prompt == null)
        {
            return (new bool[0][,], new float[0]);
        }
        if (pointCoords.Length == 0)
        {
            return (new bool[0][,], new float[0]);
        }

        try
        {
            float[,] scaledCoords = engine.ApplyCoordinateScaling(pointCoords, imgHeight, imgWidth);

            // Prepare point coordinates and labels
            int pointCount = pointLabels.Length;
            float[] flattenedCoords = engine.FlattenCoords(scaledCoords, pointCount);

            float[] maskInputDummy;
            float[] masksEnable;

            int maskInputDummyChannel = 1;
            int maskInputDummyHeight = 256;
            int maskInputDummyWidth = 256;
            maskInputDummy = new float[
                maskInputDummyChannel * maskInputDummyHeight * maskInputDummyWidth
            ];
            masksEnable = new float[1] { 0f };

            int promptCoordsIndex = prompt.FindBlobIndexByName("coords");
            int promptLabelsIndex = prompt.FindBlobIndexByName("labels");
            int promptMasksIndex = prompt.FindBlobIndexByName("masks");
            int promptMasksEnabledIndex = prompt.FindBlobIndexByName("masks_enable");
            if (
                promptCoordsIndex < 0
                || promptLabelsIndex < 0
                || promptMasksIndex < 0
                || promptMasksEnabledIndex < 0
            )
            {
                Debug.LogError("Failed to find required common blob indices");
                return (new bool[0][,], new float[0]);
            }

            ailia.Ailia.AILIAShape concatPointShape = new ailia.Ailia.AILIAShape();
            concatPointShape.dim = 3;
            concatPointShape.z = 1;
            concatPointShape.y = (uint)pointCount;
            concatPointShape.x = 2;

            ailia.Ailia.AILIAShape labelsShape = new ailia.Ailia.AILIAShape();
            labelsShape.dim = 2;
            labelsShape.y = 1;
            labelsShape.x = (uint)pointCount; // points count (1)

            ailia.Ailia.AILIAShape masksShape = new ailia.Ailia.AILIAShape();
            masksShape.dim = 3;
            masksShape.z = (uint)maskInputDummyChannel;
            masksShape.y = (uint)maskInputDummyHeight;
            masksShape.x = (uint)maskInputDummyWidth;

            ailia.Ailia.AILIAShape masksEnableShape = new ailia.Ailia.AILIAShape();
            masksEnableShape.dim = 1;
            masksEnableShape.x = 1;

            bool concatPointShapeResult = prompt.SetInputBlobShape(
                concatPointShape,
                (uint)promptCoordsIndex
            );
            bool labelsShapeResult = prompt.SetInputBlobShape(labelsShape, (uint)promptLabelsIndex);

            bool masksShapeResult = prompt.SetInputBlobShape(masksShape, (uint)promptMasksIndex);

            bool masksEnabledShapeResult = prompt.SetInputBlobShape(
                masksEnableShape,
                (uint)promptMasksEnabledIndex
            );

            if (
                !concatPointShapeResult
                || !labelsShapeResult
                || !masksShapeResult
                || !masksEnabledShapeResult
            )
            {
                Debug.LogError("Failed to set input blob shapes");
                return (new bool[0][,], new float[0]);
            }

            bool setCoordsResult = prompt.SetInputBlobData(
                flattenedCoords,
                (uint)promptCoordsIndex
            );

            bool setlabelsResult = prompt.SetInputBlobData(pointLabels, (uint)promptLabelsIndex);

            bool setMaskseResult = prompt.SetInputBlobData(maskInputDummy, (uint)promptMasksIndex);

            bool setMasksEnabledResult = prompt.SetInputBlobData(
                masksEnable,
                (uint)promptMasksEnabledIndex
            );

            if (!setCoordsResult || !setlabelsResult || !setMaskseResult || !setMasksEnabledResult)
            {
                Debug.LogError("Failed to set blob data");
                return (new bool[0][,], new float[0]);
            }

            bool promptResult = prompt.Update();

            if (isQuitting || prompt == null)
            {
                return (new bool[0][,], new float[0]);
            }

            if (!promptResult)
            {
                Debug.LogError(
                    "Prompt inference failed: "
                        + prompt.Status
                        + " ["
                        + prompt.GetErrorDetail()
                        + "]"
                );
                return (new bool[0][,], new float[0]);
            }

            int sparseEmbeddingsBlobIndex = prompt.FindBlobIndexByName("sparse_embeddings");
            int denseEmbeddingsBlobIndex = prompt.FindBlobIndexByName("dense_embeddings");
            int densePeBlobIndex = prompt.FindBlobIndexByName("dense_pe");

            if (
                sparseEmbeddingsBlobIndex < 0
                || denseEmbeddingsBlobIndex < 0
                || densePeBlobIndex < 0
            )
            {
                Debug.LogError(
                    "Could not find sparse_embeddings, dense_embeddings and dense_pe indices"
                );
                return (new bool[0][,], new float[0]);
            }

            ailia.Ailia.AILIAShape sparseEmbeddingsBlobShape = prompt.GetBlobShape(
                (uint)sparseEmbeddingsBlobIndex
            );

            float[] sparseEmbeddingsOutput = new float[
                sparseEmbeddingsBlobShape.z
                    * sparseEmbeddingsBlobShape.y
                    * sparseEmbeddingsBlobShape.x
            ];

            ailia.Ailia.AILIAShape denseEmbeddingsBlobShape = prompt.GetBlobShape(
                (uint)denseEmbeddingsBlobIndex
            );

            float[] denseEmbeddingsOutput = new float[
                denseEmbeddingsBlobShape.w
                    * denseEmbeddingsBlobShape.z
                    * denseEmbeddingsBlobShape.y
                    * denseEmbeddingsBlobShape.x
            ];

            ailia.Ailia.AILIAShape densePeBlobShape = prompt.GetBlobShape((uint)densePeBlobIndex);

            float[] densePeOutput = new float[
                densePeBlobShape.w * densePeBlobShape.z * densePeBlobShape.y * densePeBlobShape.x
            ];

            prompt.GetBlobData(sparseEmbeddingsOutput, sparseEmbeddingsBlobIndex);
            prompt.GetBlobData(denseEmbeddingsOutput, denseEmbeddingsBlobIndex);
            prompt.GetBlobData(densePeOutput, densePeBlobIndex);

            int imageEmbeddingsIndex = decoder.FindBlobIndexByName("image_embeddings");
            int imagePeIndex = decoder.FindBlobIndexByName("image_pe");
            int sparsePromptEmbeddingsIndex = decoder.FindBlobIndexByName(
                "sparse_prompt_embeddings"
            );
            int densePromptEmbeddingsIndex = decoder.FindBlobIndexByName("dense_prompt_embeddings");
            int highResFeatures1Index = decoder.FindBlobIndexByName("high_res_features1");
            int highResFeatures2Index = decoder.FindBlobIndexByName("high_res_features2");

            if (
                imageEmbeddingsIndex < 0
                || imagePeIndex < 0
                || sparsePromptEmbeddingsIndex < 0
                || densePromptEmbeddingsIndex < 0
                || highResFeatures1Index < 0
                || highResFeatures2Index < 0
            )
            {
                Debug.LogError("Could not find required indices");
                return (new bool[0][,], new float[0]);
            }

            if (isQuitting || decoder == null)
            {
                return (new bool[0][,], new float[0]);
            }

            ailia.Ailia.AILIAShape imageEmbeddingsShape = new ailia.Ailia.AILIAShape();
            imageEmbeddingsShape.dim = 4;
            imageEmbeddingsShape.w = 1; // batch=1
            imageEmbeddingsShape.z = 256; // channel=256
            imageEmbeddingsShape.y = 64; // height=64
            imageEmbeddingsShape.x = 64; // width=64

            ailia.Ailia.AILIAShape imagePeShape = new ailia.Ailia.AILIAShape();
            imagePeShape.dim = 4;
            imagePeShape.w = densePeBlobShape.w; // batch=1
            imagePeShape.z = densePeBlobShape.z; // channel=256
            imagePeShape.y = densePeBlobShape.y; // height=64
            imagePeShape.x = densePeBlobShape.x; // width=64

            ailia.Ailia.AILIAShape sparsePromptEmbeddingsShape = new ailia.Ailia.AILIAShape();
            sparsePromptEmbeddingsShape.dim = 3;
            sparsePromptEmbeddingsShape.z = sparseEmbeddingsBlobShape.z;
            sparsePromptEmbeddingsShape.y = sparseEmbeddingsBlobShape.y;
            sparsePromptEmbeddingsShape.x = sparseEmbeddingsBlobShape.x;

            ailia.Ailia.AILIAShape densePromptEmbeddingsShape = new ailia.Ailia.AILIAShape();
            densePromptEmbeddingsShape.dim = 4;
            densePromptEmbeddingsShape.w = denseEmbeddingsBlobShape.w;
            densePromptEmbeddingsShape.z = denseEmbeddingsBlobShape.z;
            densePromptEmbeddingsShape.y = denseEmbeddingsBlobShape.y;
            densePromptEmbeddingsShape.x = denseEmbeddingsBlobShape.x;

            ailia.Ailia.AILIAShape highResFeatures1Shape = new ailia.Ailia.AILIAShape();
            highResFeatures1Shape.dim = 4;
            highResFeatures1Shape.w = 1; // batch=1
            highResFeatures1Shape.z = 32; // channel=256
            highResFeatures1Shape.y = 256; // height=64
            highResFeatures1Shape.x = 256; // width=64

            ailia.Ailia.AILIAShape highResFeatures2Shape = new ailia.Ailia.AILIAShape();
            highResFeatures2Shape.dim = 4;
            highResFeatures2Shape.w = 1; // batch=1
            highResFeatures2Shape.z = 64; // channel=256
            highResFeatures2Shape.y = 128; // height=128
            highResFeatures2Shape.x = 128; // width=128

            bool imageEmbeddingsShapeResult = decoder.SetInputBlobShape(
                imageEmbeddingsShape,
                (uint)imageEmbeddingsIndex
            );

            bool imagePeShapeResult = decoder.SetInputBlobShape(imagePeShape, (uint)imagePeIndex);

            bool sparsePromptEmbeddingsShapeResult = decoder.SetInputBlobShape(
                sparsePromptEmbeddingsShape,
                (uint)sparsePromptEmbeddingsIndex
            );

            bool densePromptEmbeddingsShapeResult = decoder.SetInputBlobShape(
                densePromptEmbeddingsShape,
                (uint)densePromptEmbeddingsIndex
            );

            bool highResFeatures1ShapeResult = decoder.SetInputBlobShape(
                highResFeatures1Shape,
                (uint)highResFeatures1Index
            );

            bool highResFeatures2ShapeResult = decoder.SetInputBlobShape(
                highResFeatures2Shape,
                (uint)highResFeatures2Index
            );

            if (
                !imageEmbeddingsShapeResult
                || !imagePeShapeResult
                || !sparsePromptEmbeddingsShapeResult
                || !densePromptEmbeddingsShapeResult
                || !highResFeatures1ShapeResult
                || !highResFeatures2ShapeResult
            )
            {
                Debug.LogError("Failed to set input blob shapes");
                return (new bool[0][,], new float[0]);
            }

            if (isQuitting || decoder == null)
            {
                return (new bool[0][,], new float[0]);
            }

            bool imageEmbeddingsResult = decoder.SetInputBlobData(
                encoderOutput,
                (uint)imageEmbeddingsIndex
            );

            bool imagePeResult = decoder.SetInputBlobData(densePeOutput, (uint)imagePeIndex);

            bool sparseEmbeddingsResult = decoder.SetInputBlobData(
                sparseEmbeddingsOutput,
                (uint)sparsePromptEmbeddingsIndex
            );

            bool denseEmbeddingsResult = decoder.SetInputBlobData(
                denseEmbeddingsOutput,
                (uint)densePromptEmbeddingsIndex
            );

            bool highResFeatures1Result = decoder.SetInputBlobData(
                highResFeats[0],
                (uint)highResFeatures1Index
            );

            bool highResFeatures2Result = decoder.SetInputBlobData(
                highResFeats[1],
                (uint)highResFeatures2Index
            );

            if (
                !imageEmbeddingsResult
                || !imagePeResult
                || !sparseEmbeddingsResult
                || !denseEmbeddingsResult
                || !highResFeatures1Result
                || !highResFeatures2Result
            )
            {
                Debug.LogError("Failed to set input blob data");
                return (new bool[0][,], new float[0]);
            }

            if (isQuitting || decoder == null)
            {
                return (new bool[0][,], new float[0]);
            }

            bool decoderResult = decoder.Update();

            if (isQuitting || decoder == null)
            {
                return (new bool[0][,], new float[0]);
            }

            if (!decoderResult)
            {
                Debug.LogError(
                    "Prompt inference failed: "
                        + decoder.Status
                        + " ["
                        + decoder.GetErrorDetail()
                        + "]"
                );
                return (new bool[0][,], new float[0]);
            }

            int masksBlobIndex = decoder.FindBlobIndexByName("masks");
            int iouPredBlobIndex = decoder.FindBlobIndexByName("iou_pred");
            int samTokensOutBlobIndex = decoder.FindBlobIndexByName("sam_tokens_out");
            int objectScoreLogitsBlobIndex = decoder.FindBlobIndexByName("object_score_logits");

            if (
                masksBlobIndex < 0
                || iouPredBlobIndex < 0
                || samTokensOutBlobIndex < 0
                || objectScoreLogitsBlobIndex < 0
            )
            {
                Debug.LogError("Could not find required indices");
                return (new bool[0][,], new float[0]);
            }

            if (isQuitting || decoder == null)
            {
                return (new bool[0][,], new float[0]);
            }

            ailia.Ailia.AILIAShape masksBlobShape = decoder.GetBlobShape((uint)masksBlobIndex);
            float[] masksBlobOutput = new float[
                masksBlobShape.w * masksBlobShape.z * masksBlobShape.y * masksBlobShape.x
            ];

            ailia.Ailia.AILIAShape iouPredBlobShape = decoder.GetBlobShape((uint)iouPredBlobIndex);
            float[] iouPredBlobOutput = new float[iouPredBlobShape.y * iouPredBlobShape.x];

            ailia.Ailia.AILIAShape samTokensOutBlobShape = decoder.GetBlobShape(
                (uint)samTokensOutBlobIndex
            );
            float[] samTokensOutBlobOutput = new float[
                samTokensOutBlobShape.z * samTokensOutBlobShape.y * samTokensOutBlobShape.x
            ];

            ailia.Ailia.AILIAShape objectScoreLogitsBlobShape = decoder.GetBlobShape(
                (uint)objectScoreLogitsBlobIndex
            );
            float[] objectScoreLogitsBlobOutput = new float[
                objectScoreLogitsBlobShape.y * objectScoreLogitsBlobShape.x
            ];

            bool getMasksBlobResult = decoder.GetBlobData(masksBlobOutput, (uint)masksBlobIndex);
            bool getiouPredBlobResult = decoder.GetBlobData(
                iouPredBlobOutput,
                (uint)iouPredBlobIndex
            );
            bool getsamTokensBlobResult = decoder.GetBlobData(
                samTokensOutBlobOutput,
                (uint)samTokensOutBlobIndex
            );

            bool objectScoreLogitsBlobResult = decoder.GetBlobData(
                objectScoreLogitsBlobOutput,
                (uint)objectScoreLogitsBlobIndex
            );

            if (
                !getMasksBlobResult
                || !getiouPredBlobResult
                || !getsamTokensBlobResult
                || !objectScoreLogitsBlobResult
            )
            {
                Debug.LogError("Failed to get blob data");
                return (new bool[0][,], new float[0]);
            }

            if (isQuitting || decoder == null)
            {
                return (new bool[0][,], new float[0]);
            }

            float[,,,] resized = engine.PostprocessMasks(
                engine.ReshapeTo4D(
                    masksBlobOutput,
                    (int)masksBlobShape.w,
                    (int)masksBlobShape.z,
                    (int)masksBlobShape.y,
                    (int)masksBlobShape.x
                ),
                imgHeight,
                imgWidth
            );
            bool[][,] masksResult = engine.ConvertToBoolMasks(resized);

            return (masksResult, iouPredBlobOutput);
        }
        catch (Exception e)
        {
            Debug.LogError("Inference error: " + e.Message + "\n" + e.StackTrace);
            return (new bool[0][,], new float[0]);
        }
    }

    // Overlay mask on original image
    private Texture2D CreateMaskedImage(
        bool[,] mask,
        Color32[] pixels,
        int imageWidth,
        int imageHeight
    )
    {
        Texture2D result = new Texture2D(imageWidth, imageHeight, TextureFormat.ARGB32, false);

        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        // Apply mask to original image - optimized inner loop
        for (int y = 0; y < maskHeight; y++)
        {
            int unityY = y;
            int rowOffset = unityY * imageWidth;

            for (int x = 0; x < maskWidth; x++)
            {
                int pixelIndex = rowOffset + x;

                if (pixelIndex >= 0 && pixelIndex < pixels.Length && mask[y, x])
                {
                    Color32 originalColor = pixels[pixelIndex];
                    pixels[pixelIndex] = new Color32(
                        (byte)Mathf.Lerp(originalColor.r, MaskColor.r, 0.4f),
                        (byte)Mathf.Lerp(originalColor.g, MaskColor.g, 0.4f),
                        (byte)Mathf.Lerp(originalColor.b, MaskColor.b, 0.4f),
                        MaskColor.a
                    );
                }
            }
        }

        result.SetPixels32(pixels);
        result.Apply();
        return result;
    }

    private Texture2D CreateEmptyMaskedImage(Color32[] pixels, int imageWidth, int imageHeight)
    {
        Texture2D result = new Texture2D(imageWidth, imageHeight, TextureFormat.ARGB32, false);

        // Apply mask to original image - optimized inner loop
        for (int y = 0; y < imageHeight; y++)
        {
            int unityY = y;
            int rowOffset = unityY * imageWidth;

            for (int x = 0; x < imageWidth; x++)
            {
                int pixelIndex = rowOffset + x;

                if (pixelIndex >= 0 && pixelIndex < pixels.Length)
                {
                    Color32 originalColor = pixels[pixelIndex];
                    pixels[pixelIndex] = originalColor;
                }
            }
        }

        result.SetPixels32(pixels);
        result.Apply();
        return result;
    }

    // Visualize clicked points
    private Texture2D DrawClickPoints(float[,] coords, float[] labels, Texture2D image)
    {
        Texture2D result = new Texture2D(image.width, image.height, image.format, false);
        Graphics.CopyTexture(image, result);
        Color32[] pixels = result.GetPixels32();

        int numPoints = coords.GetLength(0);
        int markerSize = 15;

        for (int i = 0; i < numPoints; i++)
        {
            if (labels[i] >= 2)
            {
                continue;
            }

            int px = Mathf.Clamp((int)coords[i, 0], 0, image.width - 1);
            int origY = (int)coords[i, 1];

            // Convert to Unity coordinates
            int py = origY;
            py = Mathf.Clamp(py, 0, image.height - 1);

            Color32 markerColor =
                labels[i] == 1 ? new Color32(0, 255, 0, 255) : new Color32(0, 0, 255, 255);

            // Draw marker with bounds checking in the loop
            for (int dy = -markerSize; dy <= markerSize; dy++)
            {
                for (int dx = -markerSize; dx <= markerSize; dx++)
                {
                    if (Math.Abs(dx) == Math.Abs(dy))
                    {
                        int nx = px + dx;
                        int ny = py + dy;

                        if (nx >= 0 && nx < image.width && ny >= 0 && ny < image.height)
                        {
                            int idx = ny * image.width + nx;
                            pixels[idx] = markerColor;
                        }
                    }
                }
            }
        }

        result.SetPixels32(pixels);
        result.Apply();
        return result;
    }

    private float GetScale(int texWidth, int texHeight)
    {
        return engine.GetScale(texWidth, texHeight);
    }

    // Calculate embedding of input image
    // image : top bottom format
    public void ProcessEmbedding(Color32[] image, int imageWidth, int imageHeight)
    {
        if (!modelsInitialized || isProcessing || isQuitting)
        {
            return;
        }
        RunEmbedding(image, imageWidth, imageHeight);
    }

    // Check embedding exist
    public bool EmbeddingExist()
    {
        return encoderOutput != null;
    }

    // Calculate mask of input image
    // image : top bottom format
    public void ProcessMask(Color32[] image, int imageWidth, int imageHeight)
    {
        if (!modelsInitialized || isProcessing || isQuitting)
        {
            return;
        }

        try
        {
            isProcessing = true;

            // Set up point coords for inference
            float[,] coords = GetClickPoints(imageHeight);
            float[] labels = GetPointLabels();

            string coordsLog = $"Points input ({labels.Length}): ";
            for (int i = 0; i < labels.Length; i += 1)
            {
                coordsLog += $"({coords[i, 0]},{-coords[i, 1] - imageHeight + 1})[{labels[i]}]";
            }

            var (masks, scores) = RunInference(imageWidth, imageHeight, coords, labels);

            if (isQuitting)
            {
                return;
            }

            if (masks != null && masks.Length > 0 && scores != null && scores.Length > 0)
            {
                // Find best mask using shared engine
                int bestMaskIndex = engine.FindBestMaskIndex(scores);

                // TODO: remove for visualisation
                if (visualizedResult != null)
                {
                    GameObject.Destroy(visualizedResult);
                }

                visualizedResult = CreateMaskedImage(
                    masks[bestMaskIndex],
                    image,
                    imageWidth,
                    imageHeight
                );

                // Only draw click points if the showClickPoints flag is enabled
                if (showClickPoints)
                {
                    visualizedResult = DrawClickPoints(coords, labels, visualizedResult);
                }

                if (saveFrame)
                {
                    SaveFrameAsPNG(visualizedResult);
                }

                success = true;
            }
            else
            {
                visualizedResult = CreateEmptyMaskedImage(image, imageWidth, imageHeight);
                success = true;
            }
        }
        catch (OperationCanceledException)
        {
            // Silent cancellation
            success = false;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in ProcessCurrentFrame: {e.Message}\n{e.StackTrace}");
            success = false;
        }
        finally
        {
            isProcessing = false;
        }
    }

    private string CreateOutputDirectory()
    {
        try
        {
            // Create base directory using Application.persistentDataPath for cross-platform support
            string directory = System.IO.Path.Combine(Application.persistentDataPath, "ailiaSAM1");

            if (!System.IO.Directory.Exists(directory))
            {
                System.IO.Directory.CreateDirectory(directory);
                Debug.Log($"Created output directory: {directory}");
            }

            return directory;
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to create output directory: {e.Message}");
        }

        return Application.persistentDataPath;
    }

    private void SaveFrameAsPNG(Texture2D texture)
    {
        try
        {
            string fileNamePrefix = "output";
            // Create path to output directory
            string directory = System.IO.Path.Combine(
                Application.persistentDataPath,
                CreateOutputDirectory()
            );

            // Ensure directory exists
            if (!System.IO.Directory.Exists(directory))
            {
                System.IO.Directory.CreateDirectory(directory);
            }

            // Generate filename with frame number and optional timestamp
            string fileName = $"{fileNamePrefix}.png";

            // Full path to file
            string filePath = System.IO.Path.Combine(directory, fileName);

            // Convert texture to PNG bytes
            byte[] bytes = texture.EncodeToPNG();

            // Save synchronously to ensure completion
            System.IO.File.WriteAllBytes(filePath, bytes);

            Debug.Log($"Saved output to {fileName}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving output: {e.Message}\n{e.StackTrace}");
        }
    }

    public void Destroy()
    {
        encoder.Close();
        decoder.Close();
        prompt.Close();
    }

    public string EnvironmentName()
    {
        return encoder.EnvironmentName();
    }
}
