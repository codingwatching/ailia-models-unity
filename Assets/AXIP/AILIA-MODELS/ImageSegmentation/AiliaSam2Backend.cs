/* SAM2 ailia SDK Backend */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * SAM2 inference backend using ailia SDK.
 * Implements ISam2Backend for use in both Unity and standalone tests.
 *
 * Unity: included directly via the project
 * Tests: linked via <Compile Include="...AiliaSam2Backend.cs" Link="..." />
 *        Requires ailia-csharp wrapper files and native library (libailia.so / ailia.dll)
 */

using System;

/// <summary>
/// SAM2 inference backend using ailia SDK.
/// Mirrors the inference flow in SegmentAnything2Model.cs exactly.
/// </summary>
public class AiliaSam2Backend : ISam2Backend
{
    private ailia.AiliaModel encoder;
    private ailia.AiliaModel decoder;
    private ailia.AiliaModel prompt;

    public void LoadModels(string encoderPath, string decoderPath, string promptEncoderPath)
    {
        encoder = new ailia.AiliaModel();
        decoder = new ailia.AiliaModel();
        prompt = new ailia.AiliaModel();

        // Use reduced memory mode (same as SegmentAnything2Model.cs)
        uint memoryMode = ailia.Ailia.AILIA_MEMORY_REDUCE_INTERSTAGE;
        encoder.SetMemoryMode(memoryMode);
        decoder.SetMemoryMode(memoryMode);
        prompt.SetMemoryMode(memoryMode);

        // Use prototxt files alongside ONNX files
        string encoderProto = encoderPath + ".prototxt";
        string decoderProto = decoderPath + ".prototxt";
        string promptProto = promptEncoderPath + ".prototxt";

        if (!encoder.OpenFile(encoderProto, encoderPath))
            throw new Exception("Failed to open encoder: " + encoder.Status);
        if (!decoder.OpenFile(decoderProto, decoderPath))
            throw new Exception("Failed to open decoder: " + decoder.Status);
        if (!prompt.OpenFile(promptProto, promptEncoderPath))
            throw new Exception("Failed to open prompt encoder: " + prompt.Status);
    }

    public EncoderOutput RunEncoder(float[] nchwInput)
    {
        if (encoder == null) throw new InvalidOperationException("Models not loaded");

        int imgIndex = encoder.FindBlobIndexByName("input_image");

        var encInputShape = new ailia.Ailia.AILIAShape
        {
            dim = 4, w = 1, z = 3, y = 1024, x = 1024
        };
        if (!encoder.SetInputBlobShape(encInputShape, imgIndex))
            throw new Exception("Failed to set encoder input shape: " + encoder.Status);
        if (!encoder.SetInputBlobData(nchwInput, imgIndex))
            throw new Exception("Failed to set encoder input data: " + encoder.Status);
        if (!encoder.Update())
            throw new Exception("Encoder inference failed: " + encoder.Status);

        // Read all outputs
        var result = new EncoderOutput();

        result.VisionFeatures = ReadBlob(encoder, "vision_features", out var vfShape);
        result.VisionFeaturesShape = new TensorShape4D(
            (int)vfShape.w, (int)vfShape.z, (int)vfShape.y, (int)vfShape.x);

        result.BackboneFpn0 = ReadBlob(encoder, "backbone_fpn_0", out var fp0Shape);
        result.Fpn0Shape = new TensorShape4D(
            (int)fp0Shape.w, (int)fp0Shape.z, (int)fp0Shape.y, (int)fp0Shape.x);

        result.BackboneFpn1 = ReadBlob(encoder, "backbone_fpn_1", out var fp1Shape);
        result.Fpn1Shape = new TensorShape4D(
            (int)fp1Shape.w, (int)fp1Shape.z, (int)fp1Shape.y, (int)fp1Shape.x);

        result.BackboneFpn2 = ReadBlob(encoder, "backbone_fpn_2", out var fp2Shape);
        result.Fpn2Shape = new TensorShape4D(
            (int)fp2Shape.w, (int)fp2Shape.z, (int)fp2Shape.y, (int)fp2Shape.x);

        return result;
    }

    public PromptEncoderOutput RunPromptEncoder(
        float[] flatCoords, int pointCount, float[] labels,
        float[] maskInput, float masksEnable)
    {
        if (prompt == null) throw new InvalidOperationException("Models not loaded");

        int coordsIdx = prompt.FindBlobIndexByName("coords");
        int labelsIdx = prompt.FindBlobIndexByName("labels");
        int masksIdx = prompt.FindBlobIndexByName("masks");
        int masksEnableIdx = prompt.FindBlobIndexByName("masks_enable");

        // Set shapes (same as SegmentAnything2Model.cs)
        prompt.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 3, z = 1, y = (uint)pointCount, x = 2
        }, (uint)coordsIdx);

        prompt.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 2, y = 1, x = (uint)pointCount
        }, (uint)labelsIdx);

        prompt.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 3, z = 1, y = 256, x = 256
        }, (uint)masksIdx);

        prompt.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 1, x = 1
        }, (uint)masksEnableIdx);

        // Set data
        prompt.SetInputBlobData(flatCoords, (uint)coordsIdx);
        prompt.SetInputBlobData(labels, (uint)labelsIdx);
        prompt.SetInputBlobData(maskInput, (uint)masksIdx);
        prompt.SetInputBlobData(new float[] { masksEnable }, (uint)masksEnableIdx);

        if (!prompt.Update())
            throw new Exception("Prompt encoder failed: " + prompt.Status);

        // Read outputs
        var sparseEmb = ReadBlob(prompt, "sparse_embeddings", out var sparseShape);
        var denseEmb = ReadBlob(prompt, "dense_embeddings", out var denseShape);
        var densePe = ReadBlob(prompt, "dense_pe", out var peShape);

        return new PromptEncoderOutput
        {
            SparseEmbeddings = sparseEmb,
            SparseShape = new[] { (int)sparseShape.z, (int)sparseShape.y, (int)sparseShape.x },
            DenseEmbeddings = denseEmb,
            DenseShape = new[] { (int)denseShape.w, (int)denseShape.z, (int)denseShape.y, (int)denseShape.x },
            DensePe = densePe,
            DensePeShape = new[] { (int)peShape.w, (int)peShape.z, (int)peShape.y, (int)peShape.x },
        };
    }

    public DecoderOutput RunDecoder(
        float[] imageEmbeddings,
        float[] imagePe, int[] imagePeShape,
        float[] sparseEmbeddings, int[] sparseShape,
        float[] denseEmbeddings, int[] denseShape,
        float[] highResFeatures1,
        float[] highResFeatures2)
    {
        if (decoder == null) throw new InvalidOperationException("Models not loaded");

        int imageEmbIdx = decoder.FindBlobIndexByName("image_embeddings");
        int imagePeIdx = decoder.FindBlobIndexByName("image_pe");
        int sparseIdx = decoder.FindBlobIndexByName("sparse_prompt_embeddings");
        int denseIdx = decoder.FindBlobIndexByName("dense_prompt_embeddings");
        int hr1Idx = decoder.FindBlobIndexByName("high_res_features1");
        int hr2Idx = decoder.FindBlobIndexByName("high_res_features2");

        // Set shapes
        decoder.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 4, w = 1, z = 256, y = 64, x = 64
        }, (uint)imageEmbIdx);

        decoder.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 4,
            w = (uint)imagePeShape[0], z = (uint)imagePeShape[1],
            y = (uint)imagePeShape[2], x = (uint)imagePeShape[3]
        }, (uint)imagePeIdx);

        decoder.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 3,
            z = (uint)sparseShape[0], y = (uint)sparseShape[1], x = (uint)sparseShape[2]
        }, (uint)sparseIdx);

        decoder.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 4,
            w = (uint)denseShape[0], z = (uint)denseShape[1],
            y = (uint)denseShape[2], x = (uint)denseShape[3]
        }, (uint)denseIdx);

        decoder.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 4, w = 1, z = 32, y = 256, x = 256
        }, (uint)hr1Idx);

        decoder.SetInputBlobShape(new ailia.Ailia.AILIAShape
        {
            dim = 4, w = 1, z = 64, y = 128, x = 128
        }, (uint)hr2Idx);

        // Set data
        decoder.SetInputBlobData(imageEmbeddings, (uint)imageEmbIdx);
        decoder.SetInputBlobData(imagePe, (uint)imagePeIdx);
        decoder.SetInputBlobData(sparseEmbeddings, (uint)sparseIdx);
        decoder.SetInputBlobData(denseEmbeddings, (uint)denseIdx);
        decoder.SetInputBlobData(highResFeatures1, (uint)hr1Idx);
        decoder.SetInputBlobData(highResFeatures2, (uint)hr2Idx);

        if (!decoder.Update())
            throw new Exception("Decoder inference failed: " + decoder.Status);

        // Read outputs
        var masks = ReadBlob(decoder, "masks", out var masksShape);
        var iou = ReadBlob(decoder, "iou_pred", out _);

        return new DecoderOutput
        {
            Masks = masks,
            MasksShape = new[]
            {
                (int)masksShape.w, (int)masksShape.z,
                (int)masksShape.y, (int)masksShape.x
            },
            IouPred = iou,
        };
    }

    private float[] ReadBlob(ailia.AiliaModel model, string name,
        out ailia.Ailia.AILIAShape shape)
    {
        int idx = model.FindBlobIndexByName(name);
        if (idx < 0)
            throw new Exception($"Blob '{name}' not found");

        shape = model.GetBlobShape((uint)idx);
        int size = (int)(shape.w * shape.z * shape.y * shape.x);
        float[] data = new float[size];
        model.GetBlobData(data, idx);
        return data;
    }

    public void Dispose()
    {
        encoder?.Close();
        decoder?.Close();
        prompt?.Close();
    }
}
