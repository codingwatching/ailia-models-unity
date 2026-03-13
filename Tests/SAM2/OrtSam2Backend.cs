/* SAM2 OnnxRuntime Backend */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */

using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

/// <summary>
/// SAM2 inference backend using Microsoft.ML.OnnxRuntime.
/// </summary>
public class OrtSam2Backend : ISam2Backend
{
        private InferenceSession? encoderSession;
        private InferenceSession? decoderSession;
        private InferenceSession? promptSession;

        public void LoadModels(string encoderPath, string decoderPath, string promptEncoderPath)
        {
            encoderSession = new InferenceSession(encoderPath);
            decoderSession = new InferenceSession(decoderPath);
            promptSession = new InferenceSession(promptEncoderPath);
        }

        public EncoderOutput RunEncoder(float[] nchwInput)
        {
            if (encoderSession == null) throw new InvalidOperationException("Models not loaded");

            var tensor = new DenseTensor<float>(nchwInput, new[] { 1, 3, 1024, 1024 });
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(
                encoderSession.InputMetadata.Keys.First(), tensor) };

            using var results = encoderSession.Run(inputs);

            var map = results.ToDictionary(r => r.Name, r => r.AsTensor<float>().ToArray());
            var shapes = results.ToDictionary(r => r.Name, r => r.AsTensor<float>().Dimensions.ToArray());

            static TensorShape4D ToShape(int[] dims) =>
                new TensorShape4D(dims[0], dims[1], dims[2], dims[3]);

            return new EncoderOutput
            {
                VisionFeatures = map["vision_features"],
                VisionFeaturesShape = ToShape(shapes["vision_features"]),
                BackboneFpn0 = map["backbone_fpn_0"],
                Fpn0Shape = ToShape(shapes["backbone_fpn_0"]),
                BackboneFpn1 = map["backbone_fpn_1"],
                Fpn1Shape = ToShape(shapes["backbone_fpn_1"]),
                BackboneFpn2 = map["backbone_fpn_2"],
                Fpn2Shape = ToShape(shapes["backbone_fpn_2"]),
            };
        }

        public PromptEncoderOutput RunPromptEncoder(
            float[] flatCoords, int pointCount, float[] labels,
            float[] maskInput, float masksEnable)
        {
            if (promptSession == null) throw new InvalidOperationException("Models not loaded");

            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("coords",
                    new DenseTensor<float>(flatCoords, new[] { 1, pointCount, 2 })),
                NamedOnnxValue.CreateFromTensor("labels",
                    new DenseTensor<int>(
                        labels.Select(l => (int)l).ToArray(), new[] { 1, pointCount })),
                NamedOnnxValue.CreateFromTensor("masks",
                    new DenseTensor<float>(maskInput, new[] { 1, 256, 256 })),
                NamedOnnxValue.CreateFromTensor("masks_enable",
                    new DenseTensor<int>(new[] { (int)masksEnable }, new[] { 1 })),
            };

            using var results = promptSession.Run(inputs);
            var map = results.ToDictionary(r => r.Name, r => r);

            return new PromptEncoderOutput
            {
                SparseEmbeddings = map["sparse_embeddings"].AsTensor<float>().ToArray(),
                SparseShape = map["sparse_embeddings"].AsTensor<float>().Dimensions.ToArray(),
                DenseEmbeddings = map["dense_embeddings"].AsTensor<float>().ToArray(),
                DenseShape = map["dense_embeddings"].AsTensor<float>().Dimensions.ToArray(),
                DensePe = map["dense_pe"].AsTensor<float>().ToArray(),
                DensePeShape = map["dense_pe"].AsTensor<float>().Dimensions.ToArray(),
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
            if (decoderSession == null) throw new InvalidOperationException("Models not loaded");

            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("image_embeddings",
                    new DenseTensor<float>(imageEmbeddings, new[] { 1, 256, 64, 64 })),
                NamedOnnxValue.CreateFromTensor("image_pe",
                    new DenseTensor<float>(imagePe, imagePeShape)),
                NamedOnnxValue.CreateFromTensor("sparse_prompt_embeddings",
                    new DenseTensor<float>(sparseEmbeddings, sparseShape)),
                NamedOnnxValue.CreateFromTensor("dense_prompt_embeddings",
                    new DenseTensor<float>(denseEmbeddings, denseShape)),
                NamedOnnxValue.CreateFromTensor("high_res_features1",
                    new DenseTensor<float>(highResFeatures1, new[] { 1, 32, 256, 256 })),
                NamedOnnxValue.CreateFromTensor("high_res_features2",
                    new DenseTensor<float>(highResFeatures2, new[] { 1, 64, 128, 128 })),
            };

            using var results = decoderSession.Run(inputs);
            var map = results.ToDictionary(r => r.Name, r => r);

            return new DecoderOutput
            {
                Masks = map["masks"].AsTensor<float>().ToArray(),
                MasksShape = map["masks"].AsTensor<float>().Dimensions.ToArray(),
                IouPred = map["iou_pred"].AsTensor<float>().ToArray(),
            };
        }

        public void Dispose()
        {
            encoderSession?.Dispose();
            decoderSession?.Dispose();
            promptSession?.Dispose();
        }
}
