/* SAM2 Inference Backend Interface */
/* Copyright 2025 AXELL CORPORATION and ax Inc. */
/*
 * Abstraction layer for SAM2 ONNX model inference.
 * Allows testing with different backends (ORT, ailia, etc.)
 * without changing the inference pipeline logic.
 */

using System;

namespace Sam2MaskTests
{
    /// <summary>
    /// Shape descriptor for a 4D tensor (N, C, H, W).
    /// </summary>
    public struct TensorShape4D
    {
        public int N, C, H, W;
        public int TotalSize => N * C * H * W;
        public TensorShape4D(int n, int c, int h, int w) { N = n; C = c; H = h; W = w; }
    }

    /// <summary>
    /// Encoder output containing all backbone features.
    /// </summary>
    public struct EncoderOutput
    {
        public float[] VisionFeatures;
        public TensorShape4D VisionFeaturesShape;
        public float[] BackboneFpn0, BackboneFpn1, BackboneFpn2;
        public TensorShape4D Fpn0Shape, Fpn1Shape, Fpn2Shape;
    }

    /// <summary>
    /// Prompt encoder output.
    /// </summary>
    public struct PromptEncoderOutput
    {
        public float[] SparseEmbeddings;
        public int[] SparseShape; // [batch, tokens, dim]
        public float[] DenseEmbeddings;
        public int[] DenseShape; // [batch, channels, h, w]
        public float[] DensePe;
        public int[] DensePeShape; // [batch, channels, h, w]
    }

    /// <summary>
    /// Mask decoder output.
    /// </summary>
    public struct DecoderOutput
    {
        public float[] Masks;
        public int[] MasksShape; // [batch, num_masks, h, w]
        public float[] IouPred;
    }

    /// <summary>
    /// Backend-agnostic interface for SAM2 model inference.
    /// Implementations: OrtSam2Backend (OnnxRuntime), AiliaSam2Backend (ailia SDK)
    /// </summary>
    public interface ISam2Backend : IDisposable
    {
        /// <summary>
        /// Load the three SAM2 ONNX models.
        /// </summary>
        /// <param name="encoderPath">Path to image_encoder_hiera_l.onnx</param>
        /// <param name="decoderPath">Path to mask_decoder_hiera_l.onnx</param>
        /// <param name="promptEncoderPath">Path to prompt_encoder_hiera_l.onnx</param>
        void LoadModels(string encoderPath, string decoderPath, string promptEncoderPath);

        /// <summary>
        /// Run the image encoder on preprocessed input (1, 3, 1024, 1024).
        /// </summary>
        EncoderOutput RunEncoder(float[] nchwInput);

        /// <summary>
        /// Run the prompt encoder with point coordinates and labels.
        /// </summary>
        /// <param name="flatCoords">Flattened scaled coordinates [x0, y0, x1, y1, ...]</param>
        /// <param name="pointCount">Number of points</param>
        /// <param name="labels">Point labels (1=foreground, 0=background, 2/3=box)</param>
        /// <param name="maskInput">Mask input (1x256x256), zeros if no prior mask</param>
        /// <param name="masksEnable">Whether mask input is valid (0 or 1)</param>
        PromptEncoderOutput RunPromptEncoder(
            float[] flatCoords, int pointCount, float[] labels,
            float[] maskInput, float masksEnable);

        /// <summary>
        /// Run the mask decoder.
        /// </summary>
        DecoderOutput RunDecoder(
            float[] imageEmbeddings,
            float[] imagePe, int[] imagePeShape,
            float[] sparseEmbeddings, int[] sparseShape,
            float[] denseEmbeddings, int[] denseShape,
            float[] highResFeatures1, // (1, 32, 256, 256)
            float[] highResFeatures2  // (1, 64, 128, 128)
        );
    }
}
