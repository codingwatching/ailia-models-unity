# Text To Speech (ailia SDK for Unity)

This folder provides neural TTS pipelines. The scene is `TextToSpeech/TextToSpeechSample.unity`, controller `TextToSpeech/AiliaVoiceSample.cs`.

## Supported Models

- Tacotron2 + WaveGlow (English)
  - Purpose: Sequence‑to‑mel + vocoder synthesis.
  - Source: `./AiliaVoiceSample.cs`
  - Script behavior: Opens encoder/decoder/postnet/waveglow ONNX and synthesizes speech from text.

- GPT‑SoVITS (Japanese/English/Chinese)
  - Purpose: High‑fidelity TTS with SoVITS and GPT modules.
  - Source: `./AiliaVoiceSample.cs`
  - Script behavior: Opens T2S encoder/FS decoder/decoder + VITS + cnhubert encoder; uses G2P for JA/EN/ZH.

- GPT‑SoVITS v2 (Japanese/English/Chinese)
  - Purpose: Improved fidelity TTS with SoVITS v2 and GPT modules.
  - Source: `./AiliaVoiceSample.cs`
  - Script behavior: Opens T2S encoder/FS decoder/decoder + VITS + cnhubert encoder; uses G2P for JA/EN/ZH. Optional Chinese BERT support.

- GPT‑SoVITS v3 (Japanese/English/Chinese)
  - Purpose: High‑fidelity TTS with SoVITS v3, VQ, CFM, and BigVGAN modules.
  - Source: `./AiliaVoiceSample.cs`
  - Script behavior: Opens T2S encoder/FS decoder/decoder + cnhubert + VQ + CFM + BigVGAN; uses G2P for JA/EN/ZH. Optional Chinese BERT support.

- GPT‑SoVITS v2 Pro (Japanese/English/Chinese)
  - Purpose: Enhanced TTS with SoVITS v2 Pro, VITS, and SV modules.
  - Source: `./AiliaVoiceSample.cs`
  - Script behavior: Opens T2S encoder/FS decoder/decoder + cnhubert + VITS + SV; uses G2P for JA/EN/ZH. Optional Chinese BERT support.

## Model Files and Paths

- Models download into `Application.temporaryCachePath` under `tacotron2`, `gpt-sovits`, `gpt-sovits-v2`, `gpt-sovits-v3`, `gpt-sovits-v2-pro`, `g2p_en`, `g2p_cn`, and `g2pw`.

## Reusing in Your Own Code

- Initialize `AiliaVoice` pipeline:
  - For Tacotron2: `OpenModel(encoder, decoder_iter, postnet, waveglow, null, AILIA_VOICE_MODEL_TYPE_TACOTRON2, ...)`.
  - For GPT‑SoVITS v1: `OpenGPTSoVITSV1ModelFile(t2s_encoder, t2s_fsdec, t2s_sdec, vits, cnhubert)`.
  - For GPT‑SoVITS v2: `OpenGPTSoVITSV2ModelFile(t2s_encoder, t2s_fsdec, t2s_sdec, vits, cnhubert, chinese_bert, vocab)`.
  - For GPT‑SoVITS v3: `OpenGPTSoVITSV3ModelFile(t2s_encoder, t2s_fsdec, t2s_sdec, cnhubert, vq, cfm, bigvgan, chinese_bert, vocab)`.
  - For GPT‑SoVITS v2 Pro: `OpenGPTSoVITSV2ProModelFile(t2s_encoder, t2s_fsdec, t2s_sdec, cnhubert, vits, sv, chinese_bert, vocab)`.
- Optional G2P:
  - Japanese: `G2P(text, AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA)`; English: `..._EN`; Chinese: `..._ZH`.

## Source Links

- Controller: `./AiliaVoiceSample.cs`
