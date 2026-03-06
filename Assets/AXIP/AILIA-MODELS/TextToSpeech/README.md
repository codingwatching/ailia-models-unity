# Text To Speech (ailia SDK for Unity)

This folder provides neural TTS pipelines. The scene is `TextToSpeech/TextToSpeechSample.unity`, controller `TextToSpeech/AiliaVoiceSample.cs`.

## Supported Models

- Tacotron2 + WaveGlow (English)
  - Purpose: Sequence‑to‑mel + vocoder synthesis.
  - Source: `./AiliaVoiceSample.cs`
  - Script behavior: Opens encoder/decoder/postnet/waveglow ONNX and synthesizes speech from text.

- GPT‑SoVITS (Japanese/English)
  - Purpose: High‑fidelity TTS with SoVITS and GPT modules.
  - Source: `./AiliaVoiceSample.cs`
  - Script behavior: Opens T2S encoder/FS decoder/decoder + VITS + cnhubert encoder; uses G2P for JA/EN (optional `g2p_en`).

## Model Files and Paths

- Models download into `Application.temporaryCachePath` under `tacotron2`, `gpt-sovits`, and `g2p_en`.

## Reusing in Your Own Code

- Initialize `AiliaVoice` pipeline:
  - For Tacotron2: `OpenModel(encoder, decoder_iter, postnet, waveglow, null, AILIA_VOICE_MODEL_TYPE_TACOTRON2, ...)`.
  - For GPT‑SoVITS: `OpenModel(t2s_encoder, t2s_fsdec, t2s_sdec, vits, cnhubert, AILIA_VOICE_MODEL_TYPE_GPT_SOVITS, ...)`.
- Optional G2P:
  - Japanese: `G2P(text, AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA)`; English: `..._EN`.

## Source Links

- Controller: `./AiliaVoiceSample.cs`

