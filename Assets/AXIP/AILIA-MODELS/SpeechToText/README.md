# Speech To Text (ailia SDK for Unity)

This folder demonstrates streaming and one‑shot speech recognition. The scene is `SpeechToText/SpeechToTextSample.unity`, controlled by `SpeechToText/AiliaSpeechToTextSample.cs`.

## Supported AI Models

- Whisper (tiny/small/medium/turbo)
  - Purpose: Multilingual speech recognition; turbo variant is a large v3 model.
  - Source: `./AiliaSpeechToTextSample.cs`
  - Script behavior: Downloads the encoder/decoder ONNX (and weights PB for turbo), opens with `AiliaSpeechModel.Open(...)`, and optionally enables live streaming (`AILIA_SPEECH_FLAG_LIVE`). Uses Silero VAD for endpointing.

- SenseVoice Small
  - Purpose: Alternative end‑to‑end ASR model.
  - Source: `./AiliaSpeechToTextSample.cs`
  - Script behavior: Opens SenseVoice encoder ONNX and decoder `.model` pair through the same ailia speech API.

## Model Files and Paths

- Models download into `Application.temporaryCachePath` under `whisper` or `sensevoice`.
- Silero VAD models (`silero_vad.onnx` or `silero_vad_v6_2.onnx`) are also downloaded and opened via `OpenVad(..., AILIA_SPEECH_VAD_TYPE_SILERO)`.

## Reusing in Your Own Code

- Microphone input:
  - Use `AiliaMicrophone` to capture PCM each frame: `GetPcm(ref channels, ref frequency)`.
- Speech model init:
  - Get environment id: `ailia_speech.GetEnvironmentId(gpuMode)`.
  - `ailia_speech.Open(encoderPath, decoderPath, envId, memoryMode, modelType, task, flags, language)`.
  - Optional: `ailia_speech.OpenVad(vadPath, AILIA_SPEECH_VAD_TYPE_SILERO)`.
- Streaming transcribe loop:
  - Feed audio chunks with `Transcribe(wave, freq, ch, isFinalize)`, poll intermediate text via `GetIntermediateText()`, and fetch final results via `GetResults()`.

## Source Links

- Sample controller: `./AiliaSpeechToTextSample.cs`

