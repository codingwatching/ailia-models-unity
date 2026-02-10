# Audio Processing (ailia SDK for Unity)

This folder contains Unity samples demonstrating real‑time audio processing with the ailia SDK. The sample scene is `AudioProcessing/AudioProcessingSample.unity`, driven by `AudioProcessing/AiliaAudioProcessingSample.cs`.

Below are the AI models supported here, what they are used for, and links to the corresponding scripts. Notes are written to help you reuse these components in your own code.

## Supported AI Models

- Silero VAD v4
  - Purpose: Voice Activity Detection (speech/silence) in real time at 16 kHz.
  - Source: `./AiliaSileroVad.cs`
  - Script behavior: Loads the Silero VAD ONNX via `AiliaModel`, supports GPU, maintains internal RNN state, and resamples incoming PCM to 16 kHz. The `VAD(float[] pcm, int channels, int sampleRate)` method consumes mono PCM and returns per‑sample confidence along with the processed PCM window. Version selection (v4 vs v6.2) is automatic based on the model’s input blob layout.

- Silero VAD v6.2
  - Purpose: Updated Silero VAD with context buffer, improved latency/accuracy trade‑off.
  - Source: `./AiliaSileroVad.cs`
  - Script behavior: Same class as above. When opened with the v6.2 ONNX, the class switches to the v6 input/state sizes and uses a context window for streaming inference. Use the same `VAD` API; the class handles buffering, state carry‑over, and confidence filling.

- Retrieval‑based Voice Conversion (RVC)
  - Purpose: Convert the speaker’s timbre while preserving linguistic content.
  - Sources: `./AiliaRvc.cs`
  - Script behavior: Loads two models with `OpenFile(...)`: HuBERT (`hubert_base.onnx`) to extract features and an RVC generator (e.g., `AISO-HOWATTO.onnx`) to synthesize the converted waveform. Internally, it:
    - Resamples input to 16 kHz and pads (`T_PAD`).
    - Runs HuBERT, interpolates features to the generator’s frame rate.
    - Runs the VC model with noise conditioning, trims pad, clips amplitude, and returns a mono `AudioClip` (default 40 kHz; override via `SetTargetSmaplingRate`).
    - Provides both synchronous `Process(AudioClip)` and asynchronous `AsyncProcess/AsyncGetResult()` workflows. GPU is supported via environment selection.

- RVC with F0 (CREPE pitch)
  - Purpose: Improve pitch tracking and allow pitch shifting before VC.
  - Sources: `./AiliaRvc.cs`, `./AiliaRvcCrepe.cs`
  - Script behavior: Adds a CREPE‑based F0 estimator (`AiliaRvcCrepe`) opened with `OpenFileF0(...)` using `crepe_tiny.onnx` or `crepe.onnx`. The class computes per‑frame F0, applies optional pitch shift (`SetF0UpKeys(int semitones)`), quantizes to coarse bins, and feeds pitch features into the VC model. CREPE supports GPU and processes audio in 10 ms hops. For this mode the VC model with F0 input (`rvc_f0.onnx`) is required and is loaded from `StreamingAssets`.

## Helper/Utility Scripts

- `./AiliaMicrophone.cs`: Manages microphone or audio clip input. For mic mode, gathers incremental PCM from `Microphone.*` and returns the latest chunk per frame via `GetPcm(ref channels, ref frequency)`.
- `./AiliaSplitAudio.cs`: Segments the incoming stream into utterances using VAD confidence. It tracks active/silent durations and emits `AudioClip`s when a speech segment followed by enough silence is detected.
- `./AiliaDisplayAudio.cs`: Renders a preview texture showing recent PCM and VAD confidence, and waveforms of segmented clips; useful for debugging.
- `./AiliaAudioProcessingSample.cs`: Orchestrates the pipeline (download/open models, mic/file input, VAD → split → optional RVC, async playback, on‑screen preview).

## Model Files and Paths

- Silero VAD
  - v4: `silero_vad.onnx` (+ `.onnx.prototxt`)
  - v6.2: `silero_vad_v6_2.onnx` (+ `.onnx.prototxt`)
- RVC
  - Features: `hubert_base.onnx` (+ `.onnx.prototxt`)
  - Generator (example): `AISO-HOWATTO.onnx` (+ `.onnx.prototxt`)
- RVC with F0
  - Pitch: `crepe_tiny.onnx` or `crepe.onnx` (+ `.onnx.prototxt`)
  - VC with F0 inputs: `rvc_f0.onnx` (+ `.onnx.prototxt`) — place in `Application.streamingAssetsPath` as referenced in `AiliaAudioProcessingSample.cs`.

The sample uses `AiliaDownload` to fetch model assets into `Application.temporaryCachePath`. See `CreateAiliaNetwork(...)` in `AiliaAudioProcessingSample.cs` for exact file names and locations.

## Reusing in Your Own Code

- Voice Activity Detection (Silero VAD)
  - Create and open:
    - `var vad = new AiliaSileroVad();`
    - `vad.OpenFile(pathToPrototxt, pathToOnnx, gpuMode);`
  - Infer on mono PCM:
    - `var res = vad.VAD(pcm, channels: 1, sampleRate: 48000);`
    - Use `res.conf` (0..1) per input sample to detect speech; `res.pcm` is the portion already consumed; keep feeding subsequent chunks for streaming.

- Splitting by Utterance
  - `var splitter = new AiliaSplitAudio();`
  - Each frame: `splitter.Split(res);`
  - When `splitter.GetAudioClipCount() > 0`, retrieve with `splitter.PopAudioClip()`.

- Voice Conversion (RVC)
  - Create and open models:
    - `var rvc = new AiliaRvc();`
    - `rvc.OpenFile(hubertProto, hubertOnnx, vcProto, vcOnnx, rvcVersion, gpuMode);`
    - Optional F0: `rvc.OpenFileF0(crepeProto, crepeOnnx, f0GpuMode);`
    - Optional: `rvc.SetF0UpKeys(12); // +12 semitones`
    - Optional: `rvc.SetTargetSmaplingRate(48000);`
  - Convert a segmented clip:
    - Synchronous: `var outClip = rvc.Process(inClip);`
    - Asynchronous: `rvc.AsyncProcess(inClip);` then poll `rvc.AsyncResultExist()` and call `rvc.AsyncGetResult()`.

- End‑to‑end (like the sample)
  - Acquire PCM from `AiliaMicrophone` (mic) or an `AudioClip` (file).
  - Run VAD each frame, pass result to `AiliaSplitAudio`.
  - For each emitted clip, optionally run RVC (sync or async) and enqueue for playback.

## Practical Notes

- Input must be mono (1 channel). `AiliaSileroVad` and `AiliaRvc` expect mono. Down‑mix before feeding if needed.
- Sample rate can be native (e.g., 48 kHz). Silero VAD and RVC internally resample to 16 kHz where required.
- For GPU, the scripts call `AiliaModel.Environment(AILIA_ENVIRONMENT_TYPE_GPU)` when the `gpu_mode`/`rvc_f0_gpu_mode` flags are true.
- The sample calls `AiliaLicense.CheckAndDownloadLicense()` on start; include similar logic in your app if not already present.
- Third‑party model licenses are included under `AudioProcessing/LICENSE`.

## Source Links

- Sample controller: `./AiliaAudioProcessingSample.cs`
- VAD model: `./AiliaSileroVad.cs`
- RVC model: `./AiliaRvc.cs`
- F0 (CREPE): `./AiliaRvcCrepe.cs`
- Microphone utility: `./AiliaMicrophone.cs`
- Splitter: `./AiliaSplitAudio.cs`
- Display: `./AiliaDisplayAudio.cs`

