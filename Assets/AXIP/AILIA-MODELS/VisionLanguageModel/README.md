# Vision Language Model (ailia SDK for Unity)

This folder demonstrates running quantized GGUF VLMs locally with image understanding. The scene is `VisionLanguageModel/AiliaVisionLanguageModelSample.unity`, controller `VisionLanguageModel/AiliaVisionLanguageModelSample.cs`.

## Supported Models

- Gemma 3 4B (IT, Q4_K_M) + mmproj-model-f16

## Script Behavior

- Downloads a `.gguf` model and a multimodal projector to `Application.temporaryCachePath` and opens them with `AiliaLLMModel`.
- Loads a sample image via `UnityWebRequest` and saves it to a temporary file for multimodal input.
- Maintains chat history (`AiliaLLMMultimodalChatMessage`), sets a system prompt, and streams tokens via `Generate()` while updating UI.
- Attaches image data to user messages using `AiliaLLMMediaData` with a `<__media__>` tag in the content.
- Handles context reset when `ContextFull()` is true.

## Reusing in Your Own Code

- Initialize:
  - `var llm = new AiliaLLMModel(); llm.Create();`
  - `llm.Open(pathToGguf, 2048);`
  - `llm.OpenMultimodalProjector(pathToMmprojGguf);`
- Chat:
  - Maintain `List<AiliaLLMMultimodalChatMessage>` for system/user/assistant turns.
  - For user messages with images, set `message.content = "query <__media__>"` and add `AiliaLLMMediaData` with `media_type = "image"` and `file_path` to the message.
  - `llm.SetMultimodalPrompt(messages);` then loop `llm.Generate(ref done)` and append `llm.GetDeltaText()`.

## Source Links

- Controller: `./AiliaVisionLanguageModelSample.cs`
