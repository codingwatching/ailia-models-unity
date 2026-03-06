# Natural Language Processing (ailia SDK for Unity)

This folder provides embeddings and machine translation examples. The scene is `NaturalLanguageProcessing/AiliaNaturalLanguageProcessingSample.unity`, controller `NaturalLanguageProcessing/AiliaNaturalLanguageProcessingSample.cs`.

## Supported Models

- Sentence Transformer (Japanese, paraphrase‑multilingual‑mpnet‑base‑v2)
  - Purpose: Universal sentence embeddings.
  - Tokens: XLM‑RoBERTa (sentencepiece.bpe.model).

- Multilingual‑E5 Base
  - Purpose: Text embeddings for retrieval (“passage:” prefix used).
  - Tokens: XLM‑RoBERTa (sentencepiece.bpe.model).

- EmbeddingGemma 300M
  - Purpose: Lightweight high‑quality embeddings.
  - Tokens: Gemma tokenizer (tokenizer.model).

- FuguMT (en→ja, ja→en)
  - Purpose: Machine translation; en‑ja single ONNX with past KV; ja‑en encoder/decoder pair.
  - Tokens: Source/target SPM models.

## Script Behavior

- Downloads required ONNX and tokenizer files to `Application.temporaryCachePath`.
- Embedding flow: tokenizes, runs `Predict`, collects embedding vectors, and shows timing. Optionally chunks a text database and embeds sequentially.
- Translation flow: uses `AiliaSpeechTranslateModel.Open(...)` for FuguMT, then `Transcribe`‑like translate calls with post‑processing types per direction.

## Reusing in Your Own Code

- Embeddings:
  - Open `AiliaModel` and `AiliaTokenizerModel` with matching tokenizer type; call into `AiliaNaturalLanguageProcessingTextEmbedding.Embedding(...)`.
- Translation:
  - Use `AiliaSpeechTranslateModel.Open(encoder, decoderOrNull, srcSpm, tgtSpm, postType, envId, memoryMode)`; feed text per the API to get translated output.

## Source Links

- Controller: `./AiliaNaturalLanguageProcessingSample.cs`
- Embedding helper: `./AiliaNaturalLanguageProcessingTextEmbedding.cs`

