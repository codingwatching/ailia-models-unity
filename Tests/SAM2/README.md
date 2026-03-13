# SAM2 Mask Tests

SAM2 (Segment Anything 2) の C# 推論ロジックをテストするプロジェクトです。
Unity と同一の共有コード (`Sam2InferenceEngine.cs`, `AiliaSam2Backend.cs`) をテストします。

## アーキテクチャ

```
Assets/AXIP/AILIA-MODELS/ImageSegmentation/
  ├── Sam2InferenceEngine.cs   ← 共有ロジック (Unity / テスト両方で使用)
  ├── AiliaSam2Backend.cs      ← 共有 ailia SDK バックエンド
  └── SegmentAnything2Model.cs ← Unity ラッパー (テストでは不使用)

Tests/SAM2/
  ├── Sam2MaskTests.csproj     ← テストプロジェクト (.NET 8.0 / NUnit)
  ├── OrtSam2Backend.cs        ← ONNX Runtime バックエンド (テスト専用)
  ├── UnitTest1.cs             ← ユニットテスト (前処理、リサイズ、座標変換等)
  ├── PythonComparisonTest.cs  ← Python リファレンス値との比較テスト
  ├── Sam2InferenceTest.cs     ← E2E 推論テスト (ORT / ailia)
  └── Sam2OnnxInferenceTest.cs ← ONNX 推論の詳細比較テスト
```

テストプロジェクトは `<Compile Include="..." Link="..." />` で共有ファイルをリンクしており、
Unity で動作するコードと完全に同一のロジックをテストします。

## 前提条件

- .NET 8.0 SDK
- SAM2 ONNX モデル (Hiera-L)
- テスト画像 (`truck.png`)
- (ailia テスト用) ailia SDK ネイティブライブラリ (`libailia.so`) と ailia-csharp ラッパー

## モデルの配置

ONNX モデルを `/tmp/sam2_models/` に配置してください:

```
/tmp/sam2_models/
  ├── image_encoder_hiera_l.onnx
  ├── image_encoder_hiera_l.onnx.prototxt   ← ailia 用
  ├── mask_decoder_hiera_l.onnx
  ├── mask_decoder_hiera_l.onnx.prototxt    ← ailia 用
  ├── prompt_encoder_hiera_l.onnx
  └── prompt_encoder_hiera_l.onnx.prototxt  ← ailia 用
```

テスト画像と Python リファレンス出力を `/tmp/sam2_test_output/` に配置してください:

```
/tmp/sam2_test_output/
  ├── truck.png                ← テスト画像 (1800x1200)
  └── best_mask_bool_png.npy   ← Python リファレンスマスク
```

## テストの実行

### ユニットテストのみ (モデル不要)

```bash
cd Tests/SAM2
dotnet test --filter "UnitTest1|PythonComparisonTest"
```

モデルや画像がなくても実行できるテスト (60テスト中 57テスト) です。
前処理の正規化、bilinear リサイズ、座標変換、テンソル操作などを検証します。

### ORT (ONNX Runtime) E2E テスト

```bash
cd Tests/SAM2
dotnet test --filter "FullInference_ORT|FullInference_PngEndToEnd|Preprocess_PngZeroError"
```

ONNX モデルとテスト画像が必要です。モデルが見つからない場合は自動的にスキップされます。

### ailia SDK E2E テスト

ailia バックエンドのテストには追加の準備が必要です。

1. **ailia-csharp のクローン:**

```bash
git clone https://github.com/ailia-ai/ailia-csharp.git /tmp/ailia-csharp
```

2. **ネイティブライブラリの配置:**

`libailia.so` をビルド出力ディレクトリにコピーします:

```bash
dotnet build -p:DefineConstants=AILIA_SDK
cp /tmp/ailia-csharp/ailia-csharp/ailia-csharp/ailia/ailia-sdk-unity/Runtime/Plugins/linux/libailia.so \
   bin/Debug/net8.0/libailia.so
```

3. **テスト実行:**

```bash
cd Tests/SAM2
LD_LIBRARY_PATH=bin/Debug/net8.0:$LD_LIBRARY_PATH \
  dotnet test --verbosity normal -p:DefineConstants=AILIA_SDK --filter "FullInference_Ailia"
```

### ORT vs ailia バックエンド比較テスト

```bash
cd Tests/SAM2
LD_LIBRARY_PATH=bin/Debug/net8.0:$LD_LIBRARY_PATH \
  dotnet test --verbosity normal -p:DefineConstants=AILIA_SDK --filter "CompareBackends_ORT_vs_Ailia"
```

### 全テスト実行

```bash
cd Tests/SAM2
LD_LIBRARY_PATH=bin/Debug/net8.0:$LD_LIBRARY_PATH \
  dotnet test --verbosity normal -p:DefineConstants=AILIA_SDK
```

## テスト内容

| ファイル | テスト数 | 内容 |
|---|---|---|
| `UnitTest1.cs` | 35 | 前処理、リサイズ、座標変換、テンソル操作のユニットテスト |
| `PythonComparisonTest.cs` | 13 | Python SAM2 リファレンス実装との数値比較 |
| `Sam2InferenceTest.cs` | 3 | E2E 推論 (ORT / ailia / バックエンド比較) |
| `Sam2OnnxInferenceTest.cs` | 2 | ONNX 推論の前処理精度・マスク比較 |

## Python との精度

| バックエンド | 一致率 | 差異ピクセル数 | 備考 |
|---|---|---|---|
| ORT (ONNX Runtime) | 99.9999% | 1 | 2,160,000 ピクセル中 |
| ailia SDK | 99.9998% | 5 | 2,160,000 ピクセル中 |

差異はマスク境界 (値 ≈ 0.0) での float32 丸め誤差によるもので、実用上の影響はありません。
