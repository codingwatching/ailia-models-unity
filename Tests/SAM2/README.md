# SAM2 Mask Tests

SAM2 (Segment Anything 2) の C# 推論ロジックをテストするプロジェクトです。
Unity と同一の共有コード (`Sam2InferenceEngine.cs`) をテストします。
`AiliaSam2Backend` は `Sam2InferenceEngine.cs` に統合されています。

## アーキテクチャ

```
Assets/AXIP/AILIA-MODELS/ImageSegmentation/
  ├── Sam2InferenceEngine.cs   ← 共有ロジック + AiliaSam2Backend (Unity / テスト両方で使用)
  ├── Sam2Processor.cs         ← 推論パイプライン
  └── SegmentAnything2Model.cs ← Unity ラッパー (テストでは不使用)

Tests/SAM2/
  ├── Sam2MaskTests.csproj     ← テストプロジェクト (.NET 8.0 / NUnit)
  ├── UnitTest1.cs             ← ユニットテスト (前処理、リサイズ、座標変換等)
  ├── PythonComparisonTest.cs  ← Python リファレンス値との比較テスト
  ├── Sam2InferenceTest.cs     ← E2E 推論テスト (ailia)
  ├── SegmentAnything2ModelTest.cs ← Sam2Processor テスト
  └── UnityInputSimulationTest.cs  ← Unity 入力シミュレーションテスト
```

テストプロジェクトは `<Compile Include="..." Link="..." />` で共有ファイルをリンクしており、
Unity で動作するコードと完全に同一のロジックをテストします。

## セットアップ

### Step 1: .NET 8.0 SDK のインストール

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y dotnet-sdk-8.0

# macOS (Homebrew)
brew install dotnet@8

# バージョン確認
dotnet --version
```

### Step 2: ailia-csharp のクローン

テストプロジェクトは ailia SDK の C# ラッパーファイルを `/tmp/ailia-csharp` から参照します。
サブモジュール (`ailia-sdk-unity`) を含めてクローンしてください。

```bash
git clone --recursive https://github.com/ailia-ai/ailia-csharp.git /tmp/ailia-csharp
```

> **既にクローン済みでサブモジュールが未取得の場合:**
>
> ```bash
> cd /tmp/ailia-csharp
> git submodule update --init --recursive
> ```

クローン後、以下のファイルが存在することを確認してください:

```
/tmp/ailia-csharp/ailia-csharp/ailia-csharp/ailia/ailia-sdk-unity/Runtime/
  ├── Api/Ailia.cs                    ← ailia C API ラッパー
  ├── Models/AiliaModel.cs            ← モデル管理クラス
  ├── Models/AiliaLicense.cs          ← ライセンス管理
  └── Plugins/linux/libailia.so       ← ネイティブライブラリ (Linux)
```

### Step 3: ビルド確認

```bash
cd Tests/SAM2
dotnet build
```

これでユニットテストが実行可能になります。E2E テストにはさらにモデルとテスト画像が必要です (Step 4 以降)。

### Step 4: SAM2 ONNX モデルのダウンロード

E2E 推論テストにはモデルファイルが必要です。

```bash
mkdir -p /tmp/sam2_models
cd /tmp/sam2_models

# ONNX モデル
curl -O https://storage.googleapis.com/ailia-models/segment-anything-2/image_encoder_hiera_l.onnx
curl -O https://storage.googleapis.com/ailia-models/segment-anything-2/mask_decoder_hiera_l.onnx
curl -O https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_hiera_l.onnx

# ailia 用 prototxt
curl -O https://storage.googleapis.com/ailia-models/segment-anything-2/image_encoder_hiera_l.onnx.prototxt
curl -O https://storage.googleapis.com/ailia-models/segment-anything-2/mask_decoder_hiera_l.onnx.prototxt
curl -O https://storage.googleapis.com/ailia-models/segment-anything-2/prompt_encoder_hiera_l.onnx.prototxt
```

ダウンロード後のディレクトリ構成:

```
/tmp/sam2_models/
  ├── image_encoder_hiera_l.onnx            (約 812 MB)
  ├── image_encoder_hiera_l.onnx.prototxt   (ailia 用)
  ├── mask_decoder_hiera_l.onnx             (約 16 MB)
  ├── mask_decoder_hiera_l.onnx.prototxt    (ailia 用)
  ├── prompt_encoder_hiera_l.onnx           (約 89 KB)
  └── prompt_encoder_hiera_l.onnx.prototxt  (ailia 用)
```

### Step 5: テスト画像の準備

```bash
mkdir -p /tmp/sam2_test_output

# SAM2 公式リポジトリからテスト画像をダウンロード (JPG)
curl -o /tmp/sam2_test_output/truck.jpg \
  https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg

# ロスレス PNG に変換 (Python の PIL または ImageMagick を使用)
# ImageMagick の場合:
convert /tmp/sam2_test_output/truck.jpg /tmp/sam2_test_output/truck.png

# または Python の場合:
# python3 -c "from PIL import Image; Image.open('/tmp/sam2_test_output/truck.jpg').save('/tmp/sam2_test_output/truck.png')"
```

> **注意**: テストは PNG (ロスレス) を使用します。JPG はデコーダ実装の差異により Python との比較に
> 誤差が生じるため、必ず PNG に変換してください。

### Step 6 (任意): Python リファレンスマスクの生成

Python との精度比較テストを実行するには、リファレンスマスクが必要です。
以下の Python スクリプトで生成できます。

```bash
pip install torch torchvision sam2 numpy pillow

python3 -c "
import numpy as np
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-large')
image = Image.open('/tmp/sam2_test_output/truck.png')
predictor.set_image(np.array(image))
masks, scores, _ = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    multimask_output=True,
)
best_idx = scores.argmax()
np.save('/tmp/sam2_test_output/best_mask_bool_png.npy', masks[best_idx])
print(f'Saved best mask (index={best_idx}, score={scores[best_idx]:.4f})')
"
```

> このステップをスキップした場合、E2E テストは推論自体は実行しますが Python との比較はスキップされます。

### Step 7 (ailia テスト用): ネイティブライブラリの配置

ailia バックエンドのテストには、ネイティブライブラリをビルド出力にコピーする必要があります。

```bash
cd Tests/SAM2

# ビルド
dotnet build

# ネイティブライブラリをビルド出力にコピー
cp /tmp/ailia-csharp/ailia-csharp/ailia-csharp/ailia/ailia-sdk-unity/Runtime/Plugins/linux/libailia.so \
   bin/Debug/net8.0/libailia.so
```

## テストの実行

### ユニットテストのみ (モデル不要、Step 3 まででOK)

```bash
cd Tests/SAM2
dotnet test --filter "UnitTest1|PythonComparisonTest"
```

モデルや画像がなくても実行できるテスト (60テスト中 57テスト) です。
前処理の正規化、bilinear リサイズ、座標変換、テンソル操作などを検証します。

### ailia SDK E2E テスト (Step 7 まで必要)

```bash
cd Tests/SAM2
LD_LIBRARY_PATH=bin/Debug/net8.0:$LD_LIBRARY_PATH \
  dotnet test --verbosity normal --filter "FullInference_Ailia"
```

### 全テスト実行

```bash
cd Tests/SAM2
LD_LIBRARY_PATH=bin/Debug/net8.0:$LD_LIBRARY_PATH \
  dotnet test --verbosity normal
```

## テスト内容

| ファイル | テスト数 | 内容 |
|---|---|---|
| `UnitTest1.cs` | 35 | 前処理、リサイズ、座標変換、テンソル操作のユニットテスト |
| `PythonComparisonTest.cs` | 13 | Python SAM2 リファレンス実装との数値比較 |
| `Sam2InferenceTest.cs` | 1 | E2E 推論 (ailia) |
| `SegmentAnything2ModelTest.cs` | 14 | Sam2Processor テスト |
| `UnityInputSimulationTest.cs` | 17 | Unity 入力シミュレーションテスト |

## Python との精度

| バックエンド | 一致率 | 差異ピクセル数 | 備考 |
|---|---|---|---|
| ailia SDK | 99.9998% | 5 | 2,160,000 ピクセル中 |

差異はマスク境界 (値 ≈ 0.0) での float32 丸め誤差によるもので、実用上の影響はありません。

## トラブルシューティング

### `ailia-csharp` が見つからない

```
error CS9010: Primary constructor body is not allowed
```

`/tmp/ailia-csharp` にクローンされているか、サブモジュールが取得済みか確認してください。

```bash
ls /tmp/ailia-csharp/ailia-csharp/ailia-csharp/ailia/ailia-sdk-unity/Runtime/Api/Ailia.cs
```

### ailia ネイティブライブラリが見つからない

```
DllNotFoundException: Unable to load shared library 'ailia'
```

`libailia.so` がビルド出力にコピーされているか確認してください。

```bash
ls Tests/SAM2/bin/Debug/net8.0/libailia.so
```

### ailia ライセンスエラー

```
License file not found
ailiaCreate failed -20
```

ailia SDK (time_license 版) はライセンスファイルが必要です。
初回実行時に自動ダウンロードされますが、テスト環境では手動で取得が必要な場合があります。

```bash
mkdir -p ~/.shalo
curl -o ~/.shalo/AILIA.lic \
  "https://axip-console.appspot.com/license/download/product/AILIA"
```

### モデルが見つからずテストがスキップされる

E2E テストはモデルが見つからない場合 `Assert.Ignore` でスキップされます。
Step 4 のモデルダウンロードを完了してください。
