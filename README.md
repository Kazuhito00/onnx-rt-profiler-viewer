# onnx-rt-profiler-viewer
ONNX モデルの入力を自動推定し推論後、プロファイル結果を HTML レポートとして出力するツール。<BR>既存のプロファイル JSON から HTML レポートのみを生成することも可能。<BR>
表示イメージは以下URLから確認できます。
* https://kazuhito00.github.io/onnx-rt-profiler-viewer/assets/ort_profile_sample.html

<img width="85%" alt="image" src="https://github.com/user-attachments/assets/2bcb375d-801d-41af-b90b-489069c3c4d0" /><br><img width="85%" alt="image" src="https://github.com/user-attachments/assets/b0da4f38-e7c3-4ae2-958a-aee957606a54" />

## Requirements

- `onnx` — モデル読み込み・入力推定
- `onnxruntime` または `onnxruntime-gpu` — 推論実行・プロファイリング
- `numpy` — データ生成・集計

HTML レポートは以下の CDN ライブラリを使用（ネット接続が必要）:
- [Chart.js](https://www.chartjs.org/) — 棒グラフ・円グラフ
- [D3.js](https://d3js.org/) — flamegraph 描画
- [d3-flame-graph](https://github.com/spiermar/d3-flame-graph) — flamegraph ライブラリ

## Installation
```bash
# CPU のみ
pip install onnx onnxruntime numpy

# GPU (CUDA) を使用する場合
pip install onnx onnxruntime-gpu numpy
```

## Usage

2つのサブコマンドがあります。
### `run` — モデル実行 + プロファイル + HTML 生成

```bash
python main.py run <model.onnx> [オプション]
```

```bash
# 基本
python main.py run model.onnx -o output/

# シェイプ指定（動的次元を解決）
python main.py run model.onnx -s "images:1,3,480,640" -s "orig_target_sizes:1,2"

# 固定値指定（条件分岐に影響する入力等）
python main.py run model.onnx -s "input:1,512" -v "sr=16000"

# 実行回数を変更
python main.py run model.onnx -n 5 -w 2

# EP を明示指定（CUDA + CPU）
python main.py run model.onnx -e CUDA -e CPU

# CPU のみを強制
python main.py run model.onnx -e CPU
```

| オプション | 説明 | デフォルト |
|---|---|---|
| `-d`, `--dynamic-size` | 動的次元に使用するサイズ | `1` |
| `-s`, `--input-shape` | 入力シェイプを明示指定（複数指定可） | 自動推定 |
| `-v`, `--input-value` | 入力の固定値を指定（複数指定可） | ランダム生成 |
| `-o`, `--output-dir` | 出力ディレクトリ | `.` |
| `-n`, `--num-runs` | 計測実行回数 | `3` |
| `-w`, `--num-warmup` | ウォームアップ回数 | `1` |
| `-e`, `--ep` | 使用する Execution Provider（複数指定可） | 自動検出 |

### `report` — 既存 JSON から HTML レポート生成

ORT が出力したプロファイル JSON を指定して、HTML レポートのみ生成します。
ウォームアップ/計測回数の区別がないため、JSON 内の全 `model_run` を計測扱いとして集計します。

```bash
# JSON と同じディレクトリに同名の .html を生成
python main.py report ort_profile_2026-03-26_18-28-07.json

# 出力パスを指定
python main.py report ort_profile.json -o report.html
```

| オプション | 説明 | デフォルト |
|---|---|---|
| `-o`, `--output` | 出力 HTML ファイルパス | JSON と同名 `.html` |

#### 自分のコードでプロファイル JSON を出力して使う例

自分の推論コード内で ORT のプロファイル機能を有効にし、出力された JSON を `report` で可視化できます。

```python
import onnxruntime

options = onnxruntime.SessionOptions()
options.enable_profiling = True          # プロファイル機能有効化
session = onnxruntime.InferenceSession("model.onnx", options)

# --- プロファイル対象の推論 ---
output = session.run(None, {"input": input_data})
# 複数回実行してもOK（全実行が記録される）
output = session.run(None, {"input": input_data})

prof_file = session.end_profiling()  # JSON ファイルがカレントディレクトリに保存され、パスが返る
print(prof_file)  # => ort_profile_2026-03-26_18-28-07.json
```

> `end_profiling()` を呼んだ時点で JSON ファイルが自動保存されます。保存先を変えたい場合は `options.profile_file_prefix` でディレクトリ付きのプレフィックスを指定してください。

```bash
# 出力された JSON から HTML レポートを生成
python main.py report ort_profile_2026-03-26_18-28-07.json
```

## Outputs

### プロファイル JSON
ORT のネイティブプロファイル出力。

### HTML レポート
プロファイル JSON と同名の `.html` ファイルが生成されます。ブラウザで開くとインタラクティブなレポートが表示されます。

レポートの内容:

- **サマリーカード** — 平均推論時間、実行回数、ノード数、Op Type 数
- **EP 別サマリー** — 各 Execution Provider のノード数・時間・割合（複数 EP 使用時のみ）
- **Input Tensors** — モデル入力の名前、シェイプ、データ型（`run` 時のみ）
- **Execution Timeline** — chrome://tracing 風のタイムライン表示。ミニマップ、ズーム/パン、ツールチップ対応。複数 EP 使用時は EP ごとにレーンを分離
- **Flamegraph** — ノード名の階層パスに基づくフレームグラフ。クリックでズーム可能
- **Per-Run Op Type Breakdown** — Run 選択ドロップダウンで各実行の Op Type 別時間内訳を円グラフ表示
- **Top 20 Nodes** — 平均実行時間が長い上位 20 ノードの棒グラフ
- **All Nodes** — 全ノード一覧テーブル。フィルタ検索、Op Type / EP フィルタ、累積パーセンテージ表示

## Execution Provider (EP)

<img width="85%" alt="image" src="https://github.com/user-attachments/assets/19d8e734-0b0f-4cc9-801d-d1b3ebe6b7d2" />

`onnxruntime-gpu` がインストールされている場合、CUDA EP を自動検出して使用します（CPU EP をフォールバックとして併用）。<BR>
`-e` オプションで明示指定も可能です。`ExecutionProvider` サフィックスは省略できます。

| 環境 | 自動選択される EP |
|---|---|
| `onnxruntime` (CPU版) | CPU |
| `onnxruntime-gpu` (CUDA環境) | CUDA + CPU |
| `onnxruntime-gpu` (TensorRT利用可) | TensorRT + CUDA + CPU |

CUDA EP + CPU EP を併用した場合、CUDA で実行できないノードは CPU にフォールバックされます。<BR>
HTML レポートでは EP 別のノード数・時間サマリーが表示され、タイムラインでは EP ごとにレーンが分かれます。

## Automatic Input Inference

`run` コマンドでは ONNX モデルの入力定義から以下を自動推定します:

- **シェイプ**: 固定次元はそのまま使用。動的次元は `-d` の値（デフォルト: 1）で埋める。全次元が動的な 2D テンソル（音声入力等）は 512 をデフォルトにする
- **データ型**: ONNX の型定義に従い対応する NumPy 型を使用
- **値**: 整数型は 0-255 のランダム、浮動小数点型は 0-1 の一様分布（スカラー/1D は 256-1024 の正の値）、bool はランダム

## Project Structure

```
main.py                      # CLI エントリポイント (run / report サブコマンド)
onnx_profiler/
  __init__.py
  constants.py               # 定数（dtype マップ、色パレット等）
  inputs.py                  # 入力推定・ランダム生成
  profiler.py                # プロファイル集計、flamegraph ツリー構築
  report.py                  # HTML レポート生成
```

# Author
高橋かずひと(https://x.com/KzhtTkhs)

# License
onnx-rt-profiler-viewer is under [Apache 2.0 license](LICENSE).
