# EcoSort Mini - PAI最終課題 / Sabercon

RGB-D画像で色分け回収物を認識し、7軸のKUKA iiwaアームで3種類の回収箱と待確認箱へ搬送する、オリジナルのPyBullet課題作品です。実機を用いない物理シミュレーションです。色は回収先の記号であり、材質・廃棄物の種類を認識しているわけではありません。

## 実行方法

確認環境：Windows、Python 3.12.14、PyBullet 3.2.7。依存関係は `requirements.txt` に固定しています。PyBulletのWindowsインストールではC++ビルド環境が必要になる場合があります。

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py train
python run.py check --seed 7
python run.py check --seed 29
python run.py preview
python run.py video --output demo.mp4
```

`check` はGUI・FFmpeg・フォント・APIキーなしで実行できます。`video` にはPATH上のFFmpegと日本語フォントが必要です。WindowsではYu Gothicを使用します。他環境では環境変数 `ECOSORT_FONT` に日本語TTF/TTCフォントを指定してください。MP4は1920×1080・30fps、シミュレーションの固有描画は15fpsです。同じ画像を2フレーム出力し、240Hzの物理時間に対して等速です。音声はなく、日本語説明字幕を組み込んでいます。

## AIの実装

1. PyBulletで赤・青・黄の立方体を、色・照明・角度・位置を変えて生成。
2. RGBの中央値を色度に正規化し、softmax回帰を勾配降下法で学習。
3. 初回のRGB-D画像から、彩度に基づく物体候補と連結成分を検出。
4. 深度とカメラ行列から3次元位置を復元し、学習モデルで回収先を推定。
5. 学習クラス中心からの距離が学習データ由来の閾値を超える、またはsoftmax最大値が0.8未満なら `CHECK` に保留。

学習画像から特徴を得る際のみ、教師データ作成用の物体マスクを使います。実行時の画像処理はRGB・深度・カメラ行列だけを入力とし、物体IDや正解クラスを入力しません。物体IDとシミュレータ座標は、吸着の物理接続先を距離条件で選ぶ処理、および結果の自動採点に使用します。教師ラベルは実行中の分類先決定には使いません。

## ロボット制御

PyBullet同梱KUKA iiwaモデルを使用。逆運動学で姿勢を求め、位置モーターで動作させます。五次補間で移動し、接近、下降、吸着、上昇、搬送、解除、検証の状態機械を独自実装しました。吸着は先端と物体上面の距離が27mm未満の場合に固定拘束を生成する近似です。物体の重力・衝突・着地は物理計算を行います。カップは可視化用で、圧力・密閉・変形は計算しません。

## 実測結果

| 項目 | 結果 | 条件 |
|---|---:|---|
| 教師あり学習 | 360例 | 3クラス×120、seed 42 |
| 既知色テスト | 120/120正解 | 各クラス40、別seed 2026 |
| 未学習の紫色テスト | 30/30を保留 | 別seed 773、紫色だけのテスト |
| 仕分け | 6/6成功 | シーンseed 7 |
| 追加シーン | 6/6成功 | シーンseed 29、位置・色・角度に小さな揺らぎ |

成功は「持ち上がった」「指定箱の内側へ着地した」「正解の回収先と一致した」の3条件です。これらは同じ仮想環境内の限定的な結果で、一般的な未知物体や実機の成功率ではありません。softmax値も現実での正解確率として校正していません。

## 構成

- `perception.py`: softmax学習、距離による保留、RGB-D物体検出
- `simulation.py`: シーン生成、学習画像生成、状態機械、逆運動学・吸着近似
- `video.py`: 日本語字幕・AIスコア・カメラ画像・回収数を合成する動画出力
- `run.py`: 再現実行用CLI
- `models/color_ai.npz`: 学習済み係数・クラス中心・判定半径
- `evidence/`: 学習特徴、混同行列、各seedの全試行・時系列イベント・先端軌跡、代表画像

動画も同じ `Cell.run()` を使用し、可視化のために物体位置をアニメーションで書き換えることはしていません。物理時間・関節制御・拘束を実際に進めて録画しています。初期化時の姿勢設定と、可視化用カップの追従は別処理です。

## 限界と次の改善

- 初回の静的画像を一括認識する方式。途中で外乱がある場面や物体の重なりには未対応。
- 色・既知の作業領域・固定カメラに依存。黒・低彩度の物体を検出できない場合があります。
- 未知色評価は紫30例のみ。任意の未知物体を保留できる保証はありません。
- 黄色はアームの一部で隠れ、位置推定にずれが生じます。今回の把持許容範囲内で処理しました。
- 真空圧、力覚、密閉、実機安全性、セグメンテーション学習、衝突を考慮した経路探索は未実装。
- 次は把持前の再撮像、学習条件の拡張、実測カメラの校正、接触モデルの検証に取り組みます。

## 出典・制作補助

課題用ロジック、シーン、色認識学習、保留判定、字幕・評価は本作品用に作成しました。課題サンプルの丸写しは行っていません。生成AIを設計・コード・説明文作成の補助に使用し、上記の実行とログで確認しました。課題内で動くAIは外部LLMではなく、ローカルで学習・推論する色分類器です。

- [PyBullet公式Quickstart Guide](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)
- [Bullet Physics / PyBullet ソースとライセンス](https://github.com/bulletphysics/bullet3)
- [PyBullet同梱KUKAモデル](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/kuka_iiwa)

第三者アセットを再配布せず、PyBulletの同梱データを参照します。本リポジトリの独自コードはMIT Licenseです。
