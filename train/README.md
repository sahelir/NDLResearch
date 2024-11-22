# 追加学習及びモデル変換手順
次の環境で動作確認を実施しました。
```
CPU: Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
RAM: 256GB
GPU: NVIDIA A100 x 1
OS: Ubuntu 22.04.4 LTS
NVIDIA Driver Version: 535.113.01
CUDA: 11.8
```

## レイアウト認識(RTMDet)

Chengqi Lyu, Wenwei Zhang, Haian Huang, Yue Zhou, Yudong Wang, Yanyi Liu, Shilong Zhang, Kai Chen. Rtmdet: An empirical study of designing real-time object detectors. arXiv preprint arXiv:2212.07784, 2022.(https://arxiv.org/abs/2212.07784)

を利用してレイアウト認識モデルを作成します。

ここではmmdetv3-rtmdet_s_8xb32-300e_cocoのみを対象としてカスタマイズを行います。

他のサイズのモデル等については次のURLを参照してください。
https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet

この項で紹介する当館が作成したサンプルコードはrtmcodeディレクトリ以下にあります。

### 環境構築
```
cp rtmcode/* .
python3 -m venv rtmdetenv
source ./rtmdetenv/bin/activate
python3 -m pip install --upgrade pip
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmdet==3.0.0
```

### 学習データの変換
次の手順は国立国会図書館が公開している[NDL-DocLデータセット](https://github.com/ndl-lab/layout-dataset)を学習用データセットのアノテーション情報に変換する手順を示しています。データを追加する場合にはrtmcococonverter.pyを適宜編集して利用してください。

```
pip install pandas tqdm
wget https://lab.ndl.go.jp/dataset/dataset_kotenseki.zip
unzip dataset_kotenseki.zip
python3 rtmcococonverter.py
```

### 学習
事前にrtmdettrain.pyを読んで、データセットのパス等を修正してください。
```
python3 rtmdettrain.py
```

### 学習済モデルONNXへの変換

```
pip install numpy==1.21.6 onnx==1.16.2 onnxruntime-gpu==1.18.1  mmdeploy==1.3.1
git clone https://github.com/open-mmlab/mmdeploy -b v1.3.1
python3 mmdeploy/tools/deploy.py \./rtmonnx_config.py \
    ./mmdetv3-rtmdet_s_8xb32-300e_coco_sample.py \
    ./work_dir_mmdetv3_rtmdet_s/epoch_300.pth \
    ./dataset_kotenseki/10301071/10301071_0006.jpg \
    --work-dir mmdeploy_model/rtmdet_s
```
mmdeploy_model/rtmdet_sディレクトリ以下にrtmdet-s-1280x1280.onnxが生成されます。

NDL古典籍OCR-Liteで利用する場合は、--det-weightsオプションでonnxファイルのパスを指定してください。


## 文字列認識(PARSeq)
Darwin Bautista, Rowel Atienza. Scene text recognition with permuted autoregressive sequence models. arXiv:2212.06966, 2022. (https://arxiv.org/abs/2207.06966)

を利用して文字列認識モデルを作成します。

ここではparseq-tinyを利用してモデルを作成します。

この項で紹介する当館が作成したサンプルコードはparseqcodeディレクトリ以下にあります。

### 環境構築

