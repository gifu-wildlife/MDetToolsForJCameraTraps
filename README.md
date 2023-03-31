# MDetToolsForJCameraTraps

## What's this：このプログラムについて

The purpose of this program is to detect wildlife from camera trap footage using [MegaDetector (Beery et al. 2019)](https://github.com/microsoft/CameraTraps) and to use a different classification model (resnet50) trained with wildlife images taken in Japan to identify the detected animals to species. This document is minimally descriptive at this time and will be updated as needed.  
このプログラムは、[MegaDetector (Beery et al. 2019)](https://github.com/microsoft/CameraTraps)を利用してカメラトラップ映像から野生動物を検出し、検出された動物を日本国内で取得された野生動物画像で学習を行った別の分類モデル（resnet50）で種判別することを目的として作成されました。このドキュメントは現時点では最低限の記述しかされていないため、今後随時更新していく予定です。

A program on learning species classification models will also be available on github at a later date (under construction). Note that the species classification model at this time uses the same dataset used in [Ando et al. (2019, in Japanese)](https://doi.org/10.11238/mammalianscience.59.49), so the number of images per animal species is unbalanced.  
種判別モデル構築に関するプログラムも後日githubで公開予定です。なお、現時点における種判別モデルは、[安藤ら(2019)]( https://doi.org/10.11238/mammalianscience.59.49)で用いたものと同じデータセットを使っているため、動物種毎の画像数はアンバランスです。

This program was supported by the Environment Research and Technology Development Fund (JPMEERF20204G01,[Reports in Japansese](https://sites.google.com/view/hyogowildlife/suishin4g2001)) of the Environmental Restoration and Conservation Agency, and is published according to MIT license.  
このプログラムは環境省の環境研究総合推進費（4G-2001 イノシシの個体数密度およびCSF感染状況の簡易モニタリング手法の開発：[報告集](https://sites.google.com/view/hyogowildlife/suishin4g2001)）を受けて作成されたものであり、MITライセンスにしたがって公開されています。  

---



## Get Started：はじめに

<br />

### Prerequisites：環境整備

* OS  
    The following code was tested on Ubuntu 20.04LTS (x86-64).  
    During the test run, .mp4 was used as the video file format and .jpg as the image file format.  
    以下のコードはUbuntu 20.04LTS(x86-64)で動作確認しています。  
    動作確認時、動画ファイル形式は.mp4、静止画ファイル形式は.jpgを用いました。


*  NVIDIA Driver

    ```bash
    sudo apt install nvidia-driver-***
    ```

    Please refer to [NVIDIA Driver Version Check](https://www.nvidia.com/Download/index.aspx?lang=en-us).
    *** is a placeholder. Please enter the recommended nvidia driver version.  
    [NVIDIAドライババージョンチェック](https://www.nvidia.com/Download/index.aspx?lang=en-us)を参照し、***に推奨されるnvidiaドライババージョンを入力した上で実行してください。  


    Check installation.  
    インストール状況の確認。

    ```bash
    nvidia-smi  # NVIDIA Driver installation check
    ```

    If nvidia-smi does not work, Try Rebooting.  
    nvidia-smiコマンドが動作しない場合は再起動してみてください。

* Conda

    Download installer and run the script.  
    インストーラーをダウンロードしてスクリプトを実行します。

    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"
    bash Mambaforge-Linux-x86_64.sh
    source ~/.bashrc
    ```

    For more information, please refer to [miniforge repository](https://github.com/conda-forge/miniforge) and/or [Mamba documentation](https://mamba.readthedocs.io/en/latest/index.html).  
    詳細については[miniforge repository](https://github.com/conda-forge/miniforge) や[Mamba documentation](https://mamba.readthedocs.io/en/latest/index.html)を参照してください。  

<br />

### Instllation：インストール

1. Clone the Repository：リポジトリの複製

    Run ```git clone```,  
    ```git clone```を実行するか、

    ```bash
    git clone https://github.com/gifu-wildlife/MDetToolsForJCameraTraps.git
    ```

    or Download ZIP and Unzip in any directory of yours. The following codes are assumed that it was extracted to the user's home directory (`/home/${USER}/`).  
    もしくはZIPをダウンロードし、任意のディレクトリで解凍してください。なお、このページではユーザのホームディレクトリ（`/home/${USER}/`）に解凍した前提でスクリプトを記載しています。

    ![Screenshot from 2022-11-11 13-07-09](https://user-images.githubusercontent.com/50891743/201261079-74254fd8-ce4f-4a0f-9085-3a5209d40f7c.png)

2. Move Project Directory：プロジェクトディレクトリへ移動

    ```bash
    cd MDetToolsForJCameraTraps
    # or
    # cd MDetToolsForJCameraTraps-main
    ```

3. create conda environment：conda環境の構築

    ```bash
    mamba env create -f=environment.yml
    conda activate mdet
    ```

    #### Requirement
    必要なpythonパッケージは以下のとおり。

    * python=3.9
    * pytorch-gpu==1.10.1
    * torchvision==0.11.2
    * cudatoolkit=11.3
    * pandas
    * omegaconf
    * tqdm
    * opencv
    * tensorflow
    * humanfriendly
    * ca-certificates
    * certifi
    * openssl
    * matplotlib
    * jsonpickle  
  
<br />

4. Download MegaDetector weight file：MegaDetectorの重みファイルのダウンロードスクリプトを実行

    ```bash
    bash download_md_model.sh
    ```

5. Download Resnet50 weight and category files：Resnet50の重みファイルとカテゴリtxtのダウンロードスクリプトを実行

    ```bash
    sudo apt install curl
    bash download_resnet50_model.sh
    ```



---

## Usage：使い方

<br />

1. Download sample data：サンプルデータのダウンロード
   
    ```bash
    bash download_sample_data.sh
    ```

    Note: The sample data is extracted in the directory where `download_sample_data.sh` was executed. The following scripts are are assumed that the data is saved in `/home/${USER}/MDetToolsForJCameraTraps/`.  
    注意：サンプルデータはコードが実行されたディレクトリに保存されます。このページでは `/home/${USER}/MDetToolsForJCameraTraps/` に解凍された前提で以降のスクリプトを記載しています。

2. Run Scripts with sample data：サンプルデータに対するスクリプト実行例    
    Note:For the time being, only an example of execution using a video file (.mp4) is described.  
    注意：現時点では動画ファイル（.mp4）を用いた実行例のみ記載しています。

   
* Set environment variables for input/output directory  
  入出力ディレクトリに関する環境変数の設定

   ```bash
   video_dir=/home/${USER}/MDetToolsForJCameraTraps/sample_data/sample_session_v
   ```

    Note: The environment variable `${video_dir}` is used this sample code to keep the scripts short, but you can set any input/output directory for such as “session_root”, “output_dir”, and so on. When the output of the previous process is used in a subsequent process, the appropriate file path for the previous output must be specified. Also, the output directory must be writable.  
    注意：今回はスクリプトを短くするために環境変数`${video_dir}`を用いていますが、session_rootやoutput_dir等の入出力のディレクトリは任意に設定できます。前の処理の出力ファイルを後続の処理で使う時は適切なファイルパスを指定してください。また、出力先のディレクトリは書き込み可能である必要があります。

* Movie Clip  
  動画から静止画を抽出

    ```bash
    python exec_clip.py session_root=${video_dir} output_dir=${video_dir}-clip
    ```

* Run MegaDetector  
  MegaDetectorの実行

    ```bash
    python exec_mdet.py session_root=${video_dir}-clip mdet_config.model_path=./models/md_v4.1.0.pb

    # For MegaDetector v5.0,
    # python exec_mdet.py session_root=${video_dir}-clip mdet_config.model_path=./models/md_v5a.0.0.pt
    ```

* Bounding Box Crop  
  バウンディングボックスで動物の領域を切り出し

    ```bash
    python exec_mdetcrop.py session_root=${video_dir}-clip mdet_result_path=${video_dir}-clip/detector_output.json
    ```

* Classification  
  切り出された画像の種分類

    ```bash
    python exec_cls.py session_root=${video_dir}-clip-crop
    ```

* Summarize  
  結果の要約

    ```bash
    python exec_imgsummary.py session_root=${video_dir}-clip-crop mdet_result_path=${video_dir}-clip/detector_output.json
    ```

### Parameter：各種パラメーター（整備中）

exec_clip.py

```bash
python exec_clip.py session_root=??? output_dir=??? clip_config.start_frame=0 clip_config.end_frame=None clip_config.step=30 clip_config.ext=jpg clip_config.remove_banner=True
```

| Parameter | Status | Type | Description |
| ---- | :----: | ---- | ---- |
| session_root | required | str(path) | ~~~ |
| output_dir | required | str(path) | ~~~ |
| clip_config.start_frame | (optional) | int | ~~~ |
| clip_config.end_frame | (optional) | int | ~~~ |
| clip_config.step | (optional) | int | ~~~ |
| clip_config.ext | (optional) | str | ~~~ |
| clip_config.remove_banner | (optional) | bool | ~~~ |

exec_mdet.py

```bash
python exec_mdet.py session_root=??? mdet_config.model_path=models/md_v4.1.0.pb mdet_config.threshold=0.95 mdet_config.output_absolute_path=True mdet_config.ncores=[your cpu cores] mdet_config.verbose=False mdet_config.recursive=True
```

| Parameter | Status | Type | Description |
| ---- | :----: | ---- | ---- |
| session_root | required | str(path) | ~~~ |
| mdet_config.model_path | (optional) | str(path) | ~~~ |
| mdet_config.threshold | (optional) | float | ~~~ |
| mdet_config.output_absolute_path | (optional) | bool | ~~~ |
| mdet_config.ncores | (optional) | int | ~~~ |
| mdet_config.verbose | (optional) | bool | ~~~ |
| mdet_config.recursive | (optional) | bool | ~~~ |

exec_mdetcrop.py

```bash
python exec_mdet.py session_root=??? output_dir=${session_root}-crop mdet_result_path=${session_root}/detector_output.json mdet_crop_config.threshold=0.95 mdet_crop_config.ncores=[your cpu cores]
```

| Parameter | Status | Type | Description |
| ---- | :----: | ---- | ---- |
| session_root | required | str(path) | ~~~ |
| output_dir | required | str(path) | ~~~ |
| mdet_result_path | required | str(path) | ~~~ |
| mdet_crop_config.threshold | (optional) | float | ~~~ |
| mdet_crop_config.ncores | (optional) | int | ~~~ |

exec_cls.py

```bash
python exec_cls.py session_root=??? cls_config.model_path=models/classifire/15cat_50epoch_resnet50.pth cls_config.category_list_path=models/classifire/category.txt cls_config.result_file_name=classifire_prediction_result.csv cls_config.architecture=resnet50 cls_config.use_gpu=True cls_config.is_all_category_probs_output=False
```

| Parameter | Status | Type | Description |
| ---- | :----: | ---- | ---- |
| session_root | required | str(path) | ~~~ |
| cls_config.model_path | (optional) | str(path) | ~~~ |
| cls_config.category_list_path | (optional) | str(path) | ~~~ |
| cls_config.result_file_name | (optional) | str | ~~~ |
| cls_config.architecture | (optional) | Literal["resnet50"] | ~~~ |
| cls_config.use_gpu | (optional) | bool | ~~~ |
| cls_config.is_all_category_probs_output | (optional) | bool | ~~~ |


exec_imgsummary.py

```bash
python exec_imgsummary.py session_root=??? mdet_result_path=??? summary_config.cls_result_file_name=classifire_prediction_result.csv summary_config.category_list_path=models/classifire/category.txt summary_config.img_summary_name=img_wise_cls_summary.csv summary_config.is_video_summary=True
```

| Parameter | Status | Type | Description |
| ---- | :----: | ---- | ---- |
| session_root | required | str(path) | ~~~ |
| mdet_result_path | required | str(path) | ~~~ |
| summary_config.cls_result_file_name | (optional) | str | ~~~ |
| summary_config.category_list_path | (optional) | str(path) | ~~~ |
| summary_config.img_summary_name | (optional) | str | ~~~ |
| summary_config.is_video_summary | (optional) | bool | ~~~ |

---

## Verified GPU：各種GPUを用いた動作確認

| GPU | VRAM | Result |
| ---- | ---- | ----|
| RTX 3090 | 24GB | Success |
| RTX 3080 | 10GB | Success |
| RTX 3060 | 12GB | Success |
| RTX 3050 | 8GB | Success |
| GTX 1080 | 8GB | Success |
| RTX 3050 (Notebooks) | 4GB | Failure (Insufficient VRAM) |

We recommend a GPU with at least 8GB of VRAM。
以上の結果から、8GB以上のVRAMを搭載したGPUを推奨します。

---
## Directory Tree：フォルダ構造

```binary
.
├── config
│   └── mdet.yaml
├── logs
├── models
│   └── classifire
├── src
│   ├── __init__.py
│   ├── classifire
│   │   ├── dataset.py
│   │   ├── models
│   │   │   └── resnet.py
│   │   └── transforms
│   │       └── __init__.py
│   ├── megadetector  # commits on on Oct 12, 2022
│   ├── utils
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── tag.py
│   │   └── timer.py
│   ├── run_clip.py
│   ├── run_cls.py
│   ├── run_megadetector.py
│   ├── run_summary.py
│   └── runner.py
├── LICENSE
├── README.md
├── download_md_model.sh
├── environment.yml
├── exec_clip.py
├── exec_cls.py
├── exec_imgsummary.py
├── exec_mdet.py
├── exec_mdetcrop.py
└── exec_sample.sh
```
The scripts of src/megadetector is from [microsoft/CameraTraps](https://github.com/microsoft/CameraTraps) [commits on Oct 12, 2022](https://github.com/microsoft/CameraTraps/commit/33d1d9fa383e0935e8115b325e538811bd92b65f)