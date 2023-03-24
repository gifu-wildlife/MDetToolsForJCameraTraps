# MDetToolsForJCameraTraps

カメラトラップ画像/映像向けMegaDetector実行スクリプト．

## Get Started

### Prerequisites

* NVIDIA Driver

    ```bash
    sudo apt install nvidia-driver-***
    ```

    Please refer to [NVIDIA Driver Version Check](https://www.nvidia.com/Download/index.aspx?lang=en-us).
    *** is a placeholder. Please enter the recommended nvidia driver version.  

    check installation.

    ```bash
    nvidia-smi  # NVIDIA Driver installation check
    ```

    If nvidia-smi does not work, Try Rebooting.

* Conda

    Download installer and run the script for Unix-like platform.  

    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"
    bash Mambaforge-Linux-x86_64.sh
    # If you are using miniconda for the first time, Please ansew "yes" to "Do you wish the installer to initialize Miniconda3 by running conda init?" 
    source ~/.bashrc
    ```

    For more information, please refer to [miniconda official](https://docs.conda.io/en/latest/miniconda.html)

### Instllation

1. Clone the Repository

    ```bash
    git clone https://github.com/gifu-wildlife/MDetToolsForJCameraTraps.git
    ```

    or Download ZIP and Unzip in any directory of yours

    ![Screenshot from 2022-11-11 13-07-09](https://user-images.githubusercontent.com/50891743/201261079-74254fd8-ce4f-4a0f-9085-3a5209d40f7c.png)

2. Move Project Directory

    ```bash
    cd MDetToolsForJCameraTraps
    ```

    or

    ```bash
    cd MDetToolsForJCameraTraps-main
    ```

3. create conda environment.

    ```bash
    mamba env create -f=environment.yml
    conda activate mdet
    ```

#### Requirement

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

## Usage

1. Download MegaDetector weight file.

    ```bash
    bash download_md_model.sh
    ```

2. Run Script

* Movie Clip

    ```bash
    python exec_clip.py session_root=${video_dir} output_dir=${video_dir}-clip
    ```

* Run MegaDetector

    ```bash
    # python exec_mdet.py session_root=${video_dir}-clip mdet_config.model_path=./models/md_v5a.0.0.pt
    python exec_mdet.py session_root=${video_dir}-clip mdet_config.model_path=./models/md_v4.1.0.pb
    ```

* Bounding Box Crop

    ```bash
    python exec_mdetcrop.py session_root=${video_dir}-clip mdet_result_path=${video_dir}-clip/detector_output.json
    ```

* Classification

    ```bash
    python exec_cls.py session_root=${video_dir}-clip-crop
    ```

* Summarize

    ```bash
    python exec_imgsummary.py session_root=${video_dir}-clip-crop mdet_result_path=${video_dir}-clip/detector_output.json
    ```

### Parameter

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

## Verified GPU

| GPU | VRAM | Result |
| ---- | ---- | ----|
| RTX 3090 | 24GB | Success |
| RTX 3080 | 10GB | Success |
| RTX 3060 | 12GB | Success |
| RTX 3050 | 8GB | Success |
| GTX 1080 | 8GB | Success |
| RTX 3050 (Notebooks) | 4GB | Failure (Insufficient VRAM) |

We recommend a GPU with at least 8GB of VRAM

## Directory Tree

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
src/megadetector is for [microsoft/CameraTraps](https://github.com/microsoft/CameraTraps) [commits on Oct 12, 2022](https://github.com/microsoft/CameraTraps/commit/33d1d9fa383e0935e8115b325e538811bd92b65f)