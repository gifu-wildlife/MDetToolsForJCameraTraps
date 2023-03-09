# MDetToolsForJCameraTraps

カメラトラップ画像/映像向けMegaDetector実行スクリプト．

## GPU Environment Setup

Please refer to [NVIDIA Driver Version Check](https://www.nvidia.com/Download/index.aspx?lang=en-us).  
and Install nvidia-driver.
```
sudo apt install nvidia-driver-***
```
*** is a placeholder. Please enter the recommended nvidia driver version.  

check installation.
```bash
nvidia-smi  # NVIDIA Driver installation check
```

## Requirement

- python=3.9
- pytorch-gpu==1.10.1
- torchvision==0.11.2
- cudatoolkit=11.3
- pandas
- omegaconf
- tqdm
- opencv
- tensorflow
- humanfriendly
- ca-certificates
- certifi
- openssl
- matplotlib
- jsonpickle

Environments under [miniconda](https://docs.conda.io/en/latest/miniconda.html)  
環境構築はcondaを用いて行う。  

### conda installation
#### for Unix-like platform

download installer and run the script.  
```bash
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash Miniconda3-latest-Linux-x86_64.sh
# If you are using miniconda for the first time, Please answer "yes" to "Do you wish the installer to initialize Miniconda3 by running conda init?" 
source ~./.bashrc
```

For more information, please refer to [miniconda official](https://docs.conda.io/en/latest/miniconda.html)   

### Environment for Script Setup

Project Repository Download

```bash
git clone https://github.com/gifu-wildlife/MDetToolsForJCameraTraps.git
```

or Download ZIP and Unzip in any directory of yours

![Screenshot from 2022-11-11 13-07-09](https://user-images.githubusercontent.com/50891743/201261079-74254fd8-ce4f-4a0f-9085-3a5209d40f7c.png)

#### Create Python Enviroment

Move Project Directory.

```bash
cd MDetToolsForJCameraTraps
```

or

```bash
cd MDetToolsForJCameraTraps-main
```

create conda environment.

```bash
conda env create -f=environment.yml
conda activate mdet
```

## Usage

1. Download MegaDetector weight file.
```bash
bash download_md_model.sh
```
2. Run Script 
```bash
python exec_clip.py session_root=${video_dir} output_dir=${video_dir}-clip
```
```bash
# python exec_mdet.py session_root=${video_dir}-clip mdet_config.model_path=./models/md_v5a.0.0.pt
python exec_mdet.py session_root=${video_dir}-clip mdet_config.model_path=./models/md_v4.1.0.pb
```
```bash
python exec_mdetcrop.py session_root=${video_dir}-clip mdet_result_path=${video_dir}-clip/detector_output.json
```
```bash
python exec_cls.py session_root=${video_dir}-clip-crop
```
```bash
python exec_imgsummary.py session_root=${video_dir}-clip-crop mdet_result_path=${video_dir}-clip/detector_output.json
```

