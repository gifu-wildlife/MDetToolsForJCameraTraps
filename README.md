# MDetToolsForJCameraTraps

カメラトラップ画像/映像向けMegaDetector実行スクリプト．

## GPU Environment Setup

Please refer to [NVIDIA Driver Installation](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network).  
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
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh.sh"
bash Miniconda3-latest-Linux-x86_64.sh.sh
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

