# MDetToolsForJCameraTraps

カメラトラップ画像/映像向けMegaDetector実行スクリプト．

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

Environments under [miniforge](https://github.com/conda-forge/miniforge) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)  
環境構築はcondaを用いて行う。  

### conda installation
#### for Unix-like platform

download installer and run the script.  
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

For more information, please refer to [miniforge official](https://github.com/conda-forge/miniforge) or [miniconda official](https://docs.conda.io/en/latest/miniconda.html)   

### Run Environment Setup

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
```