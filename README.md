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
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    bash Miniconda3-latest-Linux-x86_64.sh
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
    conda env create -f=environment.yml
    conda activate mdet
    ```

#### Requirement

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
