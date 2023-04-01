### For using MegaDetector v5.0

# mkdir ~/git
# cd ~/git
# git clone https://github.com/ecologize/yolov5/
# git clone https://github.com/Microsoft/cameratraps
# git clone https://github.com/Microsoft/ai4eutils
# export PYTHONPATH="$PYTHONPATH:$HOME/git/ai4eutils:$HOME/git/yolov5"

###Run Megadetector v5.0 

# python exec_mdet.py session_root=${video_dir}-clip mdet_config.model_path=./models/md_v5a.0.0.pt
