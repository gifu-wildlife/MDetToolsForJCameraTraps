# mkdir ~/git
# cd ~/git
# git clone https://github.com/ecologize/yolov5/
# git clone https://github.com/Microsoft/cameratraps
# git clone https://github.com/Microsoft/ai4eutils
# export PYTHONPATH="$PYTHONPATH:$HOME/git/ai4eutils:$HOME/git/yolov5"

cd ~/MDetToolsForJCameraTraps
video_dir="/home/data_ssd/TEST-R3_Kinkazan_Boar_REST"

# python exec_clip.py session_root=${video_dir} output_dir=${video_dir}-clip
python exec_mdet.py session_root=${video_dir}-clip mdet_config.model_path=./models/md_v5a.0.0.pt
# python exec_mdetcrop.py session_root=${video_dir}-clip mdet_result_path=${video_dir}-clip/detector_output.json
# python exec_cls.py session_root=${video_dir}-clip-crop
# python exec_imgsummary.py session_root=${video_dir}-clip-crop mdet_result_path=${video_dir}-clip/detector_output.json