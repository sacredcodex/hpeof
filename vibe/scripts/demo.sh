#!/usr/bin/env bash


source /data/anu_hpeof/vibe-flow/VIBE/vibe-env/bin/activate
export PYTHONPATH="./:$PYTHONPATH"

module add ffmpeg/latest
module load ffmpeg
module list
module unload cuda/11.2


cd /data/anu_hpeof/vibe-flow/VIBE

python /data/anu_hpeof/vibe-flow/VIBE/demo.py --vid_file /data/anu_hpeof/vibe-flow/VIBE/input_video/sample_video.mp4 --output_folder /data/anu_hpeof/vibe-flow/VIBE/output/ --no_render