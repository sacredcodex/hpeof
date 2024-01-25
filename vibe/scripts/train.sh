#!/usr/bin/env bash

source /data/anu_hpeof/vibe-flow/VIBE/vibe-env/bin/activate
export PYTHONPATH="./:$PYTHONPATH"

module add ffmpeg/latest
module load ffmpeg
module add protobuf/3.5.0
module list

cd /data/anu_hpeof/vibe-flow/VIBE

python /data/anu_hpeof/vibe-flow/VIBE/train.py --cfg configs/config.yaml