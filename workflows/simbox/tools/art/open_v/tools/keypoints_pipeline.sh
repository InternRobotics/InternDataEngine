#!/bin/bash
CONFIG_PATH="YOUR_PATH_TO/keypoints_config.json"

cd workflows/simbox/tools/art/open_v/tools

# 1. rehier
python rehier.py --config $CONFIG_PATH

# 2. select points
python select_keypoint.py --config $CONFIG_PATH

# 3. Transfer keypoints
python transfer_keypoints.py --config $CONFIG_PATH

# 4. Overwrite keypoints
python overwrite_keypoints.py --config $CONFIG_PATH