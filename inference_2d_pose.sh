#!/bin/bash

# Assuming the hdVideos directory is in the current working directory
HD_VIDEOS_DIR=/nas/database/panoptic/160906_band4/hdVideos

# Output directories
BASE_OUTPUT_DIR=./data/panoptic/160906_band4
VIS_OUT_DIR=$BASE_OUTPUT_DIR/prediction_vis
PRED_OUT_DIR=$BASE_OUTPUT_DIR/prediction_data

# Create output directories if they don't exist
mkdir -p "$VIS_OUT_DIR"
mkdir -p "$PRED_OUT_DIR"

# Iterate over all video files in the hdVideos directory
for video_file in "$HD_VIDEOS_DIR"/*.mp4; do
    # Extract video name without extension
    video_name=$(basename "${video_file%.*}")
    
    echo "Processing video: $video_name"
    
    # Run inference_demo.py for each video
    python3 demo/inferencer_demo.py "$video_file" \
        --pose2d human \
        --draw-bbox \
        --vis-out-dir "$VIS_OUT_DIR/$video_name" \
        --pred-out-dir "$PRED_OUT_DIR/$video_name"
done

echo "Inference on all videos completed."
