#!/usr/bin/env bash
# Only needed for newer mediapipe versions (no mp.solutions). Downloads the two
# MediaPipe Tasks models used by src/landmarks.py into ./models/
set -e
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
echo "Saved to models/"
