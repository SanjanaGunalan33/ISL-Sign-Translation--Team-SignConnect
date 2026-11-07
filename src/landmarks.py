"""Frame sampling + MediaPipe landmark detection.

Works with either MediaPipe API:
  * legacy  `mp.solutions.holistic` (mediapipe <= 0.10.21, e.g. Colab / Python 3.10-3.12)
  * Tasks   HandLandmarker + PoseLandmarker (newer mediapipe; needs .task model files,
            see scripts/download_models.sh)
"""
from pathlib import Path

import cv2
import numpy as np

from src import config


def sample_frames(video_path, target_fps=config.TARGET_FPS, max_frames=config.MAX_FRAMES):
    """Yield (timestamp_ms, rgb_frame) at ~target_fps, in order. Also returns meta via .meta attr."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:          # broken metadata
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(fps / target_fps, 1.0)
    frames, next_pick, i = [], 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            i += 1
            continue
        if i >= next_pick:
            frames.append((int(i * 1000 / fps), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            next_pick += step
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames, dict(fps=fps, width=w, height=h)


def _assign_hands(hands, pose_xy):
    """hands: list of (21,3) arrays. pose_xy: (33,3) or None.
    Returns (left, right) by matching hand wrists to the pose wrists."""
    left = right = None
    if not hands:
        return left, right
    if pose_xy is None or np.isnan(pose_xy[15, 0]):
        return None, None   # caller falls back to handedness labels
    lw, rw = pose_xy[15, :2], pose_xy[16, :2]
    if len(hands) == 1:
        h = hands[0]
        dl, dr = np.linalg.norm(h[0, :2] - lw), np.linalg.norm(h[0, :2] - rw)
        return (h, None) if dl <= dr else (None, h)
    a, b = hands[0], hands[1]
    cost_ab = np.linalg.norm(a[0, :2] - lw) + np.linalg.norm(b[0, :2] - rw)
    cost_ba = np.linalg.norm(b[0, :2] - lw) + np.linalg.norm(a[0, :2] - rw)
    return (a, b) if cost_ab <= cost_ba else (b, a)


class LandmarkExtractor:
    def __init__(self, models_dir=config.MODELS_DIR):
        import mediapipe as mp
        self.mp = mp
        self.mode = "legacy" if hasattr(mp, "solutions") else "tasks"
        if self.mode == "legacy":
            self.holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False, model_complexity=0, smooth_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks.python import BaseOptions, vision
            hand_task = Path(models_dir) / "hand_landmarker.task"
            pose_task = Path(models_dir) / "pose_landmarker_full.task"
            for p in (hand_task, pose_task):
                if not p.exists():
                    raise FileNotFoundError(f"{p} missing. Run: bash scripts/download_models.sh")
            self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(hand_task)),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.4, min_tracking_confidence=0.4))
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_task)),
                running_mode=vision.RunningMode.VIDEO, num_poses=1))
            self._t_offset = 0   # VIDEO mode needs strictly increasing timestamps across videos

    def close(self):
        if self.mode == "legacy":
            self.holistic.close()
        else:
            self.hand.close()
            self.pose.close()

    @staticmethod
    def _arr(lms, aspect, n):
        a = np.array([[p.x * aspect, p.y, p.z * aspect] for p in lms], dtype=np.float32)
        return a if a.shape[0] == n else None

    def extract(self, video_path):
        frames, meta = sample_frames(video_path)
        T = len(frames)
        aspect = meta["width"] / max(meta["height"], 1)
        left = np.full((T, 21, 3), np.nan, np.float32)
        right = np.full((T, 21, 3), np.nan, np.float32)
        pose = np.full((T, 33, 3), np.nan, np.float32)

        for t, (ts, rgb) in enumerate(frames):
            if self.mode == "legacy":
                r = self.holistic.process(rgb)
                if r.pose_landmarks:
                    pose[t] = self._arr(r.pose_landmarks.landmark, aspect, 33)
                if r.left_hand_landmarks:      # holistic already reports the subject's left
                    left[t] = self._arr(r.left_hand_landmarks.landmark, aspect, 21)
                if r.right_hand_landmarks:
                    right[t] = self._arr(r.right_hand_landmarks.landmark, aspect, 21)
            else:
                img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                stamp = self._t_offset + ts
                pr = self.pose.detect_for_video(img, stamp)
                hr = self.hand.detect_for_video(img, stamp)
                p = self._arr(pr.pose_landmarks[0], aspect, 33) if pr.pose_landmarks else None
                if p is not None:
                    pose[t] = p
                hands = [self._arr(h, aspect, 21) for h in hr.hand_landmarks]
                hands = [h for h in hands if h is not None]
                l, r_ = _assign_hands(hands, p)
                if l is None and r_ is None and hands:
                    # no pose: fall back to labels. Tasks labels assume a mirrored
                    # (selfie) image, so "Left" is the subject's right in normal video.
                    for h, cats in zip(hands, hr.handedness):
                        if cats[0].category_name == "Left":
                            r_ = h
                        else:
                            l = h
                if l is not None:
                    left[t] = l
                if r_ is not None:
                    right[t] = r_
        if self.mode == "tasks":
            self._t_offset += (frames[-1][0] if frames else 0) + 1000
        return dict(left=left, right=right, pose=pose, **meta)
