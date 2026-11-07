"""Live (near-real-time) ISL -> English from a webcam, or a video clip played on screen.

Webcam mode:
  * Runs MediaPipe Holistic on the webcam and samples ~10 fps (as in training).
  * Recording starts when a hand appears; the sentence is translated once no
    hand has been seen for --pause seconds (lower your hands to finish).
  * SPACE starts/stops recording manually. Q or ESC quits.

Clip mode (--source video.mp4):
  * Plays the clip with the landmarks drawn on it and translates the WHOLE
    clip at the end, processing frames exactly as evaluation does, so the
    result matches the prediction in runs/pilot/test_beam1_predictions.csv.

The model reads a whole signed sentence at once (bidirectional encoder), so
translation happens at the end of each sentence, not word by word.

Usage:
    python -m src.live_translate --ckpt runs/pilot/best.pt
    python -m src.live_translate --ckpt runs/pilot/best.pt --camera 1
    python -m src.live_translate --ckpt runs/pilot/best.pt --source data/videos/.../clip.mp4
"""
import argparse
import threading
import time

import cv2
import numpy as np

from src import config
from src.translate_video import Translator


def speak_async(text):
    """Speak without freezing the video window. On Windows, use the built-in
    speech engine directly: pyttsx3 often fails inside background threads there."""
    import platform
    from src.speak_translation import _speak_windows, speak
    fn = _speak_windows if platform.system() == "Windows" else speak
    threading.Thread(target=fn, args=(text,), daemon=True).start()


def to_array(lms, aspect, n):
    """Same conversion as landmarks.py: x and z scaled by width/height."""
    if lms is None:
        return np.full((n, 3), np.nan, np.float32)
    return np.array([[p.x * aspect, p.y, p.z * aspect] for p in lms.landmark], dtype=np.float32)


def put_text(img, text, y, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def wrap(text, width=40):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/pilot/best.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", default="", help="video clip to play and translate")
    ap.add_argument("--pause", type=float, default=1.0, help="webcam: seconds without hands that end a sentence")
    ap.add_argument("--min-frames", type=int, default=8, help="webcam: ignore segments shorter than this (at 10 fps)")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--manual", action="store_true", help="webcam: only SPACE starts/stops recording")
    args = ap.parse_args()

    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise SystemExit("Live mode needs mediapipe==0.10.14 (legacy Holistic API). Run: pip install mediapipe==0.10.14")
    # same settings as feature extraction (src/landmarks.py)
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=0,
                                              smooth_landmarks=True, min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
    draw, styles = mp.solutions.drawing_utils, mp.solutions.holistic

    print("Loading model...")
    tr = Translator(args.ckpt)
    file_mode = bool(args.source)
    cap = cv2.VideoCapture(args.source if file_mode else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera/video. Try --camera 1, or check the file path.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    frame_step = max(fps / config.TARGET_FPS, 1.0)      # clip mode: same frame picking as sample_frames()
    time_step = 1.0 / config.TARGET_FPS                  # webcam mode: sample by clock

    buf = {"left": [], "right": [], "pose": []}
    recording = file_mode                                # clip mode records the whole clip
    last_sample = last_hand_time = rec_start = 0.0
    next_pick, frame_i = 0.0, 0
    status = "Playing clip..." if file_mode else "Waiting for hands..."
    result, res, view = "", None, None
    print("Ready." + ("" if file_mode else " Sign in front of the camera; lower your hands to finish. SPACE = start/stop, Q = quit."))

    def translate(trim_tail):
        nonlocal status, result, buf
        has_hand = [not (np.isnan(l[0, 0]) and np.isnan(r[0, 0])) for l, r in zip(buf["left"], buf["right"])]
        keep = len(has_hand)
        if trim_tail:                                     # webcam: drop the pause at the end
            while keep and not has_hand[keep - 1]:
                keep -= 1
        if keep < (1 if file_mode else args.min_frames):
            status = f"Too short ({keep} frames) - try again"
        else:
            t0 = time.time()
            text = tr.translate_landmarks(np.stack(buf["left"][:keep]), np.stack(buf["right"][:keep]),
                                          np.stack(buf["pose"][:keep]), beam=args.beam) or "(no translation)"
            result = text
            status = f"Translated {keep} frames in {time.time() - t0:.2f}s"
            print(f"> {text}")
            if not args.no_speak and text != "(no translation)":
                speak_async(text)
        buf = {"left": [], "right": [], "pose": []}

    def render(frame):
        h, w = frame.shape[:2]
        v = frame.copy()
        if res is not None:
            draw.draw_landmarks(v, res.pose_landmarks, styles.POSE_CONNECTIONS)
            draw.draw_landmarks(v, res.left_hand_landmarks, styles.HAND_CONNECTIONS)
            draw.draw_landmarks(v, res.right_hand_landmarks, styles.HAND_CONNECTIONS)
        if not file_mode:
            v = cv2.flip(v, 1)                             # mirror the webcam like a selfie view
        scale = max(h / 480, 0.6)
        if recording:
            cv2.circle(v, (w - 30, 30), 12, (0, 0, 255), -1)
        put_text(v, status, int(32 * scale), 0.7 * scale, (0, 255, 255))
        lines = wrap(result)
        for i, line in enumerate(lines):
            put_text(v, line, h - int(20 * scale) - int(34 * scale) * (len(lines) - 1 - i), 0.95 * scale, (255, 255, 255), 2)
        return v

    while True:
        ok, frame = cap.read()
        if not ok:
            if file_mode:
                recording = False
                status = "Translating..."
                translate(trim_tail=False)
                status += "  |  press any key to close"
                if view is not None:
                    cv2.imshow("ISL -> English", render(last_frame))
                cv2.waitKey(0)
                break
            continue
        last_frame = frame
        aspect = frame.shape[1] / max(frame.shape[0], 1)

        if file_mode:
            # exactly like evaluation: only every frame_step-th frame goes to MediaPipe
            take = frame_i >= next_pick and len(buf["pose"]) < config.MAX_FRAMES
            if take:
                next_pick += frame_step
                res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buf["left"].append(to_array(res.left_hand_landmarks, aspect, 21))
                buf["right"].append(to_array(res.right_hand_landmarks, aspect, 21))
                buf["pose"].append(to_array(res.pose_landmarks, aspect, 33))
                status = f"Playing clip... {len(buf['pose'])} frames"
            frame_i += 1
            wait = max(1, int(1000 / fps))
        else:
            now = time.time()
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # model sees the un-mirrored image
            hand_seen = res.left_hand_landmarks is not None or res.right_hand_landmarks is not None
            if hand_seen:
                last_hand_time = now
            if not args.manual:
                if not recording and hand_seen:
                    recording, rec_start, last_sample = True, now, 0.0
                elif recording and now - last_hand_time > args.pause:
                    recording = False
                    translate(trim_tail=True)
            if recording and now - rec_start > args.max_seconds:
                recording = False
                translate(trim_tail=True)
            if recording and now - last_sample >= time_step:
                last_sample = now
                buf["left"].append(to_array(res.left_hand_landmarks, aspect, 21))
                buf["right"].append(to_array(res.right_hand_landmarks, aspect, 21))
                buf["pose"].append(to_array(res.pose_landmarks, aspect, 33))
            if recording:
                status = f"Recording... {len(buf['pose'])} frames"
            wait = 1

        view = render(frame)
        cv2.imshow("ISL -> English", view)
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == 32 and not file_mode:                    # SPACE
            if recording:
                recording = False
                translate(trim_tail=True)
            else:
                recording, rec_start, last_sample = True, time.time(), 0.0
                buf = {"left": [], "right": [], "pose": []}

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()