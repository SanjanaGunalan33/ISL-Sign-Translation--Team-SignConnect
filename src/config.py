"""Shared paths and default settings for the ISL -> English pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Raw inputs (put the iSign files here, or pass paths on the command line)
RAW_DIR = DATA / "raw"
ANNOTATIONS_CSV = RAW_DIR / "iSign_v1.1.csv"
MANIFEST_CSV = RAW_DIR / "manifests" / "isign_manifest.csv"

# Generated files
CLEAN_CSV = DATA / "clean_annotations.csv"          # from inspect_dataset.py
VIDEO_DIR = DATA / "videos"                         # extracted videos
VIDEO_INDEX_CSV = DATA / "video_index.csv"          # from validate_videos.py
FEATURE_DIR = DATA / "features"                     # one <uid>.npy per video
FEATURE_INDEX_CSV = DATA / "feature_index.csv"      # from extract_features.py
SPLIT_DIR = DATA / "splits"                         # train/val/test csv + tokenizer + norm stats
MODELS_DIR = ROOT / "models"                        # MediaPipe .task files
RUNS_DIR = ROOT / "runs"                            # training checkpoints and logs

# Video sampling
TARGET_FPS = 10.0
MAX_FRAMES = 400            # hard cap after sampling (40 s at 10 fps)

# Feature layout (see features.py for the exact order)
N_HAND = 2 * 21 * 3          # 126
N_POSE = 9 * 3               # 27
N_GEOM = 15
N_DELTA = N_HAND + N_POSE    # 153
N_FLAGS = 2                  # left-hand present, right-hand present
FEATURE_DIM = N_HAND + N_POSE + N_GEOM + N_DELTA + N_FLAGS   # 323

# Special tokens (ids are fixed so word-level and SentencePiece agree)
PAD, BOS, EOS, UNK = 0, 1, 2, 3


def resolve(p, default_dir=SPLIT_DIR):
    """Find a saved file even if the project folder was moved (e.g. trained on Colab)."""
    from pathlib import Path as _P
    p = _P(p)
    return p if p.exists() else _P(default_dir) / p.name
