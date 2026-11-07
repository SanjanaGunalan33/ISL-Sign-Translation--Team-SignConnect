"""Turn raw MediaPipe landmarks into the per-frame feature vector.

Raw landmarks are saved by extract_features.py as <uid>.npz with:
    left  (T, 21, 3)  subject's left hand,  NaN where not detected
    right (T, 21, 3)  subject's right hand, NaN where not detected
    pose  (T, 33, 3)  full MediaPipe pose,  NaN where not detected
    fps, width, height
x is multiplied by width/height at extraction time so x and y share units.

Per-frame feature layout (323 values):
    [0:63]     left hand 21 x (x,y,z), body-normalized          } 126
    [63:126]   right hand
    [126:153]  9 pose points x (x,y,z): nose, shoulders, elbows, wrists, hips
    [153:168]  15 geometric features (listed in GEOM_NAMES)
    [168:321]  frame-to-frame deltas of [0:153]
    [321:323]  left/right hand present flags

Keeping features separate from detection means you can change this file
(or run ablations) without re-running MediaPipe on 100k+ videos.
"""
import numpy as np

from src import config

# MediaPipe pose indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 11, 12, 13, 14, 15, 16, 23, 24
POSE_SEL = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP]
# Hand indices
H_WRIST, H_THUMB_TIP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_TIP = 0, 4, 8, 9, 20

GEOM_NAMES = [
    "wrist_to_wrist",
    "l_wrist_to_nose", "r_wrist_to_nose",
    "l_wrist_to_thumb_tip", "l_wrist_to_index_tip", "l_wrist_to_pinky_tip",
    "r_wrist_to_thumb_tip", "r_wrist_to_index_tip", "r_wrist_to_pinky_tip",
    "l_elbow_angle", "r_elbow_angle",
    "l_hand_orient_sin", "l_hand_orient_cos",
    "r_hand_orient_sin", "r_hand_orient_cos",
]
assert len(GEOM_NAMES) == config.N_GEOM

# Slices, used for ablations (e.g. drop geometry or deltas)
SLICES = {
    "hands": slice(0, 126),
    "pose": slice(126, 153),
    "geom": slice(153, 168),
    "delta": slice(168, 321),
    "flags": slice(321, 323),
}


def _interp_nan(a: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Linearly interpolate NaN frames along axis 0 (per column). Gaps longer
    than max_gap stay NaN. Leading/trailing NaNs are filled with nearest value
    only when max_gap is None."""
    a = a.copy()
    T = a.shape[0]
    flat = a.reshape(T, -1)
    valid = ~np.isnan(flat[:, 0])
    if valid.sum() == 0 or valid.all():
        return a
    idx = np.arange(T)
    vi = idx[valid]
    for c in range(flat.shape[1]):
        flat[~valid, c] = np.interp(idx[~valid], vi, flat[valid, c])
    if max_gap is not None:
        # undo filling for gaps that are too long or at the edges
        missing = ~valid
        t = 0
        while t < T:
            if missing[t]:
                s = t
                while t < T and missing[t]:
                    t += 1
                edge = s == 0 or t == T
                if edge or (t - s) > max_gap:
                    flat[s:t] = np.nan
            else:
                t += 1
    return flat.reshape(a.shape)


def _dist(a, b):
    return np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)


def _angle(a, b, c):
    """Angle at b (radians / pi, so 0..1)."""
    v1, v2 = a[..., :2] - b[..., :2], c[..., :2] - b[..., :2]
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6)
    return np.arccos(np.clip(cos, -1, 1)) / np.pi


def build_features(left, right, pose, hand_gap: int = 2) -> np.ndarray:
    """Return a (T, 323) float32 array. Inputs are (T,21,3), (T,21,3), (T,33,3) with NaN for missing."""
    T = pose.shape[0]
    left, right, pose = (np.asarray(x, dtype=np.float32) for x in (left, right, pose))

    # Pose: fill short and long gaps (body barely moves; a missing body is rare)
    pose = _interp_nan(pose)
    if np.isnan(pose).all():
        pose = np.zeros_like(pose)
        pose[:, L_SH, :2], pose[:, R_SH, :2] = (0.4, 0.4), (0.6, 0.4)   # neutral fallback
    # Hands: fill only very short dropouts; longer gaps really are "not visible"
    left = _interp_nan(left, max_gap=hand_gap)
    right = _interp_nan(right, max_gap=hand_gap)

    # Body reference: shoulder midpoint, shoulder width as unit length
    ref = (pose[:, L_SH] + pose[:, R_SH]) / 2                       # (T,3)
    scale = np.linalg.norm(pose[:, L_SH, :2] - pose[:, R_SH, :2], axis=-1)
    scale = np.where(scale < 1e-4, np.nanmedian(scale) if np.nanmedian(scale) > 1e-4 else 1.0, scale)
    ref[:, 2] = 0.0   # z from MediaPipe is already relative; only shift x,y

    def norm(pts):
        return (pts - ref[:, None, :]) / scale[:, None, None]

    pose_n = norm(pose)
    left_n, right_n = norm(left), norm(right)
    l_ok = ~np.isnan(left_n[:, 0, 0])
    r_ok = ~np.isnan(right_n[:, 0, 0])
    left_n = np.nan_to_num(left_n, nan=0.0)
    right_n = np.nan_to_num(right_n, nan=0.0)

    hands = np.concatenate([left_n.reshape(T, -1), right_n.reshape(T, -1)], axis=1)   # 126
    pose9 = pose_n[:, POSE_SEL].reshape(T, -1)                                        # 27

    # Geometric features (pose-based ones always available; hand-based ones 0 when missing)
    g = np.zeros((T, config.N_GEOM), dtype=np.float32)
    g[:, 0] = _dist(pose_n[:, L_WR], pose_n[:, R_WR])
    g[:, 1] = _dist(pose_n[:, L_WR], pose_n[:, NOSE])
    g[:, 2] = _dist(pose_n[:, R_WR], pose_n[:, NOSE])
    for k, (hand, ok) in enumerate(((left_n, l_ok), (right_n, r_ok))):
        for j, tip in enumerate((H_THUMB_TIP, H_INDEX_TIP, H_PINKY_TIP)):
            g[:, 3 + 3 * k + j] = np.where(ok, _dist(hand[:, H_WRIST], hand[:, tip]), 0.0)
        v = hand[:, H_MIDDLE_MCP, :2] - hand[:, H_WRIST, :2]
        ang = np.arctan2(v[:, 1], v[:, 0])
        g[:, 11 + 2 * k] = np.where(ok, np.sin(ang), 0.0)
        g[:, 12 + 2 * k] = np.where(ok, np.cos(ang), 0.0)
    g[:, 9] = _angle(pose_n[:, L_SH], pose_n[:, L_EL], pose_n[:, L_WR])
    g[:, 10] = _angle(pose_n[:, R_SH], pose_n[:, R_EL], pose_n[:, R_WR])

    # Temporal deltas of hands + pose; zero at t=0 and wherever a hand appears/disappears
    base = np.concatenate([hands, pose9], axis=1)        # 153
    delta = np.zeros_like(base)
    delta[1:] = base[1:] - base[:-1]
    l_pair = np.r_[False, l_ok[1:] & l_ok[:-1]]
    r_pair = np.r_[False, r_ok[1:] & r_ok[:-1]]
    delta[~l_pair, 0:63] = 0.0
    delta[~r_pair, 63:126] = 0.0

    flags = np.stack([l_ok, r_ok], axis=1).astype(np.float32)
    feats = np.concatenate([hands, pose9, g, delta, flags], axis=1).astype(np.float32)
    assert feats.shape[1] == config.FEATURE_DIM, feats.shape
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def load_features(npz_path) -> np.ndarray:
    d = np.load(npz_path)
    return build_features(d["left"], d["right"], d["pose"])


def ablation_mask(drop: list[str]) -> np.ndarray:
    """Boolean mask over the 323 dims; False for dropped groups."""
    m = np.ones(config.FEATURE_DIM, dtype=bool)
    for name in drop:
        m[SLICES[name]] = False
    return m
