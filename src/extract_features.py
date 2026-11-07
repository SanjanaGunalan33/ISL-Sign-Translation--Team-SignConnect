"""Step 3: run MediaPipe on every usable video and save raw landmarks.

Saves data/features/<uid>.npz (left, right, pose, fps, width, height) and
appends one line per video to data/features/log_*.jsonl. Re-running skips
videos already done, so it's safe to stop and resume. At the end it writes
data/feature_index.csv from all log files.

Usage:
    python -m src.extract_features --limit 50             # small sample first
    python -m src.extract_features --workers 4
    python -m src.extract_features --shard 0/3            # split work across machines/sessions
    python -m src.extract_features --index-only           # just rebuild feature_index.csv
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config

_extractor = None


def _init_worker():
    global _extractor
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from src.landmarks import LandmarkExtractor
    _extractor = LandmarkExtractor()


def _process(job):
    uid, video_path, out_dir = job
    out = Path(out_dir) / f"{uid}.npz"
    t0 = time.time()
    try:
        d = _extractor.extract(video_path)
        T = d["pose"].shape[0]
        if T == 0:
            return dict(uid=uid, status="no_frames")
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, left=d["left"].astype(np.float16), right=d["right"].astype(np.float16),
                            pose=d["pose"].astype(np.float16), fps=d["fps"], width=d["width"], height=d["height"])
        os.replace(tmp, out)
        return dict(uid=uid, status="ok", npz=str(out), n_frames=T,
                    pose_rate=float((~np.isnan(d["pose"][:, 0, 0])).mean()),
                    left_rate=float((~np.isnan(d["left"][:, 0, 0])).mean()),
                    right_rate=float((~np.isnan(d["right"][:, 0, 0])).mean()),
                    secs=round(time.time() - t0, 2))
    except Exception as e:  # log and move on; never silently drop
        return dict(uid=uid, status="error", error=f"{type(e).__name__}: {e}")


def build_index(feat_dir: Path, out_csv: Path):
    logs = sorted(feat_dir.glob("log_*.jsonl"))
    rows = [json.loads(l) for f in logs for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        print("No logs found.")
        return
    df = pd.DataFrame(rows).drop_duplicates("uid", keep="last")
    df.to_csv(out_csv, index=False)
    print(df["status"].value_counts().to_string())
    ok = df[df.status == "ok"]
    if len(ok):
        print(f"\nframes/video median {ok.n_frames.median():.0f}, p95 {ok.n_frames.quantile(.95):.0f}")
        print(f"detection rate  pose {ok.pose_rate.mean():.2%}  left {ok.left_rate.mean():.2%}  right {ok.right_rate.mean():.2%}")
        low = (ok.pose_rate < 0.5).sum()
        if low:
            print(f"WARNING: {low} videos have pose detected in <50% of frames — inspect a few.")
    print(f"Wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-index", default=str(config.VIDEO_INDEX_CSV))
    ap.add_argument("--out-dir", default=str(config.FEATURE_DIR))
    ap.add_argument("--index-out", default=str(config.FEATURE_INDEX_CSV))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th video starting at i")
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        build_index(out_dir, Path(args.index_out))
        return

    vids = pd.read_csv(args.video_index)
    vids = vids[vids.status == "ok"].reset_index(drop=True)
    i, n = map(int, args.shard.split("/"))
    vids = vids.iloc[i::n]
    if args.limit:
        vids = vids.head(args.limit)
    jobs = [(r.uid, r.local_path, str(out_dir)) for r in vids.itertuples()
            if not (out_dir / f"{r.uid}.npz").exists()]
    print(f"{len(vids):,} videos in this shard, {len(jobs):,} still to process, {args.workers} workers")

    log_path = out_dir / f"log_{i}of{n}.jsonl"
    with open(log_path, "a") as log:
        if args.workers <= 1:
            _init_worker()
            it = map(_process, jobs)
            for res in tqdm(it, total=len(jobs)):
                log.write(json.dumps(res) + "\n"); log.flush()
        else:
            with Pool(args.workers, initializer=_init_worker) as pool:
                for res in tqdm(pool.imap_unordered(_process, jobs, chunksize=4), total=len(jobs)):
                    log.write(json.dumps(res) + "\n"); log.flush()
    build_index(out_dir, Path(args.index_out))


if __name__ == "__main__":
    main()
