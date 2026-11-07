"""Step 2: match manifest paths to the video archive, extract, and test decoding.

The iSign videos come as split pieces (iSign-videos_v1.1_part_aa, _part_ab).
These are byte-level pieces of ONE archive (zip or tar), so neither piece is
usable alone. This script reads them as one continuous file without needing a
combined copy on disk.

Outputs data/video_index.csv:
    uid, text, local_path, status, fps, n_frames, width, height, duration_s
status is one of: ok, not_in_archive, decode_failed, empty

Usage:
    # quick check on 200 videos first
    python -m src.validate_videos --parts data/raw/iSign-videos_v1.1_part_aa data/raw/iSign-videos_v1.1_part_ab --limit 200
    # list archive contents only (no extraction)
    python -m src.validate_videos --parts ... --list-only
    # everything
    python -m src.validate_videos --parts ...
"""
import argparse
import io
import os
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import pandas as pd
from tqdm import tqdm

from src import config


class MultiPartFile(io.RawIOBase):
    """Read several files back to back as one seekable file."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.starts, total = [], 0
        for s in self.sizes:
            self.starts.append(total)
            total += s
        self.total = total
        self.pos = 0
        self.handles = [open(p, "rb") for p in self.paths]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.total + offset
        self.pos = max(0, min(self.pos, self.total))
        return self.pos

    def readinto(self, b):
        if self.pos >= self.total:
            return 0
        i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
        h = self.handles[i]
        h.seek(self.pos - self.starts[i])
        n = h.readinto(memoryview(b)[: min(len(b), self.sizes[i] - (self.pos - self.starts[i]))])
        self.pos += n
        return n

    def close(self):
        for h in self.handles:
            h.close()
        super().close()


def norm_path(p: str) -> str:
    p = str(p).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def safe_dest(root: Path, member: str) -> Path:
    parts = [x for x in PurePosixPath(norm_path(member)).parts if x not in ("..", ".", "")]
    return root.joinpath(*parts)


def open_archive(parts):
    raw = io.BufferedReader(MultiPartFile(parts), buffer_size=1 << 20)
    head = raw.peek(4)[:4]
    if head[:2] == b"PK":
        return "zip", zipfile.ZipFile(raw)
    raw.seek(0)
    try:
        return "tar", tarfile.open(fileobj=raw, mode="r|*")   # stream mode, works for .tar/.tar.gz
    except tarfile.TarError as e:
        raise SystemExit(f"Could not open {parts} as zip or tar: {e}. Check with `file` on the first part.")


def probe(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return dict(status="decode_failed")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return dict(status="empty", fps=fps, n_frames=n, width=w, height=h)
    return dict(status="ok", fps=fps, n_frames=n, width=w, height=h,
                duration_s=round(n / fps, 2) if fps > 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="archive pieces IN ORDER (aa, ab, ...)")
    ap.add_argument("--clean-csv", default=str(config.CLEAN_CSV))
    ap.add_argument("--video-dir", default=str(config.VIDEO_DIR))
    ap.add_argument("--out", default=str(config.VIDEO_INDEX_CSV))
    ap.add_argument("--limit", type=int, default=0, help="only process the first N manifest rows")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.clean_csv)
    if args.limit:
        df = df.head(args.limit)
    df["key"] = df["video_path_in_archive"].map(norm_path)
    wanted = dict(zip(df["key"], df["uid"]))
    wanted_base = {PurePosixPath(k).name: k for k in wanted}   # fallback match by file name
    video_dir = Path(args.video_dir)

    kind, arc = open_archive(sorted(args.parts) if all("_part_" in p for p in args.parts) else args.parts)
    print(f"Archive type: {kind}")

    found = {}   # manifest key -> local path
    n_members = 0

    def handle(name, read_fn):
        nonlocal n_members
        n_members += 1
        key = norm_path(name)
        if key not in wanted:
            key = wanted_base.get(PurePosixPath(key).name)
            if key is None:
                return
        dest = safe_dest(video_dir, key)
        if args.list_only:
            found[key] = dest
            return
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with read_fn() as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            os.replace(tmp, dest)
        found[key] = dest

    if kind == "zip":
        for info in tqdm(arc.infolist(), desc="zip members"):
            if not info.is_dir():
                handle(info.filename, lambda i=info: arc.open(i))
            if len(found) == len(wanted):
                break
    else:
        for m in tqdm(arc, desc="tar members"):
            if m.isfile():
                handle(m.name, lambda m=m: arc.extractfile(m))
            if len(found) == len(wanted):
                break

    print(f"Archive members scanned: {n_members:,}; manifest rows matched: {len(found):,} / {len(wanted):,}")
    if args.list_only:
        return

    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="decode check"):
        local = found.get(r.key)
        if local is None:
            rows.append(dict(uid=r.uid, text=r.text, local_path="", status="not_in_archive"))
            continue
        info = probe(local)
        rows.append(dict(uid=r.uid, text=r.text, local_path=str(local), **info))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out["status"].value_counts().to_string())
    ok = out[out.status == "ok"]
    if len(ok):
        print(f"\nfps: {ok.fps.describe()[['min','50%','max']].to_dict()}")
        print(f"duration_s: {ok.duration_s.describe()[['min','50%','max']].to_dict()}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
