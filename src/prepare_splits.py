"""Step 4: leak-free train/val/test split, tokenizer and feature normalization stats.

Leakage control:
  * Clips cut from the same source video are kept in the same split. The source
    is taken from the uid (default: drop a trailing "-<number>"; change with
    --group-regex after looking at real uids).
  * Clips with the SAME English sentence are also kept together, so the test set
    never contains a sentence the model saw during training. If that merging
    creates one giant group (very common sentences), it falls back to source
    grouping and removes val/test rows whose sentence appears in train.

Outputs in data/splits/: train.csv, val.csv, test.csv, tokenizer files,
norm.npz (feature mean/std from train), split_info.json.

Usage:
    python -m src.prepare_splits
    python -m src.prepare_splits --tokenizer word
    python -m src.prepare_splits --use-official official_split.csv   # if a published split exists (uid,split)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.features import load_features
from src.tokenizer import Tokenizer


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def group_split(groups: pd.Series, fracs, seed):
    """Assign whole groups to splits so each split gets ~fracs of the rows."""
    rng = np.random.default_rng(seed)
    sizes = groups.value_counts()
    order = sizes.index.to_numpy().copy()     # newer pandas returns a read-only view
    rng.shuffle(order)
    targets = np.array(fracs) * len(groups)
    filled = np.zeros(3)
    assign = {}
    for g in order:
        k = int(np.argmax(targets - filled))     # the split furthest below its target
        assign[g] = k
        filled[k] += sizes[g]
    return groups.map(assign).map({0: "train", 1: "val", 2: "test"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-csv", default=str(config.CLEAN_CSV))
    ap.add_argument("--feature-index", default=str(config.FEATURE_INDEX_CSV))
    ap.add_argument("--out-dir", default=str(config.SPLIT_DIR))
    ap.add_argument("--group-regex", default=r"[-_]\d+$", help="removed from uid to get the source-video id")
    ap.add_argument("--fracs", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", choices=["spm", "word"], default="spm")
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--max-giant-group", type=float, default=0.10)
    ap.add_argument("--use-official", default="", help="csv with uid,split to use instead")
    ap.add_argument("--per-sentence", action="store_true",
                    help="limited-vocabulary demo: split each sentence's clips across train/val/test "
                         "(test clips are unseen videos of sentences seen in training)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(args.clean_csv)
    feats = pd.read_csv(args.feature_index)
    feats = feats[feats.status == "ok"][["uid", "npz", "n_frames"]]
    df = ann.merge(feats, on="uid", how="inner")
    df = df[df.n_frames >= args.min_frames].reset_index(drop=True)
    print(f"Usable examples (have text + landmarks): {len(df):,}")

    info = {"n_examples": len(df), "seed": args.seed}
    if args.use_official:
        off = pd.read_csv(args.use_official)[["uid", "split"]]
        df = df.merge(off, on="uid", how="inner")
        info["split_source"] = args.use_official
    elif args.per_sentence:
        info["strategy"] = "per sentence: each sentence's clips split across train/val/test"
        fracs = [float(x) for x in args.fracs.split(",")]
        rng = np.random.default_rng(args.seed)
        split = pd.Series("train", index=df.index)
        for _, idx in df.groupby("text").groups.items():
            idx = np.array(idx)
            rng.shuffle(idx)
            n = len(idx)
            n_test = max(1, round(fracs[2] * n)) if n >= 3 else 0
            n_val = max(1, round(fracs[1] * n)) if n >= 3 else 0
            split[idx[:n_test]] = "test"
            split[idx[n_test:n_test + n_val]] = "val"
        df["split"] = split
    else:
        df["source"] = df["uid"].astype(str).str.replace(args.group_regex, "", regex=True)
        info["n_sources"] = int(df.source.nunique())
        print(f"Source videos (from uid): {info['n_sources']:,}  "
              f"(if this equals the row count, the regex isn't finding a source id — check uids)")

        # merge sources that share an identical sentence
        dsu = DSU(len(df))
        for col in ("source", "text"):
            first = {}
            for i, v in enumerate(df[col]):
                if v in first:
                    dsu.union(i, first[v])
                else:
                    first[v] = i
        df["group"] = [dsu.find(i) for i in range(len(df))]
        biggest = df.group.value_counts().iloc[0] / len(df)
        info["largest_group_frac"] = round(float(biggest), 4)
        fracs = [float(x) for x in args.fracs.split(",")]
        if biggest <= args.max_giant_group:
            info["strategy"] = "group by source + identical sentence"
            df["split"] = group_split(df.group, fracs, args.seed)
        else:
            info["strategy"] = "group by source, then drop val/test rows whose sentence is in train"
            print(f"Sentence merging makes a group with {biggest:.1%} of data -> fallback strategy")
            df["split"] = group_split(df.source, fracs, args.seed)
            train_texts = set(df.loc[df.split == "train", "text"])
            leak = (df.split != "train") & df.text.isin(train_texts)
            info["dropped_for_text_overlap"] = int(leak.sum())
            df = df[~leak]
    print(f"Strategy: {info.get('strategy', 'official')}")

    for s in ("train", "val", "test"):
        part = df[df.split == s]
        cols = [c for c in ("uid", "text", "npz", "n_frames", "source") if c in part.columns]
        part[cols].to_csv(out / f"{s}.csv", index=False)
        info[f"n_{s}"] = len(part)
    tr = set(df.loc[df.split == "train", "text"])
    info["test_sentences_seen_in_train"] = int(df.loc[df.split == "test", "text"].isin(tr).sum())
    print({k: info[k] for k in ("n_train", "n_val", "n_test", "test_sentences_seen_in_train")})

    # Tokenizer: fit on training sentences only
    tok = Tokenizer.train(args.tokenizer, df.loc[df.split == "train", "text"].tolist(), out, args.vocab_size)
    lens = [len(tok.encode(t)) for t in df.loc[df.split == "train", "text"]]
    info["tokenizer"] = tok.to_dict() | {"vocab_size": tok.vocab_size,
                                         "tokens_per_sentence_p95": int(np.percentile(lens, 95))}
    print(f"Tokenizer: {tok.kind}, vocab {tok.vocab_size}, p95 length {info['tokenizer']['tokens_per_sentence_p95']}")

    # Feature normalization stats from (a sample of) the training set
    train = df[df.split == "train"]
    sample = train.sample(min(len(train), 2000), random_state=args.seed)
    s1 = s2 = None; n = 0
    for p in sample.npz:
        x = load_features(p).astype(np.float64)
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = (x ** 2).sum(0) if s2 is None else s2 + (x ** 2).sum(0)
        n += len(x)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    std = np.where(std < 1e-3, 1.0, std)
    mean[-config.N_FLAGS:], std[-config.N_FLAGS:] = 0.0, 1.0      # keep 0/1 flags as they are
    np.savez(out / "norm.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    (out / "split_info.json").write_text(json.dumps(info, indent=2))
    print(f"Wrote splits, tokenizer, norm stats to {out}")


if __name__ == "__main__":
    main()