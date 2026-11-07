"""Step 6: evaluate a checkpoint on the held-out test set.

Writes runs/<name>/test_metrics.json and test_predictions.csv (uid, reference,
prediction, sentence-level WER) so you can read good and bad examples.

Usage:
    python -m src.evaluate_model --ckpt runs/base/best.pt
    python -m src.evaluate_model --ckpt runs/base/best.pt --beam 5 --split val
"""
import argparse
import json
from pathlib import Path

import jiwer
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dataset import SignDataset, collate
from src.metrics import compute_all
from src.translate_video import Translator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-dir", default=str(config.SPLIT_DIR))
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tr = Translator(args.ckpt)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ds = SignDataset(Path(args.split_dir) / f"{args.split}.csv", tr.tok, config.resolve(ck["norm"]),
                     drop_groups=ck.get("drop", []), max_tgt_len=ck["model"]["max_tgt_len"], limit=args.limit)
    dl = DataLoader(ds, batch_size=1 if args.beam > 1 else args.batch_size, shuffle=False, collate_fn=collate)

    hyps, refs, uids = [], [], []
    with torch.no_grad():
        for src, lens, _, idx in tqdm(dl, desc=f"decoding {args.split}"):
            src, lens = src.to(tr.device), lens.to(tr.device)
            outs = tr.model.beam(src, lens, args.beam) if args.beam > 1 else tr.model.greedy(src, lens)
            for i, h in zip(idx.tolist(), outs):
                if config.EOS in h:
                    h = h[: h.index(config.EOS)]
                hyps.append(tr.tok.decode(h))
                refs.append(ds.df.text[i])          # score against the original sentence, not the tokenized one
                uids.append(ds.df.uid[i])

    metrics = compute_all(hyps, refs)
    metrics.update(n=len(hyps), split=args.split, beam=args.beam, ckpt=str(args.ckpt))
    run = Path(args.ckpt).parent
    tag = f"{args.split}_beam{args.beam}"
    (run / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    pred = pd.DataFrame(dict(uid=uids, reference=refs, prediction=hyps))
    pred["wer"] = [jiwer.wer(r, h if h.strip() else "<empty>") for r, h in zip(refs, hyps)]
    pred.to_csv(run / f"{tag}_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nBest examples:")
    print(pred.nsmallest(5, "wer")[["reference", "prediction"]].to_string(index=False))
    print("\nWorst examples:")
    print(pred.nlargest(5, "wer")[["reference", "prediction"]].to_string(index=False))
    print(f"\nSaved {run / (tag + '_metrics.json')} and predictions csv")


if __name__ == "__main__":
    main()
