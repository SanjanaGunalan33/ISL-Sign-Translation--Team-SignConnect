"""Step 5: train the BiLSTM-Transformer translator.

Usage:
    # sanity check: the model must be able to memorise 32 clips (loss -> ~0, BLEU -> ~100)
    python -m src.train_model --name overfit --overfit 32 --epochs 200

    # real run
    python -m src.train_model --name base

    # small config from the original methodology (64/128), as a baseline
    python -m src.train_model --name small --d-model 128 --enc-hidden 64 --dec-layers 2 --ff-dim 256 --subsample 0

    # ablation: no geometric or delta features
    python -m src.train_model --name no_geom_delta --drop geom delta
"""
import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src import config
from src.dataset import BucketBatchSampler, SignDataset, collate
from src.metrics import compute_all
from src.model import ModelConfig, SignTranslator, count_params
from src.tokenizer import Tokenizer


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_tokenizer(split_dir):
    info = json.loads((Path(split_dir) / "split_info.json").read_text())["tokenizer"]
    return Tokenizer(info["kind"], info["path"])


@torch.no_grad()
def evaluate(model, loader, tok, device, max_batches=None, crit=None):
    model.eval()
    hyps, refs, tot_loss, tot_tok = [], [], 0.0, 0
    for b, (src, lens, tgt, _) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        src, lens, tgt = src.to(device), lens.to(device), tgt.to(device)
        if crit is not None:
            logits = model(src, lens, tgt[:, :-1])
            n = tgt[:, 1:].ne(config.PAD).sum().item()
            tot_loss += crit(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)).item() * n
            tot_tok += n
        for h, r in zip(model.greedy(src, lens), tgt.tolist()):
            if config.EOS in h:
                h = h[: h.index(config.EOS)]
            hyps.append(tok.decode(h)); refs.append(tok.decode(r))
    model.train()
    return hyps, refs, (tot_loss / max(tot_tok, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--split-dir", default=str(config.SPLIT_DIR))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000, help="warmup steps")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6, help="epochs without val improvement")
    ap.add_argument("--val-batches", type=int, default=40, help="batches of val used for BLEU each epoch")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop", nargs="*", default=[], choices=["hands", "pose", "geom", "delta", "flags"])
    ap.add_argument("--overfit", type=int, default=0, help="train+validate on N training clips (pipeline check)")
    ap.add_argument("--limit", type=int, default=0, help="use only first N train clips")
    ap.add_argument("--cache", action="store_true", help="keep features in RAM (small datasets)")
    # model
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--enc-hidden", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=2)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--dec-layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff-dim", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-tgt-len", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    split = Path(args.split_dir)
    tok = load_tokenizer(split)
    norm = split / "norm.npz"
    common = dict(tokenizer=tok, norm_path=norm, drop_groups=args.drop, max_tgt_len=args.max_tgt_len)

    train_ds = SignDataset(split / "train.csv", limit=args.limit, cache=args.cache or bool(args.overfit), **common)
    if args.overfit:
        train_ds = Subset(train_ds, list(range(min(args.overfit, len(train_ds)))))
        val_ds = train_ds
        args.dropout, args.label_smoothing, args.warmup, args.patience = 0.0, 0.0, 50, 10 ** 9
        args.batch_size = min(args.batch_size, args.overfit)
    else:
        val_ds = SignDataset(split / "val.csv", cache=args.cache, **common)
    base = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    lengths = [base.lengths[i] for i in (train_ds.indices if isinstance(train_ds, Subset) else range(len(base)))]

    train_dl = DataLoader(train_ds, batch_sampler=BucketBatchSampler(lengths, args.batch_size, seed=args.seed),
                          collate_fn=collate, num_workers=0 if args.overfit or args.cache else args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=0 if args.overfit or args.cache else args.workers)

    mcfg = ModelConfig(in_dim=base.feat_dim, vocab_size=tok.vocab_size, d_model=args.d_model,
                       enc_hidden=args.enc_hidden, enc_layers=args.enc_layers, subsample=args.subsample,
                       dec_layers=args.dec_layers, n_heads=args.heads, ff_dim=args.ff_dim,
                       dropout=args.dropout, max_tgt_len=args.max_tgt_len)
    model = SignTranslator(mcfg).to(device)
    print(f"device {device} | params {count_params(model) / 1e6:.2f}M | train {len(train_ds)} | val {len(val_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    crit = nn.CrossEntropyLoss(ignore_index=config.PAD, label_smoothing=args.label_smoothing)
    eval_crit = nn.CrossEntropyLoss(ignore_index=config.PAD)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run = config.RUNS_DIR / args.name
    run.mkdir(parents=True, exist_ok=True)
    meta = dict(args=vars(args), model=mcfg.to_dict(), tokenizer=tok.to_dict(), norm=str(norm),
                drop=args.drop, versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                                              platform=platform.platform()))
    (run / "run_config.json").write_text(json.dumps(meta, indent=2))
    log = open(run / "log.csv", "w")
    log.write("epoch,train_loss,val_loss,val_bleu4,val_wer,secs\n")

    best, bad, step = -1.0, 0, 0
    for ep in range(1, args.epochs + 1):
        t0, tot, ntok = time.time(), 0.0, 0
        model.train()
        for src, lens, tgt, _ in train_dl:
            src, lens, tgt = src.to(device), lens.to(device), tgt.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(src, lens, tgt[:, :-1])          # teacher forcing
                loss = crit(logits.reshape(-1, logits.size(-1)).float(), tgt[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            n = tgt[:, 1:].ne(config.PAD).sum().item()
            tot += loss.item() * n; ntok += n

        hyps, refs, vloss = evaluate(model, val_dl, tok, device, max_batches=args.val_batches, crit=eval_crit)
        m = compute_all(hyps, refs, with_meteor=False)
        secs = time.time() - t0
        print(f"ep {ep:3d} | train {tot / ntok:.3f} | val loss {vloss:.3f} | BLEU-4 {m['BLEU-4']:.2f} "
              f"| WER {m['WER']:.1f} | {secs:.0f}s")
        print(f"   ref: {refs[0]}\n   hyp: {hyps[0]}")
        log.write(f"{ep},{tot / ntok:.4f},{vloss:.4f},{m['BLEU-4']},{m['WER']},{secs:.1f}\n"); log.flush()

        # model selection on val BLEU-4 (ties broken by lower val loss)
        score = m["BLEU-4"] - 1e-3 * vloss
        if score > best:
            best, bad = score, 0
            torch.save({"state_dict": model.state_dict(), **meta, "epoch": ep, "val": m}, run / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {ep}")
                break
    log.close()
    print(f"Best checkpoint: {run / 'best.pt'}")


if __name__ == "__main__":
    main()
