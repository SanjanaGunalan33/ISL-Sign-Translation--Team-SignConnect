"""PyTorch dataset, padding collate and length-bucketed batching."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src import config
from src.features import ablation_mask, load_features


class SignDataset(Dataset):
    def __init__(self, csv_path, tokenizer, norm_path, drop_groups=(), max_frames=config.MAX_FRAMES,
                 max_tgt_len=64, limit=0, cache=False):
        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        norm = np.load(norm_path)
        self.mask = ablation_mask(list(drop_groups))
        self.mean, self.std = norm["mean"][self.mask], norm["std"][self.mask]
        self.max_frames, self.max_tgt_len = max_frames, max_tgt_len
        self.cache = {} if cache else None
        self.lengths = self.df["n_frames"].clip(upper=max_frames).tolist()

    @property
    def feat_dim(self):
        return int(self.mask.sum())

    def __len__(self):
        return len(self.df)

    def features(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        x = load_features(self.df.npz[i])[:, self.mask]
        x = ((x - self.mean) / self.std).astype(np.float32)[: self.max_frames]
        if self.cache is not None:
            self.cache[i] = x
        return x

    def __getitem__(self, i):
        x = self.features(i)
        y = self.tok.encode(self.df.text[i])
        if len(y) > self.max_tgt_len:
            y = y[: self.max_tgt_len - 1] + [config.EOS]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), i


def collate(batch):
    xs, ys, idx = zip(*batch)
    B = len(xs)
    T = max(x.shape[0] for x in xs)
    L = max(y.shape[0] for y in ys)
    src = torch.zeros(B, T, xs[0].shape[1])
    tgt = torch.full((B, L), config.PAD, dtype=torch.long)
    src_lens = torch.tensor([x.shape[0] for x in xs])
    for b, (x, y) in enumerate(zip(xs, ys)):
        src[b, : x.shape[0]] = x
        tgt[b, : y.shape[0]] = y
    return src, src_lens, tgt, torch.tensor(idx)


class BucketBatchSampler(Sampler):
    """Groups clips of similar length so batches waste little padding."""

    def __init__(self, lengths, batch_size, shuffle=True, seed=0, bucket_mult=50):
        self.lengths, self.bs, self.shuffle = np.asarray(lengths), batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.chunk = batch_size * bucket_mult

    def __iter__(self):
        idx = np.arange(len(self.lengths))
        if self.shuffle:
            self.rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.chunk):
            c = idx[s: s + self.chunk]
            c = c[np.argsort(self.lengths[c], kind="stable")]
            batches += [c[i: i + self.bs].tolist() for i in range(0, len(c), self.bs)]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs
