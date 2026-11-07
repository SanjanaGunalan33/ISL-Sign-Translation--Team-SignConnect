"""BiLSTM encoder + Transformer decoder for sign-to-text translation."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src import config


@dataclass
class ModelConfig:
    in_dim: int = config.FEATURE_DIM
    vocab_size: int = 4000
    d_model: int = 256
    enc_hidden: int = 256        # per direction
    enc_layers: int = 2
    subsample: int = 1           # number of stride-2 conv layers before the LSTM (0, 1 or 2)
    dec_layers: int = 3
    n_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.2
    max_tgt_len: int = 64

    def to_dict(self):
        return asdict(self)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class SignTranslator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.in_proj = nn.Sequential(nn.Linear(cfg.in_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(cfg.dropout))
        self.subsample = nn.ModuleList(
            [nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1) for _ in range(cfg.subsample)])
        self.lstm = nn.LSTM(d, cfg.enc_hidden, num_layers=cfg.enc_layers, batch_first=True,
                            bidirectional=True, dropout=cfg.dropout if cfg.enc_layers > 1 else 0.0)
        self.enc_out = nn.Sequential(nn.Linear(2 * cfg.enc_hidden, d), nn.LayerNorm(d))

        self.embed = nn.Embedding(cfg.vocab_size, d, padding_idx=config.PAD)
        self.pos = PositionalEncoding(d, max_len=cfg.max_tgt_len + 8)
        layer = nn.TransformerDecoderLayer(d, cfg.n_heads, cfg.ff_dim, cfg.dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, cfg.dec_layers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, cfg.vocab_size)
        self.out.weight = self.embed.weight          # weight tying
        nn.init.normal_(self.embed.weight, std=d ** -0.5)   # keeps tied output logits small at init
        with torch.no_grad():
            self.embed.weight[config.PAD].zero_()
        self.drop = nn.Dropout(cfg.dropout)

    # ---- encoder ---------------------------------------------------------------
    def encode(self, src, src_lens):
        """src (B,T,F) -> memory (B,T',d), memory padding mask (B,T') True=pad."""
        x = self.in_proj(src)
        lens = src_lens.clone()
        for conv in self.subsample:
            x = torch.relu(conv(x.transpose(1, 2))).transpose(1, 2)
            lens = (lens - 1) // 2 + 1
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        mem = self.enc_out(h)
        mask = torch.arange(mem.size(1), device=mem.device)[None, :] >= lens.to(mem.device)[:, None]
        return mem, mask

    # ---- decoder ---------------------------------------------------------------
    def decode(self, tgt_in, mem, mem_mask):
        L = tgt_in.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt_in.device), 1)
        y = self.drop(self.pos(self.embed(tgt_in) * math.sqrt(self.cfg.d_model)))
        y = self.decoder(y, mem, tgt_mask=causal, tgt_key_padding_mask=tgt_in.eq(config.PAD),
                         memory_key_padding_mask=mem_mask)
        return self.out(y)

    def forward(self, src, src_lens, tgt_in):
        mem, mask = self.encode(src, src_lens)
        return self.decode(tgt_in, mem, mask)

    # ---- inference -------------------------------------------------------------
    @torch.no_grad()
    def greedy(self, src, src_lens, max_len=None):
        max_len = max_len or self.cfg.max_tgt_len
        mem, mask = self.encode(src, src_lens)
        B = src.size(0)
        ys = torch.full((B, 1), config.BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            nxt = self.decode(ys, mem, mask)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, config.PAD)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt.eq(config.EOS)
            if done.all():
                break
        return ys[:, 1:].tolist()

    @torch.no_grad()
    def beam(self, src, src_lens, beam_size=5, max_len=None, len_penalty=0.6):
        """Beam search for ONE example (batch size 1)."""
        max_len = max_len or self.cfg.max_tgt_len
        mem, mask = self.encode(src[:1], src_lens[:1])
        beams = [([config.BOS], 0.0)]
        finished = []
        for _ in range(max_len - 1):
            seqs = torch.tensor([b[0] for b in beams], device=src.device)
            logp = self.decode(seqs, mem.expand(len(beams), -1, -1), mask.expand(len(beams), -1))[:, -1].log_softmax(-1)
            cand = []
            for k, (seq, score) in enumerate(beams):
                top = logp[k].topk(beam_size)
                for lp, t in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((seq + [t], score + lp))
            cand.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for seq, score in cand:
                if seq[-1] == config.EOS:
                    finished.append((seq, score / (len(seq) ** len_penalty)))
                else:
                    beams.append((seq, score))
                if len(beams) == beam_size:
                    break
            if not beams or len(finished) >= beam_size:
                break
        finished += [(s, sc / (len(s) ** len_penalty)) for s, sc in beams]
        best = max(finished, key=lambda f: f[1])[0]
        return [best[1:]]


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
