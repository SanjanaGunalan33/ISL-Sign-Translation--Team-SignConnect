"""Word-level or SentencePiece (subword) tokenizer with fixed special-token ids:
<pad>=0, <bos>=1, <eos>=2, <unk>=3."""
import json
from collections import Counter
from pathlib import Path

from src import config


class Tokenizer:
    def __init__(self, kind, path):
        self.kind, self.path = kind, Path(path)
        if kind == "spm":
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=str(self.path))
            self.vocab_size = self.sp.get_piece_size()
        else:
            self.itos = json.loads(self.path.read_text())
            self.stoi = {w: i for i, w in enumerate(self.itos)}
            self.vocab_size = len(self.itos)

    # --- training -------------------------------------------------------------
    @staticmethod
    def train(kind, texts, out_dir, vocab_size=4000, min_freq=2):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if kind == "spm":
            import sentencepiece as spm
            txt = out_dir / "tok_train.txt"
            txt.write_text("\n".join(texts))
            spm.SentencePieceTrainer.train(
                input=str(txt), model_prefix=str(out_dir / "spm"), vocab_size=vocab_size,
                model_type="unigram", character_coverage=1.0, hard_vocab_limit=False,
                pad_id=config.PAD, bos_id=config.BOS, eos_id=config.EOS, unk_id=config.UNK,
                pad_piece="<pad>", bos_piece="<bos>", eos_piece="<eos>", unk_piece="<unk>",
                minloglevel=2)
            txt.unlink()
            return Tokenizer("spm", out_dir / "spm.model")
        counts = Counter(w for t in texts for w in t.split())
        itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + [w for w, c in counts.most_common() if c >= min_freq]
        path = out_dir / "vocab.json"
        path.write_text(json.dumps(itos))
        return Tokenizer("word", path)

    # --- use ------------------------------------------------------------------
    def encode(self, text, add_special=True):
        if self.kind == "spm":
            ids = self.sp.encode(text)
        else:
            ids = [self.stoi.get(w, config.UNK) for w in text.split()]
        return [config.BOS] + ids + [config.EOS] if add_special else ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (config.PAD, config.BOS, config.EOS)]
        if self.kind == "spm":
            return self.sp.decode(ids)
        return " ".join(self.itos[i] for i in ids)

    def to_dict(self):
        return {"kind": self.kind, "path": str(self.path)}
