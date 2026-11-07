"""Step 7: translate a new ISL video into English text.

Usage:
    python -m src.translate_video --ckpt runs/base/best.pt --video some_clip.mp4
    python -m src.translate_video --ckpt runs/base/best.pt --video some_clip.mp4 --beam 5 --speak
"""
import argparse

import numpy as np
import torch

from src import config
from src.features import ablation_mask, build_features
from src.model import ModelConfig, SignTranslator
from src.tokenizer import Tokenizer


class Translator:
    """Loads a checkpoint together with the exact tokenizer, normalization and
    feature selection it was trained with."""

    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = SignTranslator(ModelConfig(**ck["model"])).to(self.device).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.tok = Tokenizer(ck["tokenizer"]["kind"], config.resolve(ck["tokenizer"]["path"]))
        norm = np.load(config.resolve(ck["norm"]))
        self.mask = ablation_mask(ck.get("drop", []))
        self.mean, self.std = norm["mean"][self.mask], norm["std"][self.mask]
        self._extractor = None

    def translate_landmarks(self, left, right, pose, beam=1):
        x = build_features(left, right, pose)[:, self.mask]
        x = ((x - self.mean) / self.std).astype(np.float32)[: config.MAX_FRAMES]
        src = torch.from_numpy(x)[None].to(self.device)
        lens = torch.tensor([x.shape[0]], device=self.device)
        ids = self.model.beam(src, lens, beam_size=beam)[0] if beam > 1 else self.model.greedy(src, lens)[0]
        if config.EOS in ids:
            ids = ids[: ids.index(config.EOS)]
        return self.tok.decode(ids)

    def translate_video(self, video_path, beam=1):
        if self._extractor is None:
            from src.landmarks import LandmarkExtractor
            self._extractor = LandmarkExtractor()
        d = self._extractor.extract(video_path)
        if d["pose"].shape[0] == 0:
            raise ValueError("No frames could be read from the video.")
        return self.translate_landmarks(d["left"], d["right"], d["pose"], beam=beam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--speak", action="store_true")
    args = ap.parse_args()
    text = Translator(args.ckpt).translate_video(args.video, beam=args.beam)
    print(text)
    if args.speak:
        from src.speak_translation import speak
        speak(text)


if __name__ == "__main__":
    main()
