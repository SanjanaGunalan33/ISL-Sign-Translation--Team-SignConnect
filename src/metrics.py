"""Translation metrics: BLEU-1/2/4 and chrF (sacrebleu), ROUGE-L, WER (jiwer), METEOR (nltk, optional)."""
from sacrebleu.metrics import BLEU, CHRF


def rouge_l(hyp: str, ref: str) -> float:
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    dp = [0] * (len(r) + 1)
    for a in h:
        prev = 0
        for j, b in enumerate(r, 1):
            cur = dp[j]
            dp[j] = prev + 1 if a == b else max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[-1]
    p, rc = lcs / len(h), lcs / len(r)
    return 0.0 if lcs == 0 else 2 * p * rc / (p + rc)


def meteor(hyps, refs):
    try:
        from nltk.translate.meteor_score import meteor_score
        return 100 * sum(meteor_score([r.split()], h.split()) for h, r in zip(hyps, refs)) / max(len(hyps), 1)
    except Exception:          # nltk or its wordnet data not installed
        return None


def compute_all(hyps, refs, with_meteor=True):
    import jiwer
    out = {}
    for n in (1, 2, 4):
        out[f"BLEU-{n}"] = BLEU(max_ngram_order=n, effective_order=True).corpus_score(hyps, [refs]).score
    out["chrF"] = CHRF().corpus_score(hyps, [refs]).score
    out["ROUGE-L"] = 100 * sum(rouge_l(h, r) for h, r in zip(hyps, refs)) / max(len(hyps), 1)
    safe_h = [h if h.strip() else "<empty>" for h in hyps]
    out["WER"] = 100 * jiwer.wer(refs, safe_h)
    if with_meteor:
        m = meteor(hyps, refs)
        if m is not None:
            out["METEOR"] = m
    return {k: round(v, 2) for k, v in out.items()}
