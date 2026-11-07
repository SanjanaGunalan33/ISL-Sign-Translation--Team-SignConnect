# ISL - Sign Translation (Team Sign Connect)

Translates a video of a person signing a sentence in **Indian Sign Language (ISL)** into an **English sentence**, and speaks it aloud. Trained and evaluated on the **iSign v1.1** benchmark.

**Team Sign Connect:** Sanjana Gunalan, Tanushree Skandha, Chandra Vinuthna, Kovuru Sahitya
B.Tech Artificial Intelligence and Engineering, Amrita School of Computing, Chennai
**Supervisor:** Dr. M. Pallavi

## How it works

```
ISL video -> 10 fps frames -> MediaPipe Holistic (hand + body landmarks)
          -> 323 features per frame -> Conv1D + 2-layer BiLSTM encoder
          -> 2-layer Transformer decoder -> English text -> offline speech
```

- **Features (323 per frame):** hand and body landmarks normalised to the signer's shoulders (153), engineered geometric features such as fingertip distances, elbow angles and hand orientation (15), motion features as frame-to-frame changes (153), and hand-visible flags (2).
- **Model:** linear projection, a stride-2 1-D convolution, a 2-layer bidirectional LSTM (128 units per direction) and a 2-layer Transformer decoder (4 heads) over a 1,000-token SentencePiece vocabulary. 1.31 M parameters.
- **Training:** cross-entropy with label smoothing 0.1, AdamW (learning rate 1e-3, weight decay 0.01), 200 warm-up steps then cosine decay, batches of 32, 40 epochs; the checkpoint with the best validation BLEU-4 is kept.

## Results

5,000-clip pilot of iSign (sentences of up to 6 words), split 4,000 / 500 / 500 so that **no test sentence appears in training**. Scores on the 500 unseen test clips:

| BLEU-1 | BLEU-2 | BLEU-4 | chrF | ROUGE-L | WER |
|---|---|---|---|---|---|
| 12.72 | 3.12 | 0.71 | 9.20 | 5.11 | 107.9% |

For context, the published iSign pose-to-text baselines report BLEU-4 between 0.09 and 1.47 (Joshi et al., 2024). They use the full test set, so the comparison is indicative. Training logs, metrics and all test predictions are in `runs/pilot/`; the 32-clip sanity check is in `runs/overfit/`.

## Setup

Python 3.10 to 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Data

The iSign dataset and the trained model are **not** included in this repository. Download iSign v1.1 from its authors (paper: https://arxiv.org/abs/2407.05404) and place the files as:

```
data/raw/iSign_v1.1.csv
data/raw/manifests/isign_manifest.csv
data/raw/iSign-videos_v1.1_part_aa
data/raw/iSign-videos_v1.1_part_ab
```

## Running the pipeline

Run from the project folder. Each step uses the output of the previous one, and every script has `--help`.

| Step | Command | Output |
|---|---|---|
| 1. Clean the annotations | `python -m src.inspect_dataset` | `data/clean_annotations.csv` |
| 2. Pick a 5,000-clip pilot subset of short sentences | `python -c "import pandas as pd; d=pd.read_csv('data/clean_annotations.csv'); d=d[d.text.str.split().str.len()<=6]; d.sample(5000, random_state=0).to_csv('data/clean_subset.csv', index=False)"` | `data/clean_subset.csv` |
| 3. Extract and check the videos | `python -m src.validate_videos --parts data/raw/iSign-videos_v1.1_part_aa data/raw/iSign-videos_v1.1_part_ab --clean-csv data/clean_subset.csv` | `data/videos/`, `data/video_index.csv` |
| 4. Landmarks (6 parallel workers) | `python -m src.extract_features --workers 6` | `data/features/` |
| 5. Leak-free split and tokenizer | `python -m src.prepare_splits --clean-csv data/clean_subset.csv --vocab-size 1000` | `data/splits/` |
| 6. Train | `python -m src.train_model --name pilot --epochs 40 --lr 1e-3 --warmup 200 --d-model 128 --enc-hidden 128 --dec-layers 2 --ff-dim 256` | `runs/pilot/best.pt` |
| 7. Evaluate on the test set | `python -m src.evaluate_model --ckpt runs/pilot/best.pt --split test` | metrics and predictions in `runs/pilot/` |

Step 2 draws a random sample, so it will not give exactly the same 5,000 clips as our run. The full settings of our runs are saved in `runs/pilot/run_config.json` and `runs/overfit/run_config.json`.

## Live translation

After training (step 6):

```bash
# Webcam: raise your hands to start, lower them for 1 second to translate
python -m src.live_translate --ckpt runs/pilot/best.pt

# Any video file (dataset clip or your own recording)
python -m src.live_translate --ckpt runs/pilot/best.pt --source path/to/video.mp4
```

The translated sentence is shown on screen and spoken with offline text-to-speech. Add `--no-speak` to turn speech off, or `--camera 1` to use another camera.

## Code

| File | Purpose |
|---|---|
| `src/inspect_dataset.py` | Clean the iSign manifest and annotations |
| `src/validate_videos.py` | Read the two-part video archive as one zip and extract the needed clips |
| `src/landmarks.py`, `src/extract_features.py` | Frame sampling and parallel MediaPipe landmark extraction |
| `src/features.py` | Normalisation and the 323-value feature vector |
| `src/prepare_splits.py`, `src/tokenizer.py` | Leak-free split, SentencePiece tokenizer, feature statistics |
| `src/dataset.py`, `src/model.py`, `src/train_model.py` | Data loading, BiLSTM-Transformer model, training |
| `src/evaluate_model.py`, `src/metrics.py` | Test-set translation and BLEU, chrF, ROUGE-L, WER |
| `src/translate_video.py`, `src/live_translate.py`, `src/speak_translation.py` | Translation of a clip, live demo, speech output |
| `src/config.py` | Paths and constants |

## Reference

Joshi, A., Mohanty, R., Kanakanti, M., Mangla, A., Choudhary, S., Barbate, M. and Modi, A., *iSign: A Benchmark for Indian Sign Language Processing*, Findings of the Association for Computational Linguistics: ACL 2024.
