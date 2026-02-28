# AI Audio Detector

Classify speech audio as **human (natural)** or **AI-generated** using acoustic features and a machine-learning classifier. Comes with a REST API and a browser-based drag-and-drop UI.

![Eval accuracy: 99.95%](results/confusion_matrix_eval.png)

---

## How it works

1. **Feature extraction** — 48 acoustic features are pulled from each file using [librosa](https://librosa.org) and [Praat (parselmouth)](https://parselmouth.readthedocs.io/):
   - MFCCs (13 coefficients + deltas)
   - Pitch / F0 statistics (mean, std, range, voiced ratio)
   - Voice quality: jitter, shimmer, HNR — AI voices are *too perfect* (low jitter/shimmer, high HNR)
   - Spectral features: centroid, bandwidth, rolloff, flatness
   - Temporal features: RMS energy, zero-crossing rate

2. **Training** — XGBoost and RandomForest are compared with 5-fold cross-validation; the winner is saved as `models/detector.pkl`.

3. **Inference** — `pipeline.py` loads the model and predicts any audio file in under a second.

4. **API + UI** — FastAPI serves a REST API and a dark-themed single-page UI.

---

## Project structure

```
ai-audio-detector/
├── src/
│   ├── feature_extractor.py   # extract 48 acoustic features from an audio file
│   ├── dataset_builder.py     # process data/ folders → results/dataset.csv
│   ├── model.py               # train, evaluate, save model
│   ├── pipeline.py            # AudioDetector class + CLI inference
│   └── visualizer.py          # generate comparison plots
├── api/
│   └── main.py                # FastAPI app (single + batch prediction endpoints)
├── data/
│   ├── natural/               # your natural speech files (not in git)
│   ├── ai_generated/          # your AI-generated speech files (not in git)
│   ├── collect_all.py         # orchestrates all data collectors
│   ├── collect_wavefake.py    # download WaveFake dataset
│   ├── collect_asvspoof.py    # organise ASVspoof 2019 dataset
│   ├── collect_elevenlabs.py  # generate samples via ElevenLabs API
│   └── organize_asvspoof.py   # sort ASVspoof files by protocol label
├── models/
│   └── detector.pkl           # trained model bundle (not in git)
├── results/
│   ├── confusion_matrix_dev.png
│   ├── confusion_matrix_eval.png
│   └── feature_importance.png
└── static/
    └── index.html             # browser UI
```

---

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/<you>/ai-audio-detector.git
cd ai-audio-detector

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install librosa praat-parselmouth numpy pandas scikit-learn xgboost \
            matplotlib seaborn tqdm fastapi "uvicorn[standard]" python-multipart
```

> **macOS only** — XGBoost needs OpenMP:
> ```bash
> brew install libomp
> ```

### 2. Verify your environment

```bash
python src/libaries_verification.py
```

---

## Bring your own data

The audio data is not included in this repository (too large). You need two balanced folders:

| Folder | Contents |
|--------|----------|
| `data/natural/` | Real human speech (.wav / .mp3 / .flac / .ogg) |
| `data/ai_generated/` | AI-generated speech (same formats) |

Aim for a **1:1 ratio** between the two classes. A few thousand files per class is enough for a good model; the more you have the better.

### Option A — Use the included collection scripts

```bash
# Edit data/collect_elevenlabs.py and set your ELEVENLABS_API_KEY env variable first
export ELEVENLABS_API_KEY=your_key_here

python data/collect_all.py
```

The scripts can download / organise:
- [WaveFake](https://github.com/RUB-SysSec/WaveFake) — vocoder-generated speech
- [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) — after you manually download the dataset, run `python data/organize_asvspoof.py` to sort files by the protocol labels
- ElevenLabs TTS — generates samples via the API

### Option B — Drop in your own files

Just copy audio files directly:

```bash
# natural speech — any source (LibriSpeech, VCTK, your own recordings, etc.)
cp /your/natural/files/*.wav  data/natural/

# AI-generated speech — any TTS system
cp /your/ai/files/*.wav  data/ai_generated/
```

Check the counts and balance if needed:

```bash
ls data/natural/ | wc -l
ls data/ai_generated/ | wc -l
```

---

## Train your own model

### Step 1 — Build the feature dataset

Processes every file in `data/natural/` and `data/ai_generated/`, extracts 48 features per file, and saves a CSV. Takes a while the first time (roughly 1–3 seconds per file).

```bash
python src/dataset_builder.py
# → results/dataset.csv
```

Print statistics on an existing dataset:

```bash
python src/dataset_builder.py summary
```

### Step 2 — Train

Splits the dataset 70 / 15 / 15 (train / dev / eval), runs 5-fold cross-validation, picks the best model (XGBoost vs RandomForest), and saves it.

```bash
python src/model.py
# → models/detector.pkl
# → results/confusion_matrix_dev.png
# → results/confusion_matrix_eval.png
# → results/feature_importance.png
```

Sample output:

```
── Cross-Validation on Train split (5-fold) ───────
  XGBoost              AUC: 0.999 ± 0.001
  RandomForest         AUC: 0.997 ± 0.001

  Best model: XGBoost (CV AUC: 0.999)

── Eval Set Results (held-out) ────────────────────
              precision    recall  f1-score
    Natural       1.00      1.00      1.00
         AI       1.00      1.00      1.00

  ROC-AUC : 0.9999
```

---

## Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser for the drag-and-drop UI.

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Model status + uptime |
| `GET` | `/model/info` | Feature list and metadata |
| `POST` | `/predict` | Single file — returns result immediately |
| `POST` | `/predict/batch` | Multiple files — returns a job ID |
| `GET` | `/jobs/{job_id}` | Poll batch job status |

### Single file (curl)

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@sample.wav"
```

Response:

```json
{
  "file": "sample.wav",
  "label": "natural",
  "confidence": 0.9821,
  "probabilities": { "natural": 0.9821, "ai_generated": 0.0179 },
  "features_used": 48,
  "inference_ms": 312.4
}
```

### Batch (curl)

```bash
curl -X POST http://localhost:8000/predict/batch \
     -F "files=@a.wav" -F "files=@b.mp3"
# Returns {"job_id": "...", "status": "pending", ...}

# Poll until done
curl http://localhost:8000/jobs/<job_id>
```

---

## Command-line inference

No server needed — run predictions directly:

```bash
# Single file
python src/pipeline.py sample.wav

# Single file, JSON output
python src/pipeline.py sample.wav --json

# All files in a folder
python src/pipeline.py data/test/ --batch

# Custom model path
python src/pipeline.py sample.wav --model models/detector.pkl
```

---

## Inspect features on a single file

```bash
python src/feature_extractor.py sample.wav
```

Prints a human-readable report of all 48 extracted features, including the key AI indicators (jitter, shimmer, HNR, F0 std).

---

## Supported formats

`.wav` · `.mp3` · `.flac` · `.ogg` · `.m4a` — max 50 MB per file via the API.

---

## Key AI detection signals

| Feature | Natural | AI-generated |
|---------|---------|--------------|
| Jitter (local) | Higher — natural micro-variation | Very low — too perfect |
| Shimmer (local) | Higher | Very low |
| HNR | Moderate | High — unnaturally clean |
| F0 std dev | Higher — expressive pitch variation | Low — monotone |
| MFCC delta | More variable | Smoother transitions |

---

## License

MIT
