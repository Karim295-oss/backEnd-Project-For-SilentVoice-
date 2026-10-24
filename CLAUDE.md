# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SILENTVOICE is a two-way ASL translator with two halves that run as separate processes:

- **Python ML backend** — `backend.py` (FastAPI, port 8000) plus the training/inference scripts in `models/` and the data pipeline in `scripts/`.
- **React + Vite frontend** — `React Project/react-project/` (port 3000). Its dev server proxies `/api` (HTTP) and `/ws` (WebSocket) to the backend (see `vite.config.js`). The frontend never talks to the backend except through that proxy.

Both must be running together for the app to work.

## Commands

Frontend (`cd "React Project/react-project"` first):
- `npm start` — Vite dev server on :3000 (proxies to backend on :8000)
- `npm run build` — production build to `dist/`
- `npm test` — Vitest run (jsdom)
- `npx vitest run src/App.test.jsx` — single test file; `npx vitest -t "name"` filters by test name

Backend (from repo root, with `venv` activated):
- `python -m pip install -r requirements.txt` — install the backend and model-training dependencies
- `python backend.py` — uvicorn on :8000 (loads the Keras model + spaCy on startup; first run is slow)
- `python models/train_advanced.py` — train the BiLSTM, writes `models/action_advanced.keras`
- `python models/evaluate.py` — confusion matrix / classification report on `MP_Data_Norm`
- `python models/realtime_advanced.py` / `word_by_word.py` — standalone OpenCV desktop inference (pre-web; no server needed)

The root `requirements.txt` lists the backend and active training/evaluation dependencies. PyTorch for the optional TGCN model is separate in `requirements-tgcn.txt`. The legacy `/api/listen_speech` endpoint also needs PyAudio for microphone access.

## The two features (and how data flows)

**Sign → Text (webcam)** — `ASLTranslator.jsx` ↔ `backend.py` `/ws`:
The frontend opens a WebSocket, and while recording (toggled with **SPACE**) sends base64 JPEG frames at ~30fps. The frontend model selector chooses between the current 50-sign BiLSTM and the optional ASL-100 TGCN. The backend runs each frame through MediaPipe Holistic. The BiLSTM path keeps its 30-frame, 258-feature inference and existing confidence/motion filters. The TGCN path center-crops a square frame, maps MediaPipe pose and hand landmarks to the TGCN's 55-point layout, normalizes coordinates to `[-1, 1]`, samples 50 frames, and runs `asl/tgcn-wlasl/checkpoints/asl100/pytorch_model.bin`. This checkpoint was trained with OpenPose, so live MediaPipe predictions may differ from its benchmark. Install `requirements-tgcn.txt` in the backend environment to enable PyTorch inference. ASL-100 predicts isolated signs in its WLASL vocabulary, not continuous sentences.

**Speech → Sign (avatar)** — `SpeechToASL.jsx` ↔ `backend.py` `/api/speech_to_asl`:
Live transcription uses the **browser Web Speech API** (`webkitSpeechRecognition`) — interim results stream into the English box as you talk; this is **Chrome/Edge only**. Each *finalized* phrase is POSTed to `/api/speech_to_asl`, which translates English → ASL gloss (`grammar_translator.py`), resolves each word to a precomputed landmark sequence (`data/speech_sign_library/`, falling back to `models/MP_Data_Norm/`; unknown words get a synonym or are fingerspelled, see below), and returns skeleton frames the frontend animates on a canvas. Only words with nothing signable (no sign, synonym or letters) are left out. (`/api/listen_speech` is a legacy server-side blocking STT endpoint kept as a non-Chrome fallback; the live UI no longer uses it.)

Run `python scripts/build_speech_sign_library.py --workers 2` to create `data/speech_sign_library/` from local `data/dataset/SL/` and WLASL videos. It stores one 30-frame, 258-feature normalized sequence per usable single-word gloss, reusing the same MediaPipe feature extractor as live inference. Independent glosses are extracted in parallel, with one MediaPipe instance per worker; use `--workers 1` to reduce memory use. Speech-to-sign prefers these generated sequences and then falls back to `models/MP_Data_Norm/`. This does not train or modify the sign-to-text model or its label list; ASL-100 is used as a vocabulary priority, not as an animation generator.

## Key conventions to know before editing

- **Feature vector is normalized for scale invariance.** `extract_keypoints` centers every landmark on the nose and divides by the pose's standard deviation. This is the core trick that lets a model trained on WLASL videos work on arbitrary webcams — do not change the normalization on one side (training/`extract_landmarks.py`) without the other (`backend.py`/inference scripts), or the model silently degrades. `SEQUENCE_LEN = 30` everywhere.
- **Two different feature widths.** Training/storage uses the full 1662-dim MediaPipe vector per frame; the avatar path slices it to **258 dims** (pose 132 + left hand 63 + right hand 63) — see `load_word_sequence` in `backend.py` and the `POSE_CONNECTIONS`/`HAND_CONNECTIONS` slicing in `SpeechToASL.jsx`. Keep those offsets in sync.
- **The grammar translator is rule-based, not a model.** `grammar_translator.py` uses spaCy POS/dependency tags to reorder English into ASL gloss order `[TIME][OBJ][SUBJ][ADJ][VERB][OTHER][NOT][WH]` and drop copulas/articles. Sentences are split into clauses (root, ccomp, advcl, parataxis, verb conjuncts) that are ordered individually but kept in spoken order; fillers/hedges (um, basically, you know, kind of…) and tense auxiliaries (will/would/shall) are dropped, ability/obligation modals are kept, and pronouns stay as spoken. `english_to_asl_words` (lowercase list, used by the backend for sequence lookup) and `english_to_asl` (uppercase gloss string) must stay consistent. Needs the `en_core_web_sm` model (auto-downloads on first import).
- **Word vocabulary is the intersection of two files.** A word is only signable if it appears in `actions_transformer.txt` *and* has a folder under `models/MP_Data_Norm/`. Unknown words go through `models/sign_resolver.py`: manual synonym → WordNet synonym (nltk, downloaded on first use) → fingerspelling from the letter signs (10 frames per letter; missing letters skipped; so a fingerspelled sequence is not 30 frames, and the frontend just iterates `seq.length`). Names WordNet lists capitalized (john, mary) skip the synonym step. Speech → Sign and the Meet bridge share it through `cached_resolver`, which remembers results but not ones reached while WordNet was unavailable (they are retried). `resolve_words` returns the spoken glosses that resolved plus their sequences, keyed by spoken gloss.

## ML data pipeline

`scripts/download_videos.py` (WLASL clips) → `scripts/extract_landmarks.py` (MediaPipe → per-frame `.npy`) → `models/MP_Data_Norm/<word>/<seq>/<frame>.npy` → `train_advanced.py` → `action_advanced.keras` + `actions_transformer.txt`. The active model, vocabulary, and landmark data stay in `models/`; superseded experiments, notebooks, and generated logs are kept in `models/backup(unused)/`.

The optional MS-ASL downloader reads split annotations from `data/msasl/annotations/` and writes trimmed clips under `data/msasl/<split>/`. Run it with `python scripts/download_msasl.py`; use `--dry-run` to preview planned clips.

## Gotchas

- **Hardcoded absolute paths in `scripts/`.** The active model code (`backend.py`, `models/*.py`) now resolves `BASE_DIR` relative to the file, so it is machine-independent. The data-prep scripts in `scripts/` (`download_videos.py`, `extract_landmarks.py`, `verify_data.py`) still hardcode `E:\Projects\asl_recognition\data\...` (note: the parent of this repo, a separate dataset root) — adjust those for your machine before re-running the data pipeline. Archived legacy scripts in `models/backup(unused)/Backup/` also hardcode paths.
- `backend.py` loads the model and spaCy at import time, so startup failures (missing `action_advanced.keras`, missing spaCy model) surface as a crash on launch, not per-request.

## Agent skills

### Issue tracker

Issues and specs are tracked in this repo's GitHub Issues; use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default labels `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo. See `docs/agents/domain.md`.
