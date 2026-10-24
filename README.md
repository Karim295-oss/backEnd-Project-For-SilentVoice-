# SILENTVOICE

SILENTVOICE translates ASL to text from a webcam and English speech to ASL avatar animations. The project has a Python/FastAPI backend and a separate React/Vite frontend; run both for the web app.

## Project layout

| Path | Purpose |
| --- | --- |
| `backend.py` | FastAPI backend and WebSocket inference |
| `models/` | Model training, evaluation, inference, and saved model assets |
| `models/backup(unused)/` | Archived legacy models, experiments, notebooks, and logs |
| `scripts/` | Dataset preparation and verification scripts |
| `data/` | Local datasets and generated clips; not committed |
| `React Project/react-project/` | React/Vite frontend |
| `asl/tgcn-wlasl/` | Separate TGCN/WLASL experiment checkout |
| `meet_transcriber/` | Standalone Windows Playwright tool for best-effort Meet live-caption transcription |

For architecture, data-flow details, and known setup caveats, see [CLAUDE.md](./CLAUDE.md).

The optional Google Meet caption transcriber has separate setup, usage, limitations, and a private-meeting test plan in [meet_transcriber/README.md](./meet_transcriber/README.md). Its selectors depend on Meet's unsupported UI/DOM and are not stable guarantees.

## Run the app

Start the backend from the repository root in an environment with the backend dependencies installed:

```powershell
python -m pip install -r requirements.txt
python backend.py
```

In another terminal, start the frontend:

```powershell
cd "React Project/react-project"
npm install
npm start
```

Open <http://localhost:3000>. The Vite server proxies API and WebSocket requests to the backend on port 8000.

On the webcam tab, use **Recognition model** to switch between the current 50-sign BiLSTM and the ASL-100 TGCN. To enable the optional TGCN model, install its dependencies into the same Python environment used to run the backend:

```powershell
python -m pip install -r requirements-tgcn.txt
```

The ASL-100 model classifies isolated signs from a 100-word WLASL vocabulary; it does not translate continuous signing into sentences. The checkpoint was trained with OpenPose coordinates. The live adapter maps corresponding MediaPipe pose/hand landmarks into its 55-point layout, center-crops the webcam view, and samples 50 frames per prediction; because the live detector differs from the training detector, accuracy may differ from the checkpoint's benchmark.

## Expand the speech-to-sign vocabulary

To generate avatar sequences for as many available single-word glosses as possible, run this once from the repository root with the backend Python environment activated:

```powershell
python scripts/build_speech_sign_library.py --workers 2
```

The script uses clips under `data/dataset/SL/` and any matching WLASL clips under `data/Wlasl/`, normalizes MediaPipe landmarks using the same feature extraction as live inference, and saves one 30-frame sequence per usable gloss under `data/speech_sign_library/`. It prioritizes the ASL-100 labels, then processes the rest of the available vocabulary. It does not train a model or change sign-to-text labels/model. Existing generated words are reused on later runs; use `--overwrite` to rebuild them. Use `--dry-run` or `--max-glosses 1` to preview/test a small selection. Extraction uses two worker processes by default; adjust with `--workers N`, or set `--workers 1` if memory is limited. The current speech UI sends one word at a time, so multi-word gloss labels are excluded. Restart the backend after generating the library.

## MS-ASL clip preparation

Put the MS-ASL split annotation JSON files in `data/msasl/annotations/`. Install `yt-dlp` and FFmpeg, then preview the planned downloads or start downloading:

```powershell
python scripts/download_msasl.py --dry-run
python scripts/download_msasl.py
```

Clips are written under `data/msasl/<split>/<gloss>/`. The downloader accepts `--splits`, `--limit`, `--workers`, and YouTube cookie options; run it with `--help` for details.

## Frontend checks

Run from `React Project/react-project/`:

```powershell
npm test
npm run build
```
