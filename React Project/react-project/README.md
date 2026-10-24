# SilentVoice Frontend

Vite + React interface for ASL sign-to-text (webcam) and speech-to-ASL (avatar), talking to the FastAPI backend (`backend.py`) over `/api` and `/ws`.

## Run

```powershell
npm start
```

Open http://localhost:3000. The dev server proxies `/api` and `/ws` to `http://localhost:8000`, where `backend.py` should be running.
