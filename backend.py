import mediapipe_patch  # must be first – restores mediapipe.solutions API (removed in 0.10.30+)
import os
import sys
import cv2
import numpy as np
import base64
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mediapipe.solutions.holistic as mp_holistic

# ── Keras / TensorFlow ────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "action_advanced.keras")
ACTIONS_TXT = os.path.join(BASE_DIR, "models", "actions_transformer.txt")

# Load actions
with open(ACTIONS_TXT, "r", encoding="utf-8") as f:
    actions = [line.strip() for line in f if line.strip()]

# Load model
model = load_model(MODEL_PATH)
print("Model loaded successfully for Backend!")

# ── NLP Grammar Translation ───────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from landmark_features import extract_keypoints

try:
    from grammar_translator import english_to_asl_words
except Exception as e:
    print(f"[Warning] grammar_translator unavailable: {e}. Grammar will fallback to split().")
    def english_to_asl_words(sentence: str):
        return sentence.lower().split()

from sign_motions import SEQUENCE_LEN, load_sign_motion
from sign_resolver import cached_resolver, resolve_words
from tgcn_asl100 import extract_tgcn_keypoints, load_asl100_model, predict_asl100, prepare_tgcn_input

resolve_word = cached_resolver(load_sign_motion)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SpeechRequest(BaseModel):
    text: str

@app.post("/api/speech_to_asl")
def api_speech_to_asl(req: SpeechRequest):
    english_text = req.text
    asl_words = english_to_asl_words(english_text)
    
    valid_words, sequences = resolve_words(asl_words, resolve_word)

    return {
        "english": english_text,
        "asl_words": valid_words,
        "sequences": sequences
    }

@app.get("/api/listen_speech")
def api_listen_speech():
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Listen for up to 5 seconds
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            
            asl_words = english_to_asl_words(text)
            valid_words, sequences = resolve_words(asl_words, resolve_word)

            return {
                "english": text,
                "asl_words": valid_words,
                "sequences": sequences
            }
    except ImportError:
        return {"error": "SpeechRecognition not installed on backend."}
    except sr.WaitTimeoutError:
        return {"error": "Microphone timed out (no speech detected)."}
    except sr.UnknownValueError:
        return {"error": "Could not understand audio."}
    except Exception as e:
        return {"error": f"Microphone error: {str(e)}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    sequence = []
    recording = False
    active_model = "current"
    threshold = 0.85       # accept a word only when the model is confident
    margin = 0.15          # ...and clearly ahead of the runner-up
    motion_min = 5.0       # ...and the hands actually moved (reject still frames)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                data_str = await websocket.receive_text()
                msg = json.loads(data_str)
                
                action = msg.get("action")
                
                if action == "select_model":
                    selected_model = msg.get("model")
                    if recording:
                        await websocket.send_json({
                            "type": "model_error",
                            "message": "Stop recording before switching models.",
                        })
                    elif selected_model not in {"current", "asl100"}:
                        await websocket.send_json({
                            "type": "model_error",
                            "message": "Unknown recognition model.",
                        })
                    else:
                        if selected_model == "asl100":
                            await websocket.send_json({
                                "type": "status",
                                "message": "Loading ASL-100 model...",
                            })
                            try:
                                await asyncio.to_thread(load_asl100_model)
                            except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
                                print(f"Unable to load ASL-100 model: {exc}")
                                await websocket.send_json({
                                    "type": "model_error",
                                    "message": f"ASL-100 model could not be loaded: {exc}",
                                })
                                continue
                        active_model = selected_model
                        sequence = []
                        await websocket.send_json({
                            "type": "model_selected",
                            "model": active_model,
                            "message": "ASL-100 (100 signs)" if active_model == "asl100" else "Current model (50 signs)",
                        })

                elif action == "start_record":
                    recording = True
                    sequence = []
                    await websocket.send_json({"type": "status", "message": "Recording started"})
                    
                elif action == "frame" and recording:
                    frame_data = msg.get("data", "")
                    encoded_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
                    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        if active_model == "asl100":
                            height, width = frame.shape[:2]
                            side = min(height, width)
                            top = (height - side) // 2
                            left = (width - side) // 2
                            frame = frame[top:top + side, left:left + side]

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = holistic.process(image)
                        
                        keypoints = (
                            extract_tgcn_keypoints(results)
                            if active_model == "asl100"
                            else extract_keypoints(results)
                        )
                        sequence.append(keypoints)
                        
                elif action == "stop_record":
                    recording = False
                    if len(sequence) > 0:
                        if active_model == "asl100":
                            tgcn_model, tgcn_actions = load_asl100_model()
                            tgcn_sequence = np.asarray(sequence, dtype=np.float32)
                            model_input = prepare_tgcn_input(tgcn_sequence)
                            res = predict_asl100(tgcn_model, model_input)
                            actions_for_model = tgcn_actions
                            top_3_idx = np.argsort(res)[-3:][::-1]
                            best_class_idx = top_3_idx[0]
                            confidence = res[best_class_idx]
                            runner_up = res[top_3_idx[1]]
                            motion = float(np.std(tgcn_sequence[:, 13:, :], axis=0).sum())
                            accepted = confidence > 0.5 and (confidence - runner_up) > 0.05
                        else:
                            indices = np.linspace(0, len(sequence) - 1, SEQUENCE_LEN, dtype=int)
                            processed_seq = np.array([sequence[i] for i in indices])
                            res = model.predict(np.expand_dims(processed_seq, axis=0), verbose=0)[0]
                            actions_for_model = actions
                            top_3_idx = np.argsort(res)[-3:][::-1]
                            best_class_idx = top_3_idx[0]
                            confidence = res[best_class_idx]
                            runner_up = res[top_3_idx[1]] if len(top_3_idx) > 1 else 0.0
                            motion = float(np.std(processed_seq[:, 132:258], axis=0).sum())
                            accepted = confidence > threshold and (confidence - runner_up) > margin

                        top_predictions = [
                            f"{actions_for_model[idx]}: {res[idx]*100:.1f}%"
                            for idx in top_3_idx
                        ]
                        word = ""
                        if motion < (0.5 if active_model == "asl100" else motion_min):
                            top_predictions = ["No hand motion detected"]
                        elif accepted:
                            word = actions_for_model[best_class_idx]

                        await websocket.send_json({
                            "type": "prediction", 
                            "word": word, 
                            "top_predictions": top_predictions
                        })
                    else:
                        await websocket.send_json({"type": "status", "message": "No frames captured"})
                        
                    sequence = []
                    
        except WebSocketDisconnect:
            print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
