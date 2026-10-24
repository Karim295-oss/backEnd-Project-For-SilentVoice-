import os
import cv2
import numpy as np

# ── MediaPipe ─────────────────────────────────────────────────────────────────
import mediapipe.solutions.holistic as mp_holistic
import mediapipe.solutions.drawing_utils as mp_drawing

# ── Keras / TensorFlow ────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tf logs
from tensorflow.keras.models import load_model

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "action_advanced.keras")
MODEL_PATH_H5 = os.path.join(BASE_DIR, "models", "transformer_wlasl.h5")
ACTIONS_TXT   = os.path.join(BASE_DIR, "models", "actions_transformer.txt")
SEQUENCE_LEN  = 30

# ── Helpers (same as training) ────────────────────────────────────────────────
def extract_keypoints(results):
    # 1. Determine origin (nose)
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        cx, cy, cz = nose.x, nose.y, nose.z
        # 2. No Artificial Scaling
        # We previously tried scaling, but it warped the vertical geometry (pulling hands mathematically closer to the face).
        scale = 1.0
        
        pose = np.array([[(r.x - cx)*scale, (r.y - cy)*scale, (r.z - cz)*scale, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
    else:
        cx, cy, cz = 0.5, 0.5, 0.0
        scale = 1.0
        pose = np.zeros(33 * 4)

    if results.left_hand_landmarks:
        lh = np.array([[(r.x - cx)*scale, (r.y - cy)*scale, (r.z - cz)*scale] for r in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[(r.x - cx)*scale, (r.y - cy)*scale, (r.z - cz)*scale] for r in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    # --- NEW: POSE-BASED STD NORMALIZATION ---
    # Ignore visibility scores (every 4th element starting at index 3)
    pose_xyz = np.delete(pose, np.arange(3, 132, 4))
    
    # Calculate standard deviation of non-zero body coordinates
    non_zero_coords = pose_xyz[pose_xyz != 0]
    if len(non_zero_coords) > 0:
        pose_std = np.std(non_zero_coords)
        if pose_std < 0.01: # Avoid div by zero or microscopic noise
            pose_std = 1.0
    else:
        pose_std = 1.0
        
    # Normalize everything by this spread
    pose_normalized = pose / pose_std
    lh_normalized = lh / pose_std
    rh_normalized = rh / pose_std

    return np.concatenate([pose_normalized, lh_normalized, rh_normalized])

    return np.concatenate([pose, lh, rh])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    # Draw pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    # Draw left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    # Draw right hand  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

# ── Main Real-time Loop ───────────────────────────────────────────────────────
def main():
    print("Loading classes...")
    if not os.path.exists(ACTIONS_TXT):
        print(f"Error: Could not find {ACTIONS_TXT}. Has the training finished?")
        return
        
    with open(ACTIONS_TXT, "r", encoding="utf-8") as f:
        actions = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(actions)} sign classes.")

    print("Loading model...")
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    elif os.path.exists(MODEL_PATH_H5):
        model = load_model(MODEL_PATH_H5)
    else:
        print("Error: Could not find the trained model. Make sure training completes first!")
        return

    sequence = []
    sentence = []
    threshold = 0.85
    cooldown = 0
    top_predictions = []

    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print("Webcam started. Press 'q' to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detect Landmarks
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # 2. Extract and format keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LEN:]
            
            # 3. Continuous Translation Logic
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                sentence = [] # Spacebar now clears the sentence
                
            if len(sequence) == SEQUENCE_LEN:
                # Check for movement
                seq_array = np.array(sequence)
                # Hands are at indices 132:195 (lh) and 195:258 (rh)
                lh_seq = seq_array[:, 132:195]
                rh_seq = seq_array[:, 195:258]
                
                # Calculate variance to see if hands are moving
                movement = np.sum(np.std(lh_seq, axis=0)) + np.sum(np.std(rh_seq, axis=0))
                
                if cooldown > 0:
                    cooldown -= 1
                    cv2.putText(image, "PROCESSING...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2, cv2.LINE_AA)
                elif movement > 5.0:
                    cv2.putText(image, "SIGN DETECTED!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    
                    top_3_idx = np.argsort(res)[-3:][::-1]
                    top_predictions = [f"{actions[idx]}: {res[idx]*100:.1f}%" for idx in top_3_idx]
                        
                    best_class_idx = top_3_idx[0]
                    confidence = res[best_class_idx]
                    
                    if confidence > threshold:
                        predicted_word = actions[best_class_idx]
                        if len(sentence) == 0 or predicted_word != sentence[-1]:
                            sentence.append(predicted_word)
                            if len(sentence) > 7: 
                                sentence = sentence[-7:]
                            # Add a 1.5s cooldown (approx 45 frames) so it doesn't spam the same word
                            cooldown = 45 
                else:
                    cv2.putText(image, "IDLE (Not Moving)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    top_predictions = ["Waiting for motion..."]

            # Render Top 3 Predictions
            if top_predictions:
                for i, pred_text in enumerate(top_predictions):
                    color = (0, 255, 0) if i == 0 else (0, 200, 255)
                    cv2.putText(image, pred_text, (10, 80 + (i * 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                    
            cv2.putText(image, "Tap SPACE to Clear Sentence", (10, image.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Output the "sentence" of signs we have strung together at the bottom
            cv2.rectangle(image, (0, image.shape[0]-40), (image.shape[1], image.shape[0]), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (10, image.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('ASL Real-Time Translation', image)
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
