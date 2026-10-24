import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "models", "MP_Data_Norm")
SEQUENCE_LENGTH = 30

actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
label_map = {label: num for num, label in enumerate(actions)}

print("Loading dataset (this may take a few minutes)...", flush=True)
sequences, labels = [], []

for i, action in enumerate(actions):
    if i % 10 == 0:
        print(f"Processing action {i+1}/{len(actions)}: {action}", flush=True)
    action_path = os.path.join(DATA_PATH, action)
    for sequence in os.listdir(action_path):
        seq_dir = os.path.join(action_path, sequence)
        if os.path.isdir(seq_dir):
            window = []
            for i in range(SEQUENCE_LENGTH):
                npy = os.path.join(seq_dir, f"{i}.npy")
                if os.path.isfile(npy):
                    arr = np.load(npy)
                    pose = arr[:132]
                    hands = arr[1536:]
                    
                    pose_xyz = np.delete(pose, np.arange(3, 132, 4))
                    non_zero_coords = pose_xyz[pose_xyz != 0]
                    if len(non_zero_coords) > 0:
                        pose_std = np.std(non_zero_coords)
                        if pose_std < 0.01: 
                            pose_std = 1.0
                    else:
                        pose_std = 1.0
                        
                    pose_normalized = pose / pose_std
                    hands_normalized = hands / pose_std
                    
                    features = np.concatenate([pose_normalized, hands_normalized])
                    window.append(features)
                else:
                    window.append(np.zeros(258))
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences, dtype=np.float32)
y = keras.utils.to_categorical(labels).astype(int)

# Exact split used in training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=np.argmax(y, axis=1), random_state=42)

print("Loading model...")
model = keras.models.load_model(os.path.join(BASE_DIR, 'models', 'action_advanced.keras'))

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[FINAL] Test Accuracy: {acc*100:.2f}%\n")

y_pred = model.predict(X_test, verbose=0)
y_true_cls = np.argmax(y_test, axis=1)
y_pred_cls = np.argmax(y_pred, axis=1)

# Top 5
top5_correct = 0
for i in range(len(y_test)):
    top5 = np.argsort(y_pred[i])[-5:]
    if y_true_cls[i] in top5:
        top5_correct += 1
print(f"[FINAL] Top-5 Accuracy: {(top5_correct/len(y_test))*100:.2f}%\n")

cm = confusion_matrix(y_true_cls, y_pred_cls)

# Find top confused pairs
errors = []
for i in range(len(actions)):
    for j in range(len(actions)):
        if i != j and cm[i, j] > 0:
            errors.append((cm[i, j], actions[i], actions[j]))

errors.sort(reverse=True, key=lambda x: x[0])

print("=== TOP CONFUSIONS (True -> Predicted) ===")
for count, true_cls, pred_cls in errors[:10]:
    print(f"Confused {count} times: {true_cls} -> {pred_cls}")

