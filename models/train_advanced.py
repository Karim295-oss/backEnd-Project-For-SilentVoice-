"""
ASL Recognition — Advanced Scale-Invariant BiLSTM (50 classes)
Uses Pose-based Standard Deviation Normalization to completely eliminate 
Domain Shift between the WLASL dataset and user webcams.
"""
import os, sys, numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "models", "MP_Data_Norm")
SEQUENCE_LENGTH = 30

# Load actions
actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
label_map = {label: num for num, label in enumerate(actions)}

print("Loading dataset...")
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for sequence in os.listdir(action_path):
        seq_dir = os.path.join(action_path, sequence)
        if os.path.isdir(seq_dir):
            window = []
            for i in range(SEQUENCE_LENGTH):
                npy = os.path.join(seq_dir, f"{i}.npy")
                if os.path.isfile(npy):
                    arr = np.load(npy)
                    # arr has 1662 features:
                    # pose (0:132), face (132:1536), lh (1536:1599), rh (1599:1662)
                    pose = arr[:132]
                    hands = arr[1536:]
                    
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
                    hands_normalized = hands / pose_std
                    
                    # Reconstruct the 258-feature vector
                    features = np.concatenate([pose_normalized, hands_normalized])
                    window.append(features)
                else:
                    window.append(np.zeros(258))
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences, dtype=np.float32)
y = to_categorical(labels).astype(int)

print(f"Dataset Shape: {X.shape}, Labels Shape: {y.shape}")

# Stratified split to ensure all classes are represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=np.argmax(y, axis=1), random_state=42)

# Define Model (Enhanced BiLSTM)
input_shape = (SEQUENCE_LENGTH, 258)
model = keras.Sequential([
    layers.Input(shape=input_shape),
    # Batch Normalization normalizes the feature distribution along the temporal axis
    layers.BatchNormalization(),
    
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Dropout(0.3),
    layers.Bidirectional(layers.LSTM(64)),
    layers.BatchNormalization(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(actions), activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])

# Callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=30, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

# TF Augmentation Layer (Rotation, Scaling, Noise)
@tf.function
def augment(x, y):
    # Light coordinate jitter
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.01, dtype=tf.float32)
    x_aug = x + noise

    # Random scaling (adds robustness even on top of our manual std normalizer)
    scale = tf.random.uniform([], minval=0.85, maxval=1.15, dtype=tf.float32)
    x_aug = x_aug * scale

    # Frame dropout: blank ~10% of timesteps. WLASL clips are clean, but a real
    # webcam loses hand tracking for a few frames constantly — train for it.
    keep = tf.cast(tf.random.uniform(tf.shape(x_aug)[:-1]) > 0.10, tf.float32)
    x_aug = x_aug * tf.expand_dims(keep, -1)

    return x_aug, y

# Apply augmentation to training data only
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1024).batch(32).map(augment).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

print("Starting Advanced Training...")
history = model.fit(
    train_ds,
    epochs=150,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate and save
loss, acc = model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {acc*100:.2f}%")

model.save(os.path.join(BASE_DIR, 'models', 'action_advanced.keras'))
print("Saved advanced model to models/action_advanced.keras")
