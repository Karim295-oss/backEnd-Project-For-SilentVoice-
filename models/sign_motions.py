import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEECH_DATA_PATH = os.path.join(BASE_DIR, "data", "speech_sign_library")
DATA_PATH = os.path.join(BASE_DIR, "models", "MP_Data_Norm")
SEQUENCE_LEN = 30
MOTION_FEATURES = 258


def load_sign_motion(word, *, speech_dir=SPEECH_DATA_PATH, data_dir=DATA_PATH):
    """Loads the speech-only sequence first, then the original normalized dataset."""
    speech_path = os.path.join(speech_dir, f"{word.lower()}.npy")
    if os.path.isfile(speech_path):
        sequence = np.load(speech_path, allow_pickle=False)
        if sequence.shape != (SEQUENCE_LEN, MOTION_FEATURES):
            raise ValueError(
                f"Expected speech sequence {speech_path} to have shape "
                f"{(SEQUENCE_LEN, MOTION_FEATURES)}, got {sequence.shape}"
            )
        return sequence.tolist()

    word_dir = os.path.join(data_dir, word.lower())
    if not os.path.isdir(word_dir):
        return None

    seqs = sorted(os.listdir(word_dir), key=lambda x: int(x) if x.isdigit() else 0)
    for seq_id in seqs:
        seq_dir = os.path.join(word_dir, seq_id)
        if not os.path.isdir(seq_dir):
            continue
        frames = []
        for i in range(SEQUENCE_LEN):
            npy_path = os.path.join(seq_dir, f"{i}.npy")
            if os.path.isfile(npy_path):
                arr = np.load(npy_path)  # 1662 features
                pose = arr[:132]
                lh = arr[1536:1599]
                rh = arr[1599:1662]
                frames.append(np.concatenate([pose, lh, rh]).tolist())
            else:
                frames.append(np.zeros(MOTION_FEATURES).tolist())
        if frames:
            return frames
    return None


def signable_words(*, speech_dir=SPEECH_DATA_PATH, data_dir=DATA_PATH):
    words = set()
    if os.path.isdir(speech_dir):
        words.update(f[:-4].lower() for f in os.listdir(speech_dir) if f.endswith(".npy"))
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            word_dir = os.path.join(data_dir, name)
            if os.path.isdir(word_dir) and os.listdir(word_dir):
                words.add(name.lower())
    return sorted(words)
