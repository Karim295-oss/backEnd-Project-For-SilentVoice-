import configparser
from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
TGCN_DIR = BASE_DIR / "asl" / "tgcn-wlasl"
CHECKPOINT_DIR = TGCN_DIR / "checkpoints" / "asl100"
LABELS_PATH = BASE_DIR / "models" / "actions_asl100.txt"
BODY25_KEEP = (0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18)
NUM_SAMPLES = 50
NUM_POINTS = 55


def extract_tgcn_keypoints(results):
    pose = results.pose_landmarks.landmark if results.pose_landmarks else None
    missing = (-1.0, -1.0)

    def point(index):
        if pose is None:
            return missing
        landmark = pose[index]
        return (2.0 * landmark.x - 1.0, 2.0 * landmark.y - 1.0)

    def midpoint(first, second):
        a = point(first)
        b = point(second)
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    body25_to_mediapipe = {
        0: point(0),
        1: midpoint(11, 12),
        2: point(12),
        3: point(14),
        4: point(16),
        5: point(11),
        6: point(13),
        7: point(15),
        8: midpoint(23, 24),
        15: point(5),
        16: point(2),
        17: point(8),
        18: point(7),
    }
    points = [body25_to_mediapipe[index] for index in BODY25_KEEP]

    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand:
            points.extend(
                (2.0 * landmark.x - 1.0, 2.0 * landmark.y - 1.0)
                for landmark in hand.landmark
            )
        else:
            points.extend([missing] * 21)

    keypoints = np.asarray(points, dtype=np.float32)
    if keypoints.shape != (NUM_POINTS, 2):
        raise ValueError(f"Expected {(NUM_POINTS, 2)} TGCN keypoints, got {keypoints.shape}")
    return keypoints


def prepare_tgcn_input(sequence):
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.size == 0:
        raise ValueError("Cannot predict from an empty frame sequence")
    if frames.ndim != 3 or frames.shape[1:] != (NUM_POINTS, 2):
        raise ValueError(f"Expected frames shaped (time, {NUM_POINTS}, 2), got {frames.shape}")

    frame_indices = np.linspace(0, len(frames) - 1, NUM_SAMPLES, dtype=np.int64)
    sampled = frames[frame_indices]
    return np.transpose(sampled, (1, 0, 2)).reshape(1, NUM_POINTS, NUM_SAMPLES * 2)


@lru_cache(maxsize=1)
def load_asl100_model():
    import torch

    config = configparser.ConfigParser()
    config_path = CHECKPOINT_DIR / "config.ini"
    if not config.read(config_path):
        raise FileNotFoundError(f"ASL-100 model config not found: {config_path}")

    labels = [line.strip() for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 100:
        raise ValueError(f"ASL-100 requires 100 ordered labels; found {len(labels)}")

    model_path = TGCN_DIR / "tgcn_model.py"
    spec = importlib.util.spec_from_file_location("silentvoice_tgcn_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TGCN architecture from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    train_config = config["TRAIN"]
    gcn_config = config["GCN"]
    model = module.GCN_muti_att(
        input_feature=int(train_config["NUM_SAMPLES"]) * 2,
        hidden_feature=int(gcn_config["HIDDEN_SIZE"]),
        num_class=len(labels),
        p_dropout=float(train_config["DROP_P"]),
        num_stage=int(gcn_config["NUM_STAGES"]),
    )

    checkpoint_path = CHECKPOINT_DIR / "pytorch_model.bin"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, labels


def predict_asl100(model, model_input):
    import torch

    with torch.inference_mode():
        logits = model(torch.from_numpy(model_input))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probabilities
