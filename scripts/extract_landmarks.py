import cv2
import mediapipe as mp
import numpy as np
import os
import json

VIDEO_DIR = r"E:\Projects\asl_recognition\data\videos"
LANDMARK_DIR = r"E:\Projects\asl_recognition\data\landmarks"
SEQUENCE_LENGTH = 30  # frames per sample

mp_hands = mp.solutions.hands

def extract_keypoints(results):
    """Extract and normalize landmarks from both hands."""
    # Left hand: 21 landmarks x 3 = 63 values
    # Right hand: 21 landmarks x 3 = 63 values
    # Total: 126 values per frame

    if results.multi_hand_landmarks and results.multi_handedness:
        left_hand = np.zeros(63)
        right_hand = np.zeros(63)

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
            ]).flatten()

            # Normalize: subtract wrist position (landmark 0)
            wrist = landmarks[:3]
            landmarks = landmarks - np.tile(wrist, 21)

            if label == 'Left':
                left_hand = landmarks
            else:
                right_hand = landmarks

        return np.concatenate([left_hand, right_hand])
    else:
        return np.zeros(126)


def video_to_sequence(video_path):
    """Convert a video to a fixed-length sequence of landmark frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            keypoints = extract_keypoints(results)
            frames.append(keypoints)

    cap.release()

    if len(frames) == 0:
        return None

    # Resize sequence to fixed length using interpolation
    frames = np.array(frames)  # (T, 126)
    T = frames.shape[0]

    if T == SEQUENCE_LENGTH:
        return frames
    
    # Interpolate to SEQUENCE_LENGTH frames
    indices = np.linspace(0, T - 1, SEQUENCE_LENGTH)
    resampled = np.array([frames[int(round(i))] for i in indices])
    return resampled  # (SEQUENCE_LENGTH, 126)


def main():
    os.makedirs(LANDMARK_DIR, exist_ok=True)

    glosses = sorted(os.listdir(VIDEO_DIR))
    label_map = {gloss: idx for idx, gloss in enumerate(glosses)}

    # Save label map
    with open(os.path.join(LANDMARK_DIR, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"Labels: {label_map}\n")

    total_ok = 0
    total_fail = 0

    for gloss in glosses:
        gloss_video_dir = os.path.join(VIDEO_DIR, gloss)
        gloss_landmark_dir = os.path.join(LANDMARK_DIR, gloss)
        os.makedirs(gloss_landmark_dir, exist_ok=True)

        videos = [v for v in os.listdir(gloss_video_dir) if v.endswith('.mp4')]
        print(f"Processing '{gloss}' ({len(videos)} videos)...")

        for i, video_file in enumerate(videos):
            video_path = os.path.join(gloss_video_dir, video_file)
            save_path = os.path.join(
                gloss_landmark_dir, video_file.replace('.mp4', '.npy'))

            if os.path.exists(save_path):
                print(f"  [{i+1}/{len(videos)}] Already exists, skipping.")
                total_ok += 1
                continue

            sequence = video_to_sequence(video_path)

            if sequence is not None:
                np.save(save_path, sequence)
                print(f"  [{i+1}/{len(videos)}] ✓ {video_file} → shape {sequence.shape}")
                total_ok += 1
            else:
                print(f"  [{i+1}/{len(videos)}] ✗ FAILED (no landmarks): {video_file}")
                total_fail += 1

    print(f"\n✅ Done.")
    print(f"  Extracted: {total_ok}")
    print(f"  Failed:    {total_fail}")


if __name__ == '__main__':
    main()
