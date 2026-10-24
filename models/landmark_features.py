import numpy as np


def extract_keypoints(results):
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        cx, cy, cz = nose.x, nose.y, nose.z
        pose = np.array([
            [(landmark.x - cx), (landmark.y - cy), (landmark.z - cz), landmark.visibility]
            for landmark in results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(33 * 4)
        cx = cy = cz = 0.0

    if results.left_hand_landmarks:
        left_hand = np.array([
            [(landmark.x - cx), (landmark.y - cy), (landmark.z - cz)]
            for landmark in results.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left_hand = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        right_hand = np.array([
            [(landmark.x - cx), (landmark.y - cy), (landmark.z - cz)]
            for landmark in results.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right_hand = np.zeros(21 * 3)

    pose_xyz = np.delete(pose, np.arange(3, 132, 4))
    non_zero_coords = pose_xyz[pose_xyz != 0]
    pose_std = np.std(non_zero_coords) if len(non_zero_coords) > 0 else 1.0
    if pose_std < 0.01:
        pose_std = 1.0

    return np.concatenate((
        pose / pose_std,
        left_hand / pose_std,
        right_hand / pose_std,
    )).astype(np.float32)
