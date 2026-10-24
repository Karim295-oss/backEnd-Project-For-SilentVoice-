"""Sign motion -> Godot motion frames.

A sign motion is a list of 258-feature MediaPipe frames: 33 pose landmarks (x, y, z,
visibility), then 21 left-hand and 21 right-hand landmarks (x, y, z), already centered on
the nose. Godot frames carry 13 upper-body pose joints and 21 joints per hand in Godot axes
(+Y up, +X signer's left, +Z toward the viewer).
"""
from __future__ import annotations

# (joint name, MediaPipe pose landmark index)
_POSE = [("nose", 0), ("left_eye", 2), ("right_eye", 5), ("left_ear", 7), ("right_ear", 8),
         ("left_shoulder", 11), ("right_shoulder", 12), ("left_elbow", 13), ("right_elbow", 14),
         ("left_wrist", 15), ("right_wrist", 16), ("left_hip", 23), ("right_hip", 24)]
POSE_JOINTS = [name for name, _ in _POSE]
HAND_JOINTS = ["wrist"] + [f"{finger}_{part}"
                           for finger, parts in (("thumb", ("cmc", "mcp", "ip", "tip")),
                                                 *((f, ("mcp", "pip", "dip", "tip"))
                                                   for f in ("index", "middle", "ring", "pinky")))
                           for part in parts]


def motion_to_frames(motion) -> list[dict]:
    return [_frame(vec) for vec in motion]


def _frame(vec) -> dict:
    pose = _pose(vec[:132])
    return {"pose": pose,
            "left_hand": _hand(vec[132:195], pose, "left_wrist"),
            "right_hand": _hand(vec[195:258], pose, "right_wrist")}


def _pose(vec):
    """Undetected pose (all zeros) is None."""
    if not any(vec):
        return None
    return [_godot(vec[i * 4:i * 4 + 3]) for _, i in _POSE]


def _hand(vec, pose, wrist_joint):
    """MediaPipe hand depth is relative to the hand's wrist; re-base it on the pose wrist.
    Undetected hand (all zeros) is None."""
    if not any(vec):
        return None
    pose_wrist_z = pose[POSE_JOINTS.index(wrist_joint)][2] if pose else 0.0
    points = [vec[i * 3:i * 3 + 3] for i in range(21)]
    wrist_z = points[0][2]
    return [_godot((x, y, z - wrist_z), pose_wrist_z) for x, y, z in points]


def _godot(xyz, z_base=0.0):
    x, y, z = xyz
    return [round(x, 3) + 0.0, round(-y, 3) + 0.0, round(z_base - z, 3) + 0.0]
