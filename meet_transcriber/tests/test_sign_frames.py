import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sign_frames import HAND_JOINTS, POSE_JOINTS, motion_to_frames  # noqa: E402

# MediaPipe pose landmark indices of the upper-body joints we send.
LEFT_WRIST, RIGHT_WRIST = 15, 16


def motion_frame(pose=None, left=None, right=None):
    """One 258-feature MediaPipe frame: 33 pose landmarks (x, y, z, visibility), then
    21 left-hand and 21 right-hand landmarks (x, y, z). `pose` maps landmark index -> xyz."""
    vec = [0.0] * 258
    for idx, (x, y, z) in (pose or {}).items():
        vec[idx * 4:idx * 4 + 3] = [x, y, z]
        vec[idx * 4 + 3] = 1.0
    for offset, hand in ((132, left), (195, right)):
        for idx, (x, y, z) in (hand or {}).items():
            vec[offset + idx * 3:offset + idx * 3 + 3] = [x, y, z]
    return vec


class MotionToFramesTests(unittest.TestCase):
    def test_pose_is_flipped_to_godot_axes_with_the_nose_at_the_origin(self):
        frames = motion_to_frames([motion_frame(pose={0: (0, 0, 0), LEFT_WRIST: (0.5, 0.25, 0.1)})])

        pose = dict(zip(POSE_JOINTS, frames[0]["pose"]))
        self.assertEqual(pose["nose"], [0, 0, 0])
        self.assertEqual(pose["left_wrist"], [0.5, -0.25, -0.1])  # +Y up, +Z toward the viewer

    def test_undetected_pose_and_hands_are_null_not_a_point_at_the_nose(self):
        frames = motion_to_frames([motion_frame()])

        self.assertEqual(frames[0], {"pose": None, "left_hand": None, "right_hand": None})

    def test_detected_hand_has_21_joints_in_godot_axes(self):
        hand = {i: (0.01 * i, 0.02 * i, 0.0) for i in range(21)}
        frames = motion_to_frames([motion_frame(right=hand)])

        right = frames[0]["right_hand"]
        self.assertEqual(len(right), len(HAND_JOINTS))
        self.assertEqual(right[1], [0.01, -0.02, 0])   # y flipped
        self.assertIsNone(frames[0]["left_hand"])

    def test_hand_depth_is_rebased_on_the_pose_wrist(self):
        # MediaPipe hand z is relative to the hand's own wrist; the pose wrist sits at z=0.4
        # (toward the camera is negative), so Godot hand z = pose wrist z + hand-relative z.
        hand = {0: (0.3, 0.1, 0.0), 8: (0.3, 0.0, -0.05)}
        pose = {LEFT_WRIST: (0.3, 0.1, -0.4)}
        frames = motion_to_frames([motion_frame(pose=pose, left=hand)])

        left = frames[0]["left_hand"]
        self.assertEqual(left[0][2], 0.4)
        self.assertEqual(left[8][2], 0.45)


if __name__ == "__main__":
    unittest.main()
