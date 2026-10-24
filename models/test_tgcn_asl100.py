import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

from tgcn_asl100 import (
    extract_tgcn_keypoints,
    load_asl100_model,
    predict_asl100,
    prepare_tgcn_input,
)


class TGCNPreprocessingTests(unittest.TestCase):
    def test_asl100_labels_are_sorted_for_the_checkpoint_label_encoder(self):
        from tgcn_asl100 import LABELS_PATH

        labels = LABELS_PATH.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(labels), 100)
        self.assertEqual(labels, sorted(labels))

    def test_media_pipe_landmarks_map_to_55_normalized_points(self):
        pose = [SimpleNamespace(x=index / 40, y=index / 50) for index in range(33)]
        hand = [SimpleNamespace(x=index / 25, y=index / 30) for index in range(21)]
        results = SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=pose),
            left_hand_landmarks=SimpleNamespace(landmark=hand),
            right_hand_landmarks=SimpleNamespace(landmark=hand),
        )

        keypoints = extract_tgcn_keypoints(results)

        self.assertEqual(keypoints.shape, (55, 2))
        np.testing.assert_allclose(keypoints[0], (-1.0, -1.0))
        np.testing.assert_allclose(keypoints[1], (2 * (11.5 / 40) - 1, 2 * (11.5 / 50) - 1))
        np.testing.assert_allclose(keypoints[13], (-1.0, -1.0))
        np.testing.assert_allclose(keypoints[54], (2 * (20 / 25) - 1, 2 * (20 / 30) - 1))

    def test_missing_landmarks_use_the_training_zero_coordinate(self):
        results = SimpleNamespace(
            pose_landmarks=None,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
        )

        keypoints = extract_tgcn_keypoints(results)

        self.assertEqual(keypoints.shape, (55, 2))
        np.testing.assert_array_equal(keypoints, np.full((55, 2), -1.0, dtype=np.float32))

    def test_resamples_50_frames_into_model_input_layout(self):
        sequence = np.zeros((3, 55, 2), dtype=np.float32)
        sequence[1] = 1.0
        sequence[2] = 2.0

        model_input = prepare_tgcn_input(sequence)

        self.assertEqual(model_input.shape, (1, 55, 100))
        np.testing.assert_array_equal(model_input[0, 0, :2], (0.0, 0.0))
        np.testing.assert_array_equal(model_input[0, 0, -2:], (2.0, 2.0))

    def test_rejects_empty_and_wrong_shape_sequences(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            prepare_tgcn_input([])
        with self.assertRaisesRegex(ValueError, "Expected frames"):
            prepare_tgcn_input(np.zeros((4, 33, 2), dtype=np.float32))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for checkpoint inference")
class TGCNCheckpointTests(unittest.TestCase):
    def test_asl100_checkpoint_loads_and_returns_probabilities(self):
        model, labels = load_asl100_model()
        model_input = prepare_tgcn_input(np.zeros((1, 55, 2), dtype=np.float32))

        probabilities = predict_asl100(model, model_input)

        self.assertEqual(len(labels), 100)
        self.assertEqual(probabilities.shape, (100,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
