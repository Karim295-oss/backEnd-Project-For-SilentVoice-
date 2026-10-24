import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from sign_motions import load_sign_motion, signable_words


def write_speech_motion(speech_dir, word, value, shape=(30, 258)):
    np.save(os.path.join(speech_dir, f"{word}.npy"), np.full(shape, value, dtype=np.float32))


class SignMotionLookupTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.speech_dir = os.path.join(tmp.name, "speech")
        self.data_dir = os.path.join(tmp.name, "norm")
        os.makedirs(self.speech_dir)
        os.makedirs(self.data_dir)

    def load(self, word):
        return load_sign_motion(word, speech_dir=self.speech_dir, data_dir=self.data_dir)

    def test_speech_library_motion_is_returned_as_30_frames_of_258_features(self):
        write_speech_motion(self.speech_dir, "hello", 0.5)

        motion = self.load("hello")

        self.assertEqual(len(motion), 30)
        self.assertEqual(len(motion[0]), 258)
        self.assertEqual(motion[0][0], 0.5)

    def test_falls_back_to_normalized_training_data_sliced_to_pose_and_hands(self):
        seq_dir = os.path.join(self.data_dir, "water", "0")
        os.makedirs(seq_dir)
        for i in range(30):
            frame = np.zeros(1662)
            frame[:132] = 1.0        # pose
            frame[132:1536] = 9.0    # face: must be dropped
            frame[1536:1599] = 2.0   # left hand
            frame[1599:1662] = 3.0   # right hand
            np.save(os.path.join(seq_dir, f"{i}.npy"), frame)

        motion = self.load("water")

        self.assertEqual(len(motion), 30)
        self.assertEqual(motion[0], [1.0] * 132 + [2.0] * 63 + [3.0] * 63)

    def test_speech_motion_with_wrong_shape_is_rejected(self):
        write_speech_motion(self.speech_dir, "broken", 0.0, shape=(30, 100))

        with self.assertRaises(ValueError):
            self.load("broken")

    def test_speech_library_wins_over_training_data(self):
        write_speech_motion(self.speech_dir, "water", 0.5)
        seq_dir = os.path.join(self.data_dir, "water", "0")
        os.makedirs(seq_dir)
        for i in range(30):
            np.save(os.path.join(seq_dir, f"{i}.npy"), np.ones(1662))

        self.assertEqual(self.load("water")[0][0], 0.5)

    def test_unknown_word_returns_nothing(self):
        self.assertIsNone(self.load("zzzz"))

    def test_lookup_is_case_insensitive(self):
        write_speech_motion(self.speech_dir, "hello", 0.5)

        self.assertIsNotNone(self.load("HELLO"))


class SignableWordsTests(unittest.TestCase):
    def test_lists_words_from_both_stores_sorted_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            speech_dir = os.path.join(tmp, "speech")
            data_dir = os.path.join(tmp, "norm")
            os.makedirs(speech_dir)
            os.makedirs(os.path.join(data_dir, "water", "0"))
            os.makedirs(os.path.join(data_dir, "hello", "0"))
            write_speech_motion(speech_dir, "hello", 0.5)
            write_speech_motion(speech_dir, "apple", 0.5)

            words = signable_words(speech_dir=speech_dir, data_dir=data_dir)

        self.assertEqual(words, ["apple", "hello", "water"])

    def test_lowercases_words_and_skips_empty_training_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            speech_dir = os.path.join(tmp, "speech")
            data_dir = os.path.join(tmp, "norm")
            os.makedirs(speech_dir)
            os.makedirs(os.path.join(data_dir, "Water", "0"))
            os.makedirs(os.path.join(data_dir, "empty"))
            write_speech_motion(speech_dir, "Hello", 0.5)

            words = signable_words(speech_dir=speech_dir, data_dir=data_dir)

        self.assertEqual(words, ["hello", "water"])


class NoTensorFlowTests(unittest.TestCase):
    def test_loading_a_motion_and_listing_words_never_imports_tensorflow(self):
        code = (
            "import sys; import sign_motions as s; "
            "s.signable_words(); s.load_sign_motion('hello'); "
            "sys.exit(1 if 'tensorflow' in sys.modules else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
