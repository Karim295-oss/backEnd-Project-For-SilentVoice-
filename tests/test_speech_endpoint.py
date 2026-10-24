import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, ROOT)

from sign_resolver import cached_resolver  # noqa: E402

HEAVY_MODULES = ["cv2", "mediapipe", "mediapipe.python", "mediapipe.python.solutions",
                 "mediapipe.solutions.holistic", "tensorflow", "tensorflow.keras",
                 "tensorflow.keras.models", "tgcn_asl100"]


def import_backend():
    """backend.py loads TensorFlow, MediaPipe and the Keras model on import; stub them out."""
    with mock.patch.dict(sys.modules, {name: mock.MagicMock() for name in HEAVY_MODULES}):
        sys.modules.pop("backend", None)
        import backend
    return backend


def motion(value, frames=30):
    return [[float(value)] * 258 for _ in range(frames)]


class SpeechToAslEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            raise unittest.SkipTest("fastapi test client (httpx) is not installed")
        cls.backend = import_backend()
        cls.client = TestClient(cls.backend.app)

    def setUp(self):
        library = {"happy": motion(1), "j": motion(2), "o": motion(3)}
        patches = [
            mock.patch.object(self.backend, "english_to_asl_words", lambda text: text.split()),
            mock.patch.object(self.backend, "resolve_word",
                              cached_resolver(library.get, synonyms={"glad": ["happy"]},
                                              wordnet_synonyms=lambda word: [])),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def post(self, text):
        response = self.client.post("/api/speech_to_asl", json={"text": text})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_unknown_words_are_returned_as_synonym_or_fingerspelled_signs(self):
        body = self.post("glad jo")

        self.assertEqual(body["english"], "glad jo")
        self.assertEqual(body["asl_words"], ["glad", "jo"])
        self.assertEqual(body["sequences"]["glad"], motion(1))
        self.assertEqual(len(body["sequences"]["jo"]), 20)  # 10 frames for each of j and o

    def test_words_with_nothing_signable_are_left_out(self):
        body = self.post("glad xyz")

        self.assertEqual(body["asl_words"], ["glad"])
        self.assertNotIn("xyz", body["sequences"])


if __name__ == "__main__":
    unittest.main()
