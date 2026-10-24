"""
mediapipe_patch.py
==================
Compatibility shim for mediapipe >= 0.10.30 which removed the `solutions` API.
This module reconstructs mediapipe.solutions.holistic, drawing_utils, hands, and pose
using the new mediapipe.tasks API, then injects them into sys.modules so that
all existing `import mediapipe.solutions.*` statements continue to work.

Usage: import this module ONCE, BEFORE any mediapipe.solutions imports.
       In backend.py: add `import mediapipe_patch` as the very first import.
"""
import sys
import os
import types
import urllib.request

import mediapipe as mp
import mediapipe.tasks as mp_tasks

# ── Model download ─────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".mediapipe_compat_models")

_POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _get_model(url: str, filename: str) -> str:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    path = os.path.join(_MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"[mediapipe_patch] Downloading {filename} …", flush=True)
        urllib.request.urlretrieve(url, path)
        print(f"[mediapipe_patch] {filename} ready.", flush=True)
    return path


# ── Landmark shims (mimic old NormalizedLandmark / NormalizedLandmarkList) ────
class _Lm:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, src):
        self.x = src.x
        self.y = src.y
        self.z = src.z
        self.visibility = getattr(src, "visibility", 0.0)


class _LmList:
    __slots__ = ("landmark",)

    def __init__(self, src_list):
        self.landmark = [_Lm(lm) for lm in src_list]


class _HolisticResult:
    __slots__ = (
        "pose_landmarks",
        "left_hand_landmarks",
        "right_hand_landmarks",
        "face_landmarks",
    )

    def __init__(self):
        self.pose_landmarks = None
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.face_landmarks = None  # not supported (not used by this project)


# ── Holistic ───────────────────────────────────────────────────────────────────
class Holistic:
    """Drop-in replacement for mediapipe.solutions.holistic.Holistic."""

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        **_ignored,
    ):
        self._static = static_image_mode
        self._ts_ms = 0

        mode = (
            mp_tasks.vision.RunningMode.IMAGE
            if static_image_mode
            else mp_tasks.vision.RunningMode.VIDEO
        )

        pose_path = _get_model(_POSE_URL, "pose_landmarker_full.task")
        hand_path = _get_model(_HAND_URL, "hand_landmarker.task")

        self._pose = mp_tasks.vision.PoseLandmarker.create_from_options(
            mp_tasks.vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=pose_path),
                running_mode=mode,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        self._hands = mp_tasks.vision.HandLandmarker.create_from_options(
            mp_tasks.vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=hand_path),
                running_mode=mode,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

    def process(self, rgb_image):
        """Process an RGB numpy image. Returns an object with pose/hand landmarks."""
        self._ts_ms += 33  # ~30 fps increment

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        if self._static:
            pose_res = self._pose.detect(mp_img)
            hand_res = self._hands.detect(mp_img)
        else:
            pose_res = self._pose.detect_for_video(mp_img, self._ts_ms)
            hand_res = self._hands.detect_for_video(mp_img, self._ts_ms)

        out = _HolisticResult()

        # Pose (33 landmarks)
        if pose_res.pose_landmarks:
            out.pose_landmarks = _LmList(pose_res.pose_landmarks[0])

        # Hands – Tasks API labels from the *camera* perspective.
        # "Left" in camera view = signer's right hand, and vice-versa.
        for i, handedness_list in enumerate(hand_res.handedness):
            label = handedness_list[0].category_name  # "Left" or "Right"
            lm_list = _LmList(hand_res.hand_landmarks[i])
            if label == "Left":
                out.right_hand_landmarks = lm_list
            else:
                out.left_hand_landmarks = lm_list

        return out

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        for attr in ("_pose", "_hands"):
            try:
                getattr(self, attr).close()
            except Exception:
                pass


# ── Drawing utils stubs ────────────────────────────────────────────────────────
class DrawingSpec:
    def __init__(self, color=(0, 255, 0), thickness=1, circle_radius=2):
        self.color = color
        self.thickness = thickness
        self.circle_radius = circle_radius


def draw_landmarks(
    image,
    landmark_list,
    connections=None,
    landmark_drawing_spec=None,
    connection_drawing_spec=None,
):
    """Minimal OpenCV-based landmark drawing (replaces mp_drawing.draw_landmarks)."""
    if landmark_list is None:
        return
    try:
        import cv2

        h, w = image.shape[:2]
        spec = landmark_drawing_spec
        color = spec.color if (spec and hasattr(spec, "color")) else (0, 255, 0)
        radius = spec.circle_radius if (spec and hasattr(spec, "circle_radius")) else 2
        for lm in landmark_list.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), radius, color, -1)
    except Exception:
        pass


POSE_CONNECTIONS: frozenset = frozenset()
HAND_CONNECTIONS: frozenset = frozenset()
FACEMESH_TESSELATION: frozenset = frozenset()
FACEMESH_CONTOURS: frozenset = frozenset()


# ── Hands stub ────────────────────────────────────────────────────────────────
class Hands:
    """Stub for mediapipe.solutions.hands.Hands (wraps Holistic)."""

    def __init__(self, *args, **kwargs):
        self._inner = Holistic(*args, **kwargs)

    def process(self, rgb_image):
        r = self._inner.process(rgb_image)

        class _HandResult:
            pass

        res = _HandResult()
        res.multi_hand_landmarks = []
        if r.left_hand_landmarks:
            res.multi_hand_landmarks.append(r.left_hand_landmarks)
        if r.right_hand_landmarks:
            res.multi_hand_landmarks.append(r.right_hand_landmarks)
        return res

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._inner.close()


# ── Inject into sys.modules ────────────────────────────────────────────────────
def _make_module(name: str, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# mediapipe.solutions
_sol = _make_module("mediapipe.solutions")
setattr(mp, "solutions", _sol)

# mediapipe.solutions.holistic
_hol = _make_module(
    "mediapipe.solutions.holistic",
    Holistic=Holistic,
    POSE_CONNECTIONS=POSE_CONNECTIONS,
)
setattr(_sol, "holistic", _hol)

# mediapipe.solutions.drawing_utils
_draw = _make_module(
    "mediapipe.solutions.drawing_utils",
    draw_landmarks=draw_landmarks,
    DrawingSpec=DrawingSpec,
    POSE_CONNECTIONS=POSE_CONNECTIONS,
    HAND_CONNECTIONS=HAND_CONNECTIONS,
    FACEMESH_TESSELATION=FACEMESH_TESSELATION,
    FACEMESH_CONTOURS=FACEMESH_CONTOURS,
)
setattr(_sol, "drawing_utils", _draw)

# mediapipe.solutions.drawing_styles (stub - used in some scripts)
_draw_styles = _make_module(
    "mediapipe.solutions.drawing_styles",
    get_default_pose_landmarks_style=lambda: DrawingSpec(),
    get_default_hand_landmarks_style=lambda: DrawingSpec(),
    get_default_face_mesh_tesselation_style=lambda: DrawingSpec(),
)
setattr(_sol, "drawing_styles", _draw_styles)

# mediapipe.solutions.hands
_hands_mod = _make_module(
    "mediapipe.solutions.hands",
    Hands=Hands,
    HAND_CONNECTIONS=HAND_CONNECTIONS,
)
setattr(_sol, "hands", _hands_mod)

# mediapipe.solutions.pose
_pose_mod = _make_module(
    "mediapipe.solutions.pose",
    Pose=Holistic,
    POSE_CONNECTIONS=POSE_CONNECTIONS,
)
setattr(_sol, "pose", _pose_mod)

print("[mediapipe_patch] mediapipe.solutions compatibility layer installed.", flush=True)
