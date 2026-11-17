"""
mediapipe_patch.py
==================
Makes `import mediapipe.solutions.<name>` work on every MediaPipe version.

- MediaPipe <= 0.10.21 ships the real solutions API under `mediapipe.python.solutions`.
  We only alias it as `mediapipe.solutions`, so the real Holistic model (the one the
  sign model was trained with) is used unchanged.
- MediaPipe >= 0.10.30 removed the solutions API. Only then do we build a stand-in
  Holistic from the Tasks API (PoseLandmarker + HandLandmarker). It has no face
  landmarks (this project never uses them) and its landmarks are close to, but not
  identical to, the old Holistic ones, so recognition accuracy may drop a little.

Usage: `import mediapipe_patch` before any `import mediapipe.solutions.*`.
`USING_COMPAT_SHIM` tells you which path was taken.
"""
import importlib
import os
import sys
import types

import mediapipe as mp

_SUBMODULES = ("holistic", "drawing_utils", "drawing_styles", "hands", "pose")


def _alias_real_solutions() -> bool:
    """Expose mediapipe.python.solutions as mediapipe.solutions. False if it doesn't exist."""
    try:
        real = importlib.import_module("mediapipe.python.solutions")
        for name in _SUBMODULES:
            importlib.import_module(f"mediapipe.python.solutions.{name}")
    except ImportError:
        return False
    sys.modules["mediapipe.solutions"] = real
    for name in _SUBMODULES:
        sys.modules[f"mediapipe.solutions.{name}"] = sys.modules[f"mediapipe.python.solutions.{name}"]
    mp.solutions = real
    return True


def _install_shim() -> None:
    import urllib.request

    import mediapipe.tasks as mp_tasks

    models_dir = os.path.join(os.path.expanduser("~"), ".mediapipe_compat_models")
    pose_url = ("https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
    hand_url = ("https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

    def get_model(url: str, filename: str) -> str:
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            print(f"[mediapipe_patch] Downloading {filename} ...", flush=True)
            tmp = path + ".part"  # a failed download must not leave a broken model behind
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        return path

    class Lm:
        __slots__ = ("x", "y", "z", "visibility")

        def __init__(self, src):
            self.x, self.y, self.z = src.x, src.y, src.z
            # Tasks landmarks may carry visibility=None; old Holistic always had a float.
            self.visibility = getattr(src, "visibility", None) or 0.0

    class LmList:
        __slots__ = ("landmark",)

        def __init__(self, src_list):
            self.landmark = [Lm(lm) for lm in src_list]

    class Classification:
        def __init__(self, label, score):
            self.label, self.score = label, score

    class ClassificationList:
        def __init__(self, label, score):
            self.classification = [Classification(label, score)]

    class HolisticResult:
        def __init__(self):
            self.pose_landmarks = None
            self.left_hand_landmarks = None
            self.right_hand_landmarks = None
            self.face_landmarks = None  # not supported (not used by this project)
            self.hand_labels = []        # (Tasks label, score, LmList) for the Hands stand-in

    class Holistic:
        """Stand-in for mediapipe.solutions.holistic.Holistic."""

        def __init__(self, static_image_mode=False, min_detection_confidence=0.5,
                     min_tracking_confidence=0.5, **_ignored):
            self._static = static_image_mode
            self._ts_ms = 0
            vision = mp_tasks.vision
            mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=get_model(pose_url, "pose_landmarker_full.task")),
                running_mode=mode,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence))
            self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=get_model(hand_url, "hand_landmarker.task")),
                running_mode=mode,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence))

        def process(self, rgb_image):
            self._ts_ms += 33  # VIDEO mode needs increasing timestamps; ~30 fps
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            if self._static:
                pose_res, hand_res = self._pose.detect(img), self._hands.detect(img)
            else:
                pose_res = self._pose.detect_for_video(img, self._ts_ms)
                hand_res = self._hands.detect_for_video(img, self._ts_ms)

            out = HolisticResult()
            if pose_res.pose_landmarks:
                out.pose_landmarks = LmList(pose_res.pose_landmarks[0])
            for i, handedness in enumerate(hand_res.handedness):
                label, score = handedness[0].category_name, handedness[0].score
                out.hand_labels.append((label, score, LmList(hand_res.hand_landmarks[i])))
            self._assign_hands(out)
            return out

        @staticmethod
        def _assign_hands(out):
            """Give each detected hand to the signer's left or right side.

            The pose wrists (landmarks 15 = left, 16 = right) decide: each hand goes to the
            side whose pose wrist is nearest its own wrist, which matches what Holistic does.
            Without a pose, fall back to the Tasks label (checked against Holistic on WLASL
            clips: "Right" is the signer's right hand on the raw, un-mirrored frame)."""
            hands = [lm for _, _, lm in out.hand_labels]
            if not hands:
                return
            if out.pose_landmarks is None:
                for label, _, lm in out.hand_labels:
                    if label == "Right":
                        out.right_hand_landmarks = lm
                    else:
                        out.left_hand_landmarks = lm
                return
            pose = out.pose_landmarks.landmark

            def dist(hand, idx):
                w, p = hand.landmark[0], pose[idx]
                return (w.x - p.x) ** 2 + (w.y - p.y) ** 2

            if len(hands) == 1:
                h = hands[0]
                if dist(h, 15) <= dist(h, 16):
                    out.left_hand_landmarks = h
                else:
                    out.right_hand_landmarks = h
                return
            a, b = hands[0], hands[1]
            if dist(a, 15) + dist(b, 16) <= dist(a, 16) + dist(b, 15):
                out.left_hand_landmarks, out.right_hand_landmarks = a, b
            else:
                out.left_hand_landmarks, out.right_hand_landmarks = b, a

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

        def close(self):
            for landmarker in (self._pose, self._hands):
                try:
                    landmarker.close()
                except Exception:
                    pass

    class Hands:
        """Stand-in for mediapipe.solutions.hands.Hands (same labels as the old API)."""

        def __init__(self, *args, max_num_hands=2, **kwargs):
            self._inner = Holistic(*args, **kwargs)

        def process(self, rgb_image):
            r = self._inner.process(rgb_image)
            res = types.SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
            if r.hand_labels:
                res.multi_hand_landmarks = [lm for _, _, lm in r.hand_labels]
                res.multi_handedness = [ClassificationList(label, score)
                                        for label, score, _ in r.hand_labels]
            return res

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self._inner.close()

        def close(self):
            self._inner.close()

    class DrawingSpec:
        def __init__(self, color=(0, 255, 0), thickness=1, circle_radius=2):
            self.color, self.thickness, self.circle_radius = color, thickness, circle_radius

    def draw_landmarks(image, landmark_list, connections=None,
                       landmark_drawing_spec=None, connection_drawing_spec=None):
        """Small OpenCV version of mp_drawing.draw_landmarks (points and connections)."""
        if landmark_list is None:
            return
        import cv2

        h, w = image.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmark_list.landmark]
        line = connection_drawing_spec or DrawingSpec(color=(255, 255, 255))
        for a, b in connections or ():
            if a < len(pts) and b < len(pts):
                cv2.line(image, pts[a], pts[b], line.color, line.thickness)
        dot = landmark_drawing_spec or DrawingSpec()
        for p in pts:
            cv2.circle(image, p, dot.circle_radius, dot.color, -1)

    # Same index pairs as the old solutions API.
    hand_connections = frozenset([
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)])
    pose_connections = frozenset([
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
        (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
        (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)])
    empty = frozenset()

    def make_module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    sol = make_module("mediapipe.solutions")
    mp.solutions = sol
    sol.holistic = make_module("mediapipe.solutions.holistic", Holistic=Holistic,
                               POSE_CONNECTIONS=pose_connections,
                               HAND_CONNECTIONS=hand_connections,
                               FACEMESH_TESSELATION=empty, FACEMESH_CONTOURS=empty)
    sol.drawing_utils = make_module("mediapipe.solutions.drawing_utils",
                                    draw_landmarks=draw_landmarks, DrawingSpec=DrawingSpec)
    sol.drawing_styles = make_module(
        "mediapipe.solutions.drawing_styles",
        get_default_pose_landmarks_style=DrawingSpec,
        get_default_hand_landmarks_style=DrawingSpec,
        get_default_hand_connections_style=DrawingSpec,
        get_default_face_mesh_tesselation_style=DrawingSpec)
    sol.hands = make_module("mediapipe.solutions.hands", Hands=Hands,
                            HAND_CONNECTIONS=hand_connections)
    sol.pose = make_module("mediapipe.solutions.pose", Pose=Holistic,
                           POSE_CONNECTIONS=pose_connections)
    print("[mediapipe_patch] MediaPipe has no solutions API; using the Tasks API stand-in "
          "(no face landmarks, accuracy may differ slightly).", flush=True)


USING_COMPAT_SHIM = not _alias_real_solutions()
if USING_COMPAT_SHIM:
    _install_shim()
