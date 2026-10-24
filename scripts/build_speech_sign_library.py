import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe.solutions.holistic as mp_holistic
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_DIR / "models"
SL_VIDEO_DIR = REPO_DIR / "data" / "dataset" / "SL"
WLASL_ANNOTATIONS = REPO_DIR / "data" / "Wlasl" / "WLASL_v0.3.json"
WLASL_VIDEO_DIR = REPO_DIR / "data" / "Wlasl" / "videos"
ASL100_LABELS = MODELS_DIR / "actions_asl100.txt"
DEFAULT_OUTPUT_DIR = REPO_DIR / "data" / "speech_sign_library"
SEQUENCE_LEN = 30
FEATURE_WIDTH = 258
WORD_GLOSS = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*\Z")
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

sys.path.insert(0, str(MODELS_DIR))
from landmark_features import extract_keypoints


_WORKER_HOLISTIC = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract one normalized avatar sequence per available single-word "
            "gloss. This does not train or modify the sign-to-text model."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output folder (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-glosses",
        type=int,
        help="Process only the first N glosses (useful for a quick test).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel video-processing processes (default: 2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild sequences that already exist in the output folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List vocabulary and source clip counts without extracting landmarks.",
    )
    return parser.parse_args()


def load_sources():
    videos_by_gloss = defaultdict(list)

    if SL_VIDEO_DIR.is_dir():
        for gloss_dir in sorted(SL_VIDEO_DIR.iterdir()):
            if not gloss_dir.is_dir():
                continue
            gloss = gloss_dir.name.strip().casefold()
            if not WORD_GLOSS.fullmatch(gloss):
                continue
            for path in sorted(gloss_dir.iterdir()):
                if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES:
                    videos_by_gloss[gloss].append(path)

    if WLASL_ANNOTATIONS.is_file() and WLASL_VIDEO_DIR.is_dir():
        annotations = json.loads(WLASL_ANNOTATIONS.read_text(encoding="utf-8"))
        for entry in annotations:
            gloss = str(entry.get("gloss", "")).strip().casefold()
            if not WORD_GLOSS.fullmatch(gloss):
                continue
            for instance in entry.get("instances", []):
                video_id = str(instance.get("video_id", "")).strip()
                if video_id:
                    path = WLASL_VIDEO_DIR / f"{video_id}.mp4"
                    if path.is_file():
                        videos_by_gloss[gloss].append(path)

    for gloss, paths in videos_by_gloss.items():
        videos_by_gloss[gloss] = list(dict.fromkeys(paths))

    if not videos_by_gloss:
        raise FileNotFoundError(
            "No source clips found. Expected videos under "
            f"{SL_VIDEO_DIR} or {WLASL_VIDEO_DIR} with {WLASL_ANNOTATIONS}."
        )

    asl100 = {
        line.strip().casefold()
        for line in ASL100_LABELS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if ASL100_LABELS.is_file() else set()

    ordered_glosses = sorted(
        videos_by_gloss,
        key=lambda gloss: (gloss not in asl100, gloss),
    )
    return ordered_glosses, videos_by_gloss, asl100


def video_to_sequence(video_path, holistic):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None

    frames = []
    detected_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            if (
                results.pose_landmarks
                or results.left_hand_landmarks
                or results.right_hand_landmarks
            ):
                detected_frames += 1
            frames.append(extract_keypoints(results))
    finally:
        capture.release()

    if not frames or not detected_frames:
        return None

    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LEN, dtype=np.int64)
    sequence = np.asarray(frames, dtype=np.float32)[indices]
    if sequence.shape != (SEQUENCE_LEN, FEATURE_WIDTH):
        raise ValueError(
            f"Unexpected feature sequence from {video_path}: {sequence.shape}"
        )
    return sequence


def save_sequence(path, sequence):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.tmp.npy")
    np.save(temporary_path, sequence, allow_pickle=False)
    temporary_path.replace(path)


def initialize_worker():
    global _WORKER_HOLISTIC
    cv2.setNumThreads(1)
    _WORKER_HOLISTIC = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def process_gloss(task):
    gloss, video_paths = task
    errors = []
    for video_path in video_paths:
        try:
            sequence = video_to_sequence(Path(video_path), _WORKER_HOLISTIC)
        except Exception as error:
            errors.append((Path(video_path).name, str(error)))
            continue
        if sequence is not None:
            return gloss, sequence, video_path, errors
    return gloss, None, None, errors


def main():
    args = parse_args()
    if args.max_glosses is not None and args.max_glosses < 1:
        raise ValueError("--max-glosses must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    glosses, videos_by_gloss, asl100 = load_sources()
    if args.max_glosses is not None:
        glosses = glosses[:args.max_glosses]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Single-word glosses: {len(videos_by_gloss)} "
        f"(ASL-100 labels with clips: {len(asl100 & videos_by_gloss.keys())})"
    )
    print(f"Selected glosses: {len(glosses)}")
    print(f"Output: {args.output_dir}")
    if args.dry_run:
        for gloss in glosses[:20]:
            print(f"  {gloss}: {len(videos_by_gloss[gloss])} source clip(s)")
        if len(glosses) > 20:
            print(f"  ... and {len(glosses) - 20} more")
        return

    succeeded = 0
    failed = 0
    skipped = 0
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Expected a JSON object in {manifest_path}")
    else:
        manifest = {}

    tasks = []
    for index, gloss in enumerate(glosses, start=1):
        destination = args.output_dir / f"{gloss}.npy"
        if destination.is_file() and not args.overwrite:
            try:
                existing = np.load(destination, allow_pickle=False)
            except (OSError, ValueError) as error:
                print(
                    f"[{index}/{len(glosses)}] {gloss}: invalid saved file "
                    f"({error}); rebuilding"
                )
            else:
                if existing.shape == (SEQUENCE_LEN, FEATURE_WIDTH):
                    manifest[gloss] = {
                        "source": "existing",
                        "asl100": gloss in asl100,
                    }
                    skipped += 1
                    print(f"[{index}/{len(glosses)}] {gloss}: reused")
                    continue
                print(
                    f"[{index}/{len(glosses)}] {gloss}: wrong saved shape "
                    f"{existing.shape}; rebuilding"
                )
        tasks.append((gloss, [str(path) for path in videos_by_gloss[gloss]], index))

    if tasks:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
        ) as executor:
            futures = {
                executor.submit(process_gloss, (gloss, paths)): (index, gloss)
                for gloss, paths, index in tasks
            }
            for future in as_completed(futures):
                index, gloss = futures[future]
                result_gloss, sequence, selected_video, errors = future.result()
                if result_gloss != gloss:
                    raise RuntimeError(
                        f"Worker returned {result_gloss!r} for task {gloss!r}"
                    )
                for video_name, error in errors:
                    print(f"  Could not process {video_name}: {error}")

                if sequence is None:
                    print(f"[{index}/{len(glosses)}] {gloss}: no usable clip")
                    failed += 1
                    continue

                destination = args.output_dir / f"{gloss}.npy"
                save_sequence(destination, sequence)
                selected_video = Path(selected_video)
                manifest[gloss] = {
                    "source": str(selected_video.relative_to(REPO_DIR)),
                    "asl100": gloss in asl100,
                }
                succeeded += 1
                print(
                    f"[{index}/{len(glosses)}] {gloss}: saved "
                    f"(from {selected_video.name})"
                )

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Done. Built {succeeded}, reused {skipped}, unavailable {failed}; "
        f"speech-sign vocabulary: {succeeded + skipped} glosses."
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
