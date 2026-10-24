import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = REPO_DIR / "data" / "msasl" / "annotations"
DEFAULT_OUTPUT_DIR = REPO_DIR / "data" / "msasl"
SPLITS = ("train", "val", "test")


def safe_name(value):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return name.strip("._") or "unknown"


def normalized_url(value):
    url = str(value or "").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def download_clip(url, start_time, end_time, destination, cookies, cookies_from_browser):
    output_template = str(destination.with_suffix("")) + ".%(ext)s"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--download-sections",
        f"*{start_time}-{end_time}",
        "--force-keyframes-at-cuts",
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "--output",
        output_template,
    ]
    if cookies:
        command.extend(["--cookies", str(cookies)])
    elif cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    command.append(url)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0 and destination.exists():
        return True, ""

    error = result.stderr.strip() or result.stdout.strip() or "yt-dlp did not create the clip"
    return False, error.splitlines()[-1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download trimmed MS-ASL clips into data/msasl."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Dataset splits to download (default: train val test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: data/msasl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum annotations to process per split.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of clips to download concurrently (default: 4).",
    )
    cookie_options = parser.add_mutually_exclusive_group()
    cookie_options.add_argument(
        "--cookies",
        type=Path,
        help="Netscape-format YouTube cookies file (keep it private; do not commit it).",
    )
    cookie_options.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER[:PROFILE]",
        help="Read YouTube cookies from a browser, for example chrome or edge.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned clips without downloading or creating files.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.cookies_from_browser and args.workers != 1:
        parser.error(
            "--cookies-from-browser requires --workers 1; use --cookies with multiple workers"
        )
    return args


def main():
    args = parse_args()
    if args.cookies and not args.cookies.is_file():
        print(f"Cookie file not found: {args.cookies}", file=sys.stderr)
        return 1
    if not args.dry_run:
        missing = []
        if importlib.util.find_spec("yt_dlp") is None:
            missing.append("yt-dlp (install with: python -m pip install yt-dlp)")
        if not shutil.which("ffmpeg"):
            missing.append("FFmpeg (install it and add ffmpeg to PATH)")
        if missing:
            print("Missing required downloader tools:", file=sys.stderr)
            for requirement in missing:
                print(f"  - {requirement}", file=sys.stderr)
            return 1

    downloaded = 0
    skipped = 0
    failed = 0

    for split in args.splits:
        annotation_path = ANNOTATION_DIR / f"MSASL_{split}.json"
        if not annotation_path.is_file():
            print(f"Missing annotation file: {annotation_path}", file=sys.stderr)
            failed += 1
            continue

        with annotation_path.open(encoding="utf-8") as annotation_file:
            entries = json.load(annotation_file)
        if args.limit is not None:
            entries = entries[:args.limit]

        print(f"{split}: processing {len(entries)} annotations")
        jobs = []
        for index, entry in enumerate(entries):
            url = normalized_url(entry.get("url"))
            start_time = entry.get("start_time")
            end_time = entry.get("end_time")
            gloss = safe_name(entry.get("clean_text") or entry.get("org_text"))
            filename = (
                f"{index:06d}_{gloss}_{round(float(start_time or 0) * 1000)}_"
                f"{round(float(end_time or 0) * 1000)}.mp4"
            )
            destination = args.output_dir / split / gloss / filename

            if not url or start_time is None or end_time is None or end_time <= start_time:
                print(f"  [{index + 1}/{len(entries)}] Invalid annotation; skipping")
                failed += 1
                continue
            if destination.exists():
                skipped += 1
                continue
            if args.dry_run:
                print(f"  {destination.relative_to(args.output_dir)}")
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((index, url, start_time, end_time, destination, gloss))

        if jobs:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_jobs = {
                    executor.submit(
                        download_clip,
                        url,
                        start_time,
                        end_time,
                        destination,
                        args.cookies,
                        args.cookies_from_browser,
                    ): (index, url, destination, gloss)
                    for index, url, start_time, end_time, destination, gloss in jobs
                }
                for future in as_completed(future_jobs):
                    index, url, destination, gloss = future_jobs[future]
                    success, error = future.result()
                    if success:
                        downloaded += 1
                        print(f"  [{index + 1}/{len(entries)}] Downloaded {gloss}")
                    else:
                        failed += 1
                        print(f"  [{index + 1}/{len(entries)}] Failed {url}: {error}")

    print(f"Done. Downloaded: {downloaded}; already present: {skipped}; failed: {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
