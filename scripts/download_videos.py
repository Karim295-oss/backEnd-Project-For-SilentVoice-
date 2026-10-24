import json
import os
import urllib.request
import urllib.error

VERIFIED_JSON = r"E:\Projects\asl_recognition\data\top20_verified.json"
VIDEO_DIR = r"E:\Projects\asl_recognition\data\videos"

def download_video(url, save_path, timeout=15):
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(save_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

def main():
    with open(VERIFIED_JSON, 'r') as f:
        data = json.load(f)

    os.makedirs(VIDEO_DIR, exist_ok=True)

    total_downloaded = 0
    total_failed = 0

    for entry in data:
        gloss = entry['gloss']
        instances = entry['instances']
        gloss_dir = os.path.join(VIDEO_DIR, gloss)
        os.makedirs(gloss_dir, exist_ok=True)

        print(f"\nDownloading '{gloss}' ({len(instances)} videos)...")

        for i, inst in enumerate(instances):
            url = inst.get('url', '')
            video_id = inst.get('video_id', f'{gloss}_{i}')
            save_path = os.path.join(gloss_dir, f"{video_id}.mp4")

            if os.path.exists(save_path):
                print(f"  [{i+1}/{len(instances)}] Already exists, skipping.")
                total_downloaded += 1
                continue

            print(f"  [{i+1}/{len(instances)}] Downloading {video_id}...")
            success = download_video(url, save_path)

            if success:
                print(f"    ✓ Saved to {save_path}")
                total_downloaded += 1
            else:
                total_failed += 1

    print(f"\n✅ Done.")
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Failed:     {total_failed}")

if __name__ == '__main__':
    main()
