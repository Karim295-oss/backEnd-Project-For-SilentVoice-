import json
import urllib.request
import urllib.error
from collections import defaultdict
import os

WLASL_JSON = r"E:\Projects\asl_recognition\data\WLASL_v0.3.json"
OUTPUT_JSON = r"E:\Projects\asl_recognition\data\top20_verified.json"
TOP_N = 20

def check_url(url, timeout=5):
    try:
        req = urllib.request.Request(url, method='HEAD',
              headers={'User-Agent': 'Mozilla/5.0'})
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except:
        return False

def main():
    with open(WLASL_JSON, 'r') as f:
        data = json.load(f)

    # Count valid entries per gloss
    gloss_counts = {}
    for entry in data:
        gloss = entry['gloss']
        instances = entry.get('instances', [])
        gloss_counts[gloss] = len(instances)

    # Pick top 20 by instance count
    top20 = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    top20_glosses = [g for g, _ in top20]

    print(f"Top {TOP_N} signs selected:")
    for g, c in top20:
        print(f"  {g}: {c} instances")

    # Verify URLs for top 20
    verified = []
    for entry in data:
        if entry['gloss'] not in top20_glosses:
            continue

        valid_instances = []
        instances = entry.get('instances', [])
        total = len(instances)

        print(f"\nChecking '{entry['gloss']}' ({total} instances)...")
        for i, inst in enumerate(instances):
            url = inst.get('url', '')
            if url and check_url(url):
                valid_instances.append(inst)
                print(f"  [{i+1}/{total}] ✓ {url}")
            else:
                print(f"  [{i+1}/{total}] ✗ DEAD")

        print(f"  → {len(valid_instances)}/{total} alive")
        verified.append({
            'gloss': entry['gloss'],
            'instances': valid_instances
        })

    # Save verified data
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(verified, f, indent=2)

    print(f"\n✅ Done. Verified data saved to {OUTPUT_JSON}")
    print("\nSummary:")
    for entry in verified:
        print(f"  {entry['gloss']}: {len(entry['instances'])} valid videos")

if __name__ == '__main__':
    main()
