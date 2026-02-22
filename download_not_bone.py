"""
download_not_bone.py
Downloads diverse "not bone" images from public URLs into the not_bone/ folder.
Categories: animals, food, cars, nature, text/documents, faces, objects.
Uses only urllib (built-in) — no extra packages needed.
"""

import os
import urllib.request
import urllib.error
import time

OUT_DIR = "not_bone"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Public image URLs (diverse categories, all royalty-free / public domain) ──
# Using picsum.photos (random photos), and specific Wikimedia/Unsplash-source URLs
URLS = [
    # Random nature/landscape photos (Lorem Picsum — stable public CDN)
    ("picsum_001.jpg", "https://picsum.photos/seed/cat1/512/512"),
    ("picsum_002.jpg", "https://picsum.photos/seed/dog2/512/512"),
    ("picsum_003.jpg", "https://picsum.photos/seed/car3/512/512"),
    ("picsum_004.jpg", "https://picsum.photos/seed/food4/512/512"),
    ("picsum_005.jpg", "https://picsum.photos/seed/tree5/512/512"),
    ("picsum_006.jpg", "https://picsum.photos/seed/city6/512/512"),
    ("picsum_007.jpg", "https://picsum.photos/seed/sky7/512/512"),
    ("picsum_008.jpg", "https://picsum.photos/seed/flower8/512/512"),
    ("picsum_009.jpg", "https://picsum.photos/seed/ocean9/512/512"),
    ("picsum_010.jpg", "https://picsum.photos/seed/mountain10/512/512"),
    ("picsum_011.jpg", "https://picsum.photos/seed/book11/512/512"),
    ("picsum_012.jpg", "https://picsum.photos/seed/street12/512/512"),
    ("picsum_013.jpg", "https://picsum.photos/seed/bird13/512/512"),
    ("picsum_014.jpg", "https://picsum.photos/seed/fruit14/512/512"),
    ("picsum_015.jpg", "https://picsum.photos/seed/desk15/512/512"),
    ("picsum_016.jpg", "https://picsum.photos/seed/forest16/512/512"),
    ("picsum_017.jpg", "https://picsum.photos/seed/beach17/512/512"),
    ("picsum_018.jpg", "https://picsum.photos/seed/coffee18/512/512"),
    ("picsum_019.jpg", "https://picsum.photos/seed/laptop19/512/512"),
    ("picsum_020.jpg", "https://picsum.photos/seed/sunset20/512/512"),
    ("picsum_021.jpg", "https://picsum.photos/seed/rain21/512/512"),
    ("picsum_022.jpg", "https://picsum.photos/seed/snow22/512/512"),
    ("picsum_023.jpg", "https://picsum.photos/seed/fire23/512/512"),
    ("picsum_024.jpg", "https://picsum.photos/seed/water24/512/512"),
    ("picsum_025.jpg", "https://picsum.photos/seed/grass25/512/512"),
    ("picsum_026.jpg", "https://picsum.photos/seed/stone26/512/512"),
    ("picsum_027.jpg", "https://picsum.photos/seed/wood27/512/512"),
    ("picsum_028.jpg", "https://picsum.photos/seed/metal28/512/512"),
    ("picsum_029.jpg", "https://picsum.photos/seed/fabric29/512/512"),
    ("picsum_030.jpg", "https://picsum.photos/seed/paper30/512/512"),
    ("picsum_031.jpg", "https://picsum.photos/seed/glass31/512/512"),
    ("picsum_032.jpg", "https://picsum.photos/seed/plastic32/512/512"),
    ("picsum_033.jpg", "https://picsum.photos/seed/rubber33/512/512"),
    ("picsum_034.jpg", "https://picsum.photos/seed/leather34/512/512"),
    ("picsum_035.jpg", "https://picsum.photos/seed/ceramic35/512/512"),
    ("picsum_036.jpg", "https://picsum.photos/seed/concrete36/512/512"),
    ("picsum_037.jpg", "https://picsum.photos/seed/brick37/512/512"),
    ("picsum_038.jpg", "https://picsum.photos/seed/tile38/512/512"),
    ("picsum_039.jpg", "https://picsum.photos/seed/carpet39/512/512"),
    ("picsum_040.jpg", "https://picsum.photos/seed/wallpaper40/512/512"),
    ("picsum_041.jpg", "https://picsum.photos/seed/art41/512/512"),
    ("picsum_042.jpg", "https://picsum.photos/seed/painting42/512/512"),
    ("picsum_043.jpg", "https://picsum.photos/seed/sculpture43/512/512"),
    ("picsum_044.jpg", "https://picsum.photos/seed/architecture44/512/512"),
    ("picsum_045.jpg", "https://picsum.photos/seed/interior45/512/512"),
    ("picsum_046.jpg", "https://picsum.photos/seed/exterior46/512/512"),
    ("picsum_047.jpg", "https://picsum.photos/seed/vehicle47/512/512"),
    ("picsum_048.jpg", "https://picsum.photos/seed/transport48/512/512"),
    ("picsum_049.jpg", "https://picsum.photos/seed/technology49/512/512"),
    ("picsum_050.jpg", "https://picsum.photos/seed/electronics50/512/512"),
]

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; BoneResearchBot/1.0)"
}

print(f"Downloading {len(URLS)} images to '{OUT_DIR}/'...")
success, failed = 0, 0

for fname, url in URLS:
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"  [SKIP] {fname} already exists")
        success += 1
        continue
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"  [OK]   {fname}")
        success += 1
        time.sleep(0.2)  # polite delay
    except Exception as e:
        print(f"  [FAIL] {fname}: {e}")
        failed += 1

print(f"\nDone: {success} downloaded, {failed} failed.")
print(f"Total images in '{OUT_DIR}': {len(os.listdir(OUT_DIR))}")
