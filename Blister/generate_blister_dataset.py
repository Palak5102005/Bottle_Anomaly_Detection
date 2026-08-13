"""
Generate synthetic blister-pack dataset images for training and testing.

Creates the exact folder structure that train_blister_patchcore.py and
evaluate_blister_pr.py expect:

    blister_dataset/
    ├── train/good/          (40 images — good strips with natural variation)
    └── test/
        ├── good/            (15 images)
        ├── missing_tablet/  (8 images)
        ├── crushed_tablet/  (8 images)
        └── torn_foil/       (8 images)

Each image = a single blister strip on a dark background, with small random
shifts in position, rotation, brightness, and tablet color to give PatchCore
realistic variation to learn from.
"""

import os

import cv2
import numpy as np

OUTPUT_ROOT = "blister_dataset"

# Image size (matches the ROI crop dimensions roughly)
IMG_W, IMG_H = 320, 240

# Strip geometry
STRIP_W, STRIP_H = 280, 100
TABLET_ROWS, TABLET_COLS = 2, 5
TABLET_RADIUS = 14


def draw_strip(img, cx, cy, rng, missing=None, crushed=None, torn=None):
    missing = missing or set()
    crushed = crushed or set()
    torn = torn or set()

    # Slight color variation per image
    b_off = int(rng.uniform(-10, 10))
    strip_col = (180 + b_off, 190 + b_off, 200 + b_off)
    tablet_col = (60 + b_off, 130 + b_off, 220 + b_off)

    x1 = cx - STRIP_W // 2
    y1 = cy - STRIP_H // 2
    x2 = cx + STRIP_W // 2
    y2 = cy + STRIP_H // 2

    # Strip body
    cv2.rectangle(img, (x1 + 6, y1), (x2 - 6, y2), strip_col, -1)
    cv2.rectangle(img, (x1, y1 + 6), (x2, y2 - 6), strip_col, -1)
    for corner_x, corner_y in [(x1+6, y1+6), (x2-6, y1+6), (x1+6, y2-6), (x2-6, y2-6)]:
        cv2.circle(img, (corner_x, corner_y), 6, strip_col, -1)

    pad_x = STRIP_W // (TABLET_COLS + 1)
    pad_y = STRIP_H // (TABLET_ROWS + 1)

    for row in range(TABLET_ROWS):
        for col in range(TABLET_COLS):
            tx = x1 + pad_x * (col + 1)
            ty = y1 + pad_y * (row + 1)

            cv2.circle(img, (tx, ty), TABLET_RADIUS + 2, (200, 200, 200), -1)

            if (row, col) in missing:
                cv2.circle(img, (tx, ty), TABLET_RADIUS, (80, 80, 90), -1)
            elif (row, col) in crushed:
                pts = np.array([
                    [tx-9, ty-4], [tx-3, ty-11], [tx+8, ty-6],
                    [tx+12, ty+2], [tx+4, ty+10], [tx-8, ty+8],
                ], np.int32)
                cv2.fillPoly(img, [pts], (40, 90, 160))
                cv2.line(img, (tx-6, ty-4), (tx+8, ty+6), (30, 50, 100), 2)
            elif (row, col) in torn:
                cv2.circle(img, (tx, ty), TABLET_RADIUS, tablet_col, -1)
                # Torn foil = jagged lines over the tablet
                for _ in range(3):
                    sx = tx + int(rng.uniform(-10, 10))
                    sy = ty + int(rng.uniform(-10, 10))
                    ex = tx + int(rng.uniform(-12, 12))
                    ey = ty + int(rng.uniform(-12, 12))
                    cv2.line(img, (sx, sy), (ex, ey), (140, 140, 150), 2)
            else:
                cv2.circle(img, (tx, ty), TABLET_RADIUS, tablet_col, -1)
                cv2.circle(img, (tx - 3, ty - 3), 4, (90, 160, 240), -1)

    cv2.putText(img, "MED 500mg", (x1 + 10, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 130), 1)


def generate_images(folder, count, rng, defect_type=None):
    os.makedirs(folder, exist_ok=True)
    for i in range(count):
        img = np.full((IMG_H, IMG_W, 3), (30, 30, 35), dtype=np.uint8)
        jx = int(rng.uniform(-8, 8))
        jy = int(rng.uniform(-5, 5))
        cx = IMG_W // 2 + jx
        cy = IMG_H // 2 + jy

        missing, crushed, torn = set(), set(), set()

        if defect_type == "missing_tablet":
            # Random 1-2 tablets missing
            for _ in range(rng.randint(1, 3)):
                missing.add((rng.randint(0, TABLET_ROWS), rng.randint(0, TABLET_COLS)))
        elif defect_type == "crushed_tablet":
            for _ in range(rng.randint(1, 3)):
                crushed.add((rng.randint(0, TABLET_ROWS), rng.randint(0, TABLET_COLS)))
        elif defect_type == "torn_foil":
            for _ in range(rng.randint(1, 3)):
                torn.add((rng.randint(0, TABLET_ROWS), rng.randint(0, TABLET_COLS)))

        draw_strip(img, cx, cy, rng, missing, crushed, torn)

        # Slight brightness jitter
        brightness = rng.uniform(0.9, 1.1)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)

        name = defect_type or "good"
        cv2.imwrite(os.path.join(folder, f"{name}_{i:04d}.jpg"), img)

    print(f"  {folder}: {count} images")


def main():
    rng = np.random.RandomState(123)

    print("Generating synthetic blister dataset...\n")

    generate_images(f"{OUTPUT_ROOT}/train/good", 40, rng)
    generate_images(f"{OUTPUT_ROOT}/test/good", 15, rng)
    generate_images(f"{OUTPUT_ROOT}/test/missing_tablet", 8, rng, "missing_tablet")
    generate_images(f"{OUTPUT_ROOT}/test/crushed_tablet", 8, rng, "crushed_tablet")
    generate_images(f"{OUTPUT_ROOT}/test/torn_foil", 8, rng, "torn_foil")

    print(f"\nDone. Dataset root: {OUTPUT_ROOT}/")
    print("Now run: python train_blister_patchcore.py")


if __name__ == "__main__":
    main()
