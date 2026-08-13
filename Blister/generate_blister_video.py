"""
Generate a synthetic blister-pack inspection video.

Creates a ~20-second video showing:
  0-7s   : GOOD blister strip (all 10 tablets present)
  7-14s  : DEFECTIVE — missing tablet (one cavity empty)
  14-20s : GOOD again (all tablets back)

The strip is drawn as a rounded rectangle with circular tablet cavities,
on a dark matte background — mimicking a top-down fixed-camera setup.
Small random jitter simulates natural placement wobble.

Output: moving_blister.mp4  (same naming as Roshni's moving_bottle.mp4)
"""

import numpy as np
import cv2

# Video settings
OUTPUT = "moving_blister.mp4"
WIDTH, HEIGHT = 640, 480
FPS = 25
DURATION_SEC = 21

# Blister strip geometry (pixels)
STRIP_W, STRIP_H = 360, 120
TABLET_ROWS, TABLET_COLS = 2, 5
TABLET_RADIUS = 18
CAVITY_COLOR = (200, 200, 200)       # light grey cavity
TABLET_COLOR = (60, 130, 220)        # warm pill color
STRIP_COLOR = (180, 190, 200)        # aluminium foil
STRIP_BORDER = (140, 150, 160)
BG_COLOR = (30, 30, 35)              # dark matte surface


def draw_blister(frame, cx, cy, missing=None, crushed=None):
    """Draw a blister strip centred at (cx, cy).

    missing: set of (row, col) indices where the tablet is absent.
    crushed: set of (row, col) indices where the tablet is deformed.
    """
    missing = missing or set()
    crushed = crushed or set()

    x1 = cx - STRIP_W // 2
    y1 = cy - STRIP_H // 2
    x2 = cx + STRIP_W // 2
    y2 = cy + STRIP_H // 2

    # Strip body (rounded rectangle via filled rect + circles at corners)
    cv2.rectangle(frame, (x1 + 8, y1), (x2 - 8, y2), STRIP_COLOR, -1)
    cv2.rectangle(frame, (x1, y1 + 8), (x2, y2 - 8), STRIP_COLOR, -1)
    for corner_x, corner_y in [(x1+8, y1+8), (x2-8, y1+8), (x1+8, y2-8), (x2-8, y2-8)]:
        cv2.circle(frame, (corner_x, corner_y), 8, STRIP_COLOR, -1)

    # Border
    cv2.rectangle(frame, (x1 + 8, y1), (x2 - 8, y1 + 2), STRIP_BORDER, -1)
    cv2.rectangle(frame, (x1 + 8, y2 - 2), (x2 - 8, y2), STRIP_BORDER, -1)

    # Tablet cavities
    pad_x = STRIP_W // (TABLET_COLS + 1)
    pad_y = STRIP_H // (TABLET_ROWS + 1)

    for row in range(TABLET_ROWS):
        for col in range(TABLET_COLS):
            tx = x1 + pad_x * (col + 1)
            ty = y1 + pad_y * (row + 1)

            # Cavity (always drawn)
            cv2.circle(frame, (tx, ty), TABLET_RADIUS + 3, CAVITY_COLOR, -1)

            if (row, col) in missing:
                # Empty cavity — dark hole
                cv2.circle(frame, (tx, ty), TABLET_RADIUS, (80, 80, 90), -1)
            elif (row, col) in crushed:
                # Crushed — irregular shape
                pts = np.array([
                    [tx - 12, ty - 5],
                    [tx - 4, ty - 14],
                    [tx + 10, ty - 8],
                    [tx + 15, ty + 3],
                    [tx + 5, ty + 13],
                    [tx - 10, ty + 10],
                ], np.int32)
                cv2.fillPoly(frame, [pts], (40, 90, 160))
                # crack line
                cv2.line(frame, (tx - 8, ty - 6), (tx + 10, ty + 8), (30, 50, 100), 2)
            else:
                # Normal tablet
                cv2.circle(frame, (tx, ty), TABLET_RADIUS, TABLET_COLOR, -1)
                # Slight highlight for 3D feel
                cv2.circle(frame, (tx - 4, ty - 4), 6, (90, 160, 240), -1)

    # Print text on strip edge
    cv2.putText(frame, "MED 500mg", (x1 + 15, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 130), 1)


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = FPS * DURATION_SEC
    good_end = int(FPS * 7)
    defect_end = int(FPS * 14)

    rng = np.random.RandomState(42)

    for f in range(total_frames):
        frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)

        # Small jitter to simulate natural wobble
        jx = int(rng.uniform(-3, 3))
        jy = int(rng.uniform(-2, 2))
        cx = WIDTH // 2 + jx
        cy = HEIGHT // 2 + jy

        if f < good_end:
            # Phase 1: GOOD
            draw_blister(frame, cx, cy)
            phase = "GOOD STRIP"
        elif f < defect_end:
            # Phase 2: DEFECTIVE — missing tablet + crushed tablet
            draw_blister(frame, cx, cy,
                         missing={(0, 2)},
                         crushed={(1, 4)})
            phase = "DEFECTIVE STRIP"
        else:
            # Phase 3: GOOD again
            draw_blister(frame, cx, cy)
            phase = "GOOD STRIP"

        # Timestamp + phase label (like a real inspection feed)
        sec = f / FPS
        cv2.putText(frame, f"t={sec:05.1f}s  {phase}",
                    (10, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 100, 100), 1)

        writer.write(frame)

    writer.release()
    print(f"Saved: {OUTPUT}  ({total_frames} frames, {DURATION_SEC}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
