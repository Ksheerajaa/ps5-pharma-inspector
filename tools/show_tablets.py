"""Full-screen slideshow of MVTec pill test tablets for a live-camera demo (phone camera pointed at the screen).

Usage: python tools/show_tablets.py [--secs 6]
Keys: space / right arrow = next, left arrow = previous, q or Esc = quit.
Shows one tablet at a time, centred on black, so the phone sees a single item on a dark background.
Order: a good tablet first (use it for "Set reference item"), then defects and goods mixed.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("ml/data/pill")
ORDER = ["good", "crack", "good", "contamination", "color", "scratch", "good", "faulty_imprint", "combined", "pill_type"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=6.0, help="seconds per tablet (0 = manual)")
    a = ap.parse_args()
    split = json.loads(Path("ml/splits/pill.json").read_text())
    by_cls = {}
    for rel in split["eval"]:
        by_cls.setdefault(rel.split("/")[1], []).append(ROOT / rel)
    picks, used = [], {}
    for c in ORDER:
        i = used.get(c, 0)
        if i < len(by_cls.get(c, [])):
            picks.append((c, by_cls[c][i])); used[c] = i + 1
    win = "tablets"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    _, _, sw, sh = cv2.getWindowImageRect(win)
    sw, sh = max(sw, 1280), max(sh, 720)
    k = 0
    while True:
        cls, p = picks[k % len(picks)]
        img = cv2.imread(str(p))
        side = int(min(sw, sh) * 0.9)
        canvas = np.zeros((sh, sw, 3), np.uint8)
        tile = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
        y0, x0 = (sh - side) // 2, (sw - side) // 2
        canvas[y0:y0 + side, x0:x0 + side] = tile
        cv2.putText(canvas, f"MVTec test tablet ({cls})", (20, sh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        key = cv2.waitKeyEx(int(a.secs * 1000) if a.secs > 0 else 0)
        if key in (27, ord("q")):
            break
        k = k - 1 if key in (2424832, 65361) else k + 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
