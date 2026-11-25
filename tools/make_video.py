"""Place a portrait phone screen recording on a 1920x1080 canvas with timed captions (silent MP4 for voice-over).

Usage: python tools/make_video.py --input phone_recording.mp4 --offset 3.5 --out video/demo_1080p.mp4
  --offset: seconds into the recording at which you long-pressed the title (demo start).
Captions follow docs/DEMO_SCRIPT.md. Adds a 4 s title card and a 6 s closing card.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H = 1920, 1080
BG, INK, ORANGE, GREY = (237, 237, 237), (17, 17, 17), (31, 75, 255), (110, 110, 110)  # BGR

# (start s, end s after demo start, heading, line)
CAPTIONS = [
    (0, 8, "Offline inspection station", "Runs fully on the phone. No network permission."),
    (8, 20, "Tray inspected in under a second", "Every tablet found, scored and given a verdict."),
    (20, 30, "Defect, severity, confidence", "Heatmap shows where the defect is."),
    (30, 40, "Llama 3.2 on-device advice", "ExecuTorch on the phone CPU. Rules decide, Llama explains."),
    (40, 52, "Never a silent pass", "Touching items are flagged for the operator."),
    (52, 61, "Quality gate", "Glare, blur, dark or bright frames are refused."),
    (61, 70, "All clear", "A good tray is cleared at a glance."),
    (70, 84, "Local inspection log", "Verdict, severity and advice for every item. CSV export."),
    (84, 93, "Measured on this phone", "F1 0.95 on held-out images · ~51 ms vision per tablet · advice in ~5 s"),
    (93, 106, "Built with Llama", "Vision: ONNX Runtime · Advice: ExecuTorch, 4-bit Llama 3.2 1B · CPU only"),
]


def text(img, s, org, scale, colour, thick):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_DUPLEX, scale, colour, thick, cv2.LINE_AA)


def card(title, sub):
    img = np.full((H, W, 3), INK, np.uint8)
    cv2.circle(img, (160, 470), 12, ORANGE, -1)
    text(img, title, (190, 490), 2.6, (255, 255, 255), 3)
    text(img, sub, (190, 580), 1.1, (190, 190, 190), 2)
    return img


def wrap(s, n=34):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur] if cur else lines


def find_demo_start(path, reject_card_at=9.5):
    """Demo start = first frame showing the orange REJECT card (shown ~9.5 s into the demo) minus that delay."""
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            raise SystemExit("orange REJECT card not found; pass --offset explicitly")
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        b, g, r = f[..., 0].astype(int), f[..., 1].astype(int), f[..., 2].astype(int)
        orange = (r > 230) & (g > 50) & (g < 110) & (b < 60)
        if orange.mean() > 0.03:
            return max(0.0, t - reject_card_at)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--offset", type=float, default=-1.0, help="seconds; default: detect automatically")
    ap.add_argument("--out", default="video/demo_1080p.mp4")
    ap.add_argument("--segments", help="JSON list [rec_start, rec_end, heading, line, (seconds on screen)] tied to the recording; "
                    "if the on-screen time is longer than the segment the last frame is held, if shorter the segment is trimmed")
    ap.add_argument("--title-secs", type=float, default=4.0)
    ap.add_argument("--end-secs", type=float, default=6.0)
    a = ap.parse_args()
    # Phone screen recordings have a variable frame rate (frames only when the screen changes), so frames are
    # placed by their own timestamps on a constant 30 fps output timeline.
    fps = 30
    offset = a.offset if a.offset >= 0 else find_demo_start(a.input)
    print(f"demo start in recording: {offset:.1f} s")
    cap = cv2.VideoCapture(a.input)
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for img in [card("Pharma Inspector", "Offline AI tablet inspection on an Android phone · Built with Llama")] * int(a.title_secs * fps):
        out.write(img)
    # Segments are (recording start s, recording end s, heading, line). Gaps between segments are cut.
    if a.segments:
        segments = [tuple(s) for s in json.loads(Path(a.segments).read_text(encoding="utf-8"))]
    else:
        segments = [(offset + s, offset + e, h, l) for s, e, h, l in CAPTIONS]
    # Optional per segment: [5] source video (default --input), [6] badge text, [7] second phone
    # {"src", "from", "to", "label"} shown side by side (left label in [8]).
    class Reader:
        def __init__(self, src, start, end):
            self.cap = cv2.VideoCapture(src); self.start, self.end = start, end
            self.ok, self.frame = self.cap.read(); self.nok, self.nxt = self.cap.read()
            self.nt = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        def at(self, k):
            t = min(self.start + k / fps, self.end)
            while self.nok and self.nt <= t:
                self.frame = self.nxt; self.nok, self.nxt = self.cap.read(); self.nt = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            return self.frame

    def phone(canvas, frame, x0, ph, label=None):
        pw = int(frame.shape[1] * ph / frame.shape[0])
        y0 = (H - ph) // 2 + (18 if label else 0)
        cv2.rectangle(canvas, (x0 - 12, y0 - 12), (x0 + pw + 12, y0 + ph + 12), INK, -1)
        canvas[y0:y0 + ph, x0:x0 + pw] = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        if label:
            text(canvas, label, (x0 - 8, y0 - 24), 0.6, INK, 2)
        return pw

    ok = True
    for seg in segments:
      src = seg[5] if len(seg) > 5 and seg[5] else a.input
      main_r = Reader(src, seg[0], seg[1])
      second = seg[7] if len(seg) > 7 and seg[7] else None
      side_r = Reader(second["src"], second["from"], second["to"]) if second else None
      ok = main_r.ok
      shown = seg[4] if len(seg) > 4 else seg[1] - seg[0]
      for k in range(int(round(shown * fps))):
        if not ok:
            break
        frame = main_r.at(k)
        cap_now = seg
        canvas = np.full((H, W, 3), BG, np.uint8)
        if side_r:
            ph = H - 200
            pw = phone(canvas, frame, 1010, ph, seg[8] if len(seg) > 8 else "MVTec test images (replay)")
            phone(canvas, side_r.at(k), 1010 + pw + 50, ph, second.get("label", "Live camera"))
        else:
            phone(canvas, frame, 1180, H - 80)
        if cap_now:
            cv2.circle(canvas, (130, 380), 11, ORANGE, -1)
            for i, line in enumerate(wrap(cap_now[2], 22 if len(cap_now) > 7 and cap_now[7] else 24)):
                text(canvas, line, (160, 400 + i * 80), 2.0, INK, 3)
            for i, line in enumerate(wrap(cap_now[3], 38 if len(cap_now) > 7 and cap_now[7] else 44)):
                text(canvas, line, (160, 560 + i * 52), 1.05, GREY, 2)
        held = seg[0] + k / fps > seg[1] and (not side_r or second["from"] + k / fps > second["to"])
        if held:
            cv2.rectangle(canvas, (130, 120), (560, 168), (210, 210, 210), -1)
            text(canvas, "||  PAUSED FOR NARRATION", (150, 154), 0.8, INK, 2)
        else:
            badge = seg[6] if len(seg) > 6 and seg[6] else "LIVE SCREEN RECORDING ON THE PHONE  |  REAL TIME"
            bw = cv2.getTextSize(badge, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0][0]
            cv2.rectangle(canvas, (130, 120), (175 + bw + 20, 168), INK, -1)
            cv2.circle(canvas, (155, 144), 9, ORANGE, -1)
            text(canvas, badge, (175, 154), 0.8, (255, 255, 255), 2)
        text(canvas, "PHARMA INSPECTOR  |  OFFLINE  |  ON-DEVICE", (160, 980), 0.8, GREY, 1)
        out.write(canvas)
    for img in [card("AI quality control in your pocket.", "Pharma Inspector · github.com/Ksheerajaa/ps5-pharma-inspector")] * int(a.end_secs * fps):
        out.write(img)
    out.release()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
