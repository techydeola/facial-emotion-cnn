"""Live facial emotion recognition demo.
Webcam -> Haar face detection -> CNN trained in train.ipynb -> label + confidence.

Run from the project folder:
    python demo.py                              # live demo (press q to quit)
    python demo.py --record figures/demo.gif    # also save a GIF of the session
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # hide TensorFlow's startup info messages

import argparse                                  # NEW
import time
import cv2
import numpy as np
from PIL import Image                            # NEW
from tensorflow import keras

MODEL_PATH = "models/emotion_cnn.keras"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CLASSES = ["angry", "fear", "happy", "sad", "surprise", "neutral"]   # same order as in training
SMOOTHING = 0.6      # share of the previous frames' prediction kept each frame (0 = no smoothing)
GIF_WIDTH = 480      # NEW: GIF frame width in pixels (keeps the file small)
GIF_MAX_FRAMES = 120 # NEW: cap on recorded frames (~15 s), so the GIF can't grow huge


def preprocess_face(gray, box):
    """Identical to training: grayscale crop -> 48x48 -> scale to 0-1 -> shape (1, 48, 48, 1)."""
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)
    return (face.astype("float32") / 255.0)[np.newaxis, ..., np.newaxis]


def draw_label(frame, text, x, y, colour):
    """Text on a filled box so it stays readable on any background."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    y = max(y, th + base + 8)
    cv2.rectangle(frame, (x, y - th - base - 8), (x + tw + 10, y), colour, -1)
    cv2.putText(frame, text, (x + 5, y - base - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def save_gif(frames, path, fps):                 # NEW
    """Save a list of RGB frames as a looping GIF."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Live facial emotion recognition demo")   # NEW
    parser.add_argument("--record", metavar="GIF_PATH",
                        help="save the session as a GIF, e.g. figures/demo.gif")
    args = parser.parse_args()

    model = keras.models.load_model(MODEL_PATH)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Check System Settings > Privacy & Security > Camera.")

    smooth, fps, prev = None, 0.0, time.time()
    gif_frames, frame_idx, rec_start = [], 0, time.time()                                  # NEW

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            box = max(faces, key=lambda b: b[2] * b[3])
            probs = model(preprocess_face(gray, box), training=False).numpy()[0]
            smooth = probs if smooth is None else SMOOTHING * smooth + (1 - SMOOTHING) * probs
            k = int(np.argmax(smooth))

            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_label(frame, f"{CLASSES[k]} {smooth[k]:.0%}", x, y, (0, 255, 0))
        else:
            smooth = None

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now
        status = f"FPS {fps:.0f}   q = quit" + ("   REC" if args.record else "")
        draw_label(frame, status, 10, 40, (255, 255, 255))

        cv2.imshow("Facial emotion demo", frame)

        # NEW: keep every 2nd frame, shrunk and converted to RGB, for the GIF
        if args.record and frame_idx % 2 == 0 and len(gif_frames) < GIF_MAX_FRAMES:
            h_small = int(frame.shape[0] * GIF_WIDTH / frame.shape[1])
            small = cv2.resize(frame, (GIF_WIDTH, h_small), interpolation=cv2.INTER_AREA)
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if args.record and gif_frames:                                                          # NEW
        gif_fps = len(gif_frames) / max(time.time() - rec_start, 1e-6)
        save_gif(gif_frames, args.record, gif_fps)
        print(f"Saved {len(gif_frames)} frames to {args.record} at {gif_fps:.1f} fps")


if __name__ == "__main__":
    main()