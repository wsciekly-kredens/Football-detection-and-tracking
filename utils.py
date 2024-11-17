import numpy as np
import cv2

def load_frames(source_path: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = list()
    cap = cv2.VideoCapture(source_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames