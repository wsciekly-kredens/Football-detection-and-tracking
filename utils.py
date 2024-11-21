import numpy as np
import cv2

def load_frames_from_video(source_path: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = list()
    cap = cv2.VideoCapture(source_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def load_frames_from_images(source_path: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = list()
    for i in range(750):
        frame = cv2.imread(f'{source_path}/{i+1:06}.jpg')
        if frame is None:
            break
        frames.append(frame)
    return frames