from ultralytics import YOLO
import numpy as np
import cv2

class KeypointDetector:

    def __init__(self, model: YOLO = None):
        self.model = model

    def detect(self, video: list[np.array], batch_size: int = 40) -> list:
        if self.model is None:
            raise ValueError("Model is not set")
        detections = list()
        for frame in video:
            results = self.model.predict(frame)
            detections += results
        return detections

    def get_keypoints(self, video: list[np.array]) -> list:
        detections = self.detect(video)
        keypoints = list()
        for detection in detections:
            keypoints.append(detection.keypoints.xy.view(-1, 2).tolist())
        self.keypoints = keypoints
        return keypoints


    def annotate_frames(self, video: list[np.array]) -> list[np.array]:
        keypoints_all_frames = self.get_keypoints(video)
        for frame, keypoints in zip(video, keypoints_all_frames):
            for keypoint in keypoints:
                x, y = keypoint
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        return video

    def save_video(self, video: list[np.array], path: str, fps: int = 25, resolution: tuple = (1280, 720)):
        annotated_video = self.annotate_frames(video)
        if not path.endswith('.mp4'):
            path += '.mp4'
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
        for frame in annotated_video:
            out.write(frame)
        out.release()