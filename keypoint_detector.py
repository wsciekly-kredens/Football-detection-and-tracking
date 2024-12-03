from ultralytics import YOLO
import numpy as np
import cv2


class KeypointDetector:

    def __init__(self, model: YOLO = None):
        self.model = model
        self._detections = None
        self._keypoints = None

    def detect(self, video: list[np.ndarray], batch_size: int = 40) -> list:
        if self.model is None:
            raise ValueError("Model is not set")
        detections = list()
        for frame in video:
            results = self.model.predict(frame)
            detections += results
        self._detections = detections
        self.calculate_keypoints(video)
        return detections

    def get_keypoints(self):
        if self._keypoints is None:
            raise ValueError("No keypoints calculated")
        return self._keypoints

    def calculate_keypoints(self, video: list[np.ndarray]) -> list:
        detections = self.detect(video) if self._detections is None else self._detections
        keypoints = list()
        for detection in detections:
            keypoints.append(detection.keypoints.xy[0].view(-1, 2).tolist())
        self._keypoints = keypoints
        return keypoints


    def _get_n_keypoints(self, i: int) -> int:
        keypoints_list = self._keypoints
        keypoints = keypoints_list[i]
        n = 0
        for keypoint in keypoints:
            if keypoint[0] != 0 or keypoint[1] != 0:
                n += 1
        return n

    def fill_blank(self, video: list[np.ndarray]) -> list:
        keypoints = self.calculate_keypoints(video) if self._keypoints is None else self._keypoints
        last_valid = None
        for i, kp in enumerate(keypoints):
            keypoints_n = self._get_n_keypoints(i)
            if len(kp) == 0 or keypoints_n < 4:
                keypoints[i] = last_valid
            else:
                last_valid = kp
        return keypoints

    def annotate_frames(self, video: list[np.ndarray]) -> list[np.ndarray]:
        keypoints_all_frames = self.keypoints_fill_blank(video)
        for frame, keypoints in zip(video, keypoints_all_frames):
            for keypoint in keypoints:
                x, y = keypoint
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        return video

    def save_video(self, video: list[np.ndarray], path: str, fps: int = 25, resolution: tuple = (1280, 720)):
        annotated_video = self.annotate_frames(video)
        if not path.endswith('.mp4'):
            path += '.mp4'
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
        for frame in annotated_video:
            out.write(frame)
        out.release()
