from ultralytics import YOLO
import numpy as np
import supervision as sv
from detector import Detector
import cv2

class Tracker:
    def __init__(self, detector: Detector):
        self.detector = detector
        self._tracker = sv.ByteTrack()
        self._tracks = None

    def set_tracker(self, tracker):
        self._tracker = tracker

    def track(self, video: list[np.ndarray]) -> list:
        detections = self.detector.detect(video)
        tracks: list = list()
        for i, detection in enumerate(detections):
            dets = sv.Detections.from_ultralytics(detection)
            tracks.append(self._tracker.update_with_detections(dets))
        self._tracks = tracks
        return tracks

    def annotate_frames(self, video: list[np.ndarray]) -> list[np.ndarray]:
        tracks = self.track(video) if self._tracks is None else self._tracks
        for frame, tracker in zip(video, tracks):
            for bbox, tracker_id in zip(tracker.xyxy, tracker.tracker_id):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(tracker_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 4)
        return video

    def save_video(self, video: list[np.ndarray], path: str, fps: int = 25, resolution: tuple = (1280, 720)):
        annotated_video = self.annotate_frames(video)
        if not path.endswith('.mp4'):
            path += '.mp4'
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
        for frame in annotated_video:
            out.write(frame)
        out.release()