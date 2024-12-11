import numpy as np

class Detector:
    def __init__(self, model, conf: float):
        self.model = model
        self.conf = conf

    def detect(self, video: list[np.array], batch_size: int = 40) -> list:
        detections = list()
        for frame in video:
            results = self.model.predict(frame, conf=self.conf)
            detections += results
        return detections
