import numpy as np
import cv2
from .radar import Radar
from matplotlib import pyplot as plt
from tqdm import tqdm


class VideoGenerator:
    def __init__(self, video: list[np.ndarray], radar: Radar):
        self.video = video
        self.radar = radar
        self.h, self.w = self.video[0].shape[:2]

    def get_frame_with_radar(self, frame: np.ndarray, radar, show: bool = False, path: str = None) -> np.ndarray:
        frame_height, frame_width, _ = frame.shape
        radar.canvas.draw()
        radar_w, radar_h = radar.canvas.get_width_height()
        img = np.frombuffer(radar.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(radar_h, radar_w, 4)
        img = np.roll(img, -1, axis=2)
        overlay_resized = cv2.resize(img, (frame_width // 3, frame_height // 3))
        overlay_height, overlay_width, _ = overlay_resized.shape

        frame_with_overlay = frame.copy()

        start_y = frame_height - overlay_height
        start_x = (frame_width - overlay_width) // 2
        end_y = frame_height
        end_x = start_x + overlay_width

        pitch_alpha = overlay_resized[..., 3] / 255.0
        for c in range(3):
            frame_with_overlay[start_y:end_y, start_x:end_x, c] = (
                    pitch_alpha * overlay_resized[..., c] +
                    (1 - pitch_alpha) * frame_with_overlay[start_y:end_y, start_x:end_x, c]
            )
        if show:
            cv2.imshow('Frame with Pitch', frame_with_overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if path is not None:
            cv2.imwrite(path, frame_with_overlay)
        return frame_with_overlay


    def generate_video_with_radar(self, path: str, fps: int = 30, n_frames_back: int = 15, show: bool = False) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path, fourcc, fps, (self.w, self.h))
        video_copy = self.video.copy()
        for i, frame in tqdm(enumerate(video_copy), total=len(video_copy)):
            radar, _ = self.radar.draw_trace(i, n_frames_back, draw=False)
            frame_with_radar = self.get_frame_with_radar(frame, radar)
            out.write(frame_with_radar)
            plt.close()
        out.release()
        cv2.destroyAllWindows()
        print(f'Video saved to {path}')