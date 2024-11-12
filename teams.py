from shirt_color import ShirtColor
from tracker import Tracker
import numpy as np
import cv2

class TeamAffiliation:
    def __init__(self, trucks: list, video: list[np.ndarray]):
        self.trucks = trucks
        self.video = video

    def get_shirt_colors(self) -> tuple:
        shirt_colors = []
        teams = []
        for truck, frame in zip(self.trucks, self.video):
            shirt_color = ShirtColor(frame, truck.xyxy)
            color, team = shirt_color.get_predicted_colors()
            shirt_colors.append(color)
            teams.append(team)
        return shirt_colors, teams

    def annotate_frames(self, team_colors: tuple) -> list[np.ndarray]:
        shirt_colors, teams_list = self.get_shirt_colors()
        for frame, truck, shirts, teams in zip(self.video, self.trucks, shirt_colors, teams_list):
            for bbox, tracker_id, color, team in zip(truck.xyxy, truck.tracker_id, shirts, teams):
                x1, y1, x2, y2 = map(int, bbox)
                team_color = team_colors[team]
                cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
                cv2.putText(frame, str(tracker_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, team_color, 4)
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), color, -1)
        return self.video

    def save_video(self, path: str, fps: int = 25, resolution: tuple = (1280, 720), team_colors: tuple = ((255, 255, 255), (0, 0, 0))):
        annotated_video = self.annotate_frames(team_colors)
        if not path.endswith('.mp4'):
            path += '.mp4'
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
        for frame in annotated_video:
            out.write(frame)
        out.release()