from shirt_color import ShirtColor
from tracker import Tracker
import numpy as np
import cv2
from sklearn.cluster import KMeans

class TeamAffiliation:
    def __init__(self, trucks: list, video: list[np.ndarray]):
        self.trucks = trucks
        self.video = video
        self.kmeans = KMeans(n_clusters=2)
        # TODO przypisanie do drużyny jako majority voting z przypisania po kolorach
        self.assigned_ids = {}

    def get_shirt_colors(self) -> tuple[list[list], list[list]]:
        shirt_colors = []
        rgb_shirt_colors = []
        teams = []
        for truck, frame in zip(self.trucks, self.video):
            shirt_color = ShirtColor(frame, truck.xyxy)
            color = shirt_color.get_shirt_color()
            rgb_color = shirt_color.get_rgb_shirt_color()
            shirt_colors.append(color)
            rgb_shirt_colors.append(rgb_color)
        return shirt_colors, rgb_shirt_colors

    def train_kmeans(self, shirt_colors: list[list]):
        color_values = [color for sublsit in shirt_colors for color in sublsit]
        color_values = np.array(color_values).reshape(-1, 1)
        self.kmeans.fit(color_values)

    def calculate_teams(self, shirt_colors: list) -> list:
        teams = []
        frames_ids = []
        for colors, tracks in zip(shirt_colors, self.trucks):
            frame_ids = []
            for color, track_id in zip(colors, tracks.tracker_id):
                team = self.kmeans.predict(np.reshape(color, (-1, 1)))[0]
                frame_ids.append(track_id)
                if track_id in self.assigned_ids.keys():
                    self.assigned_ids[track_id].append(team)
                else:
                    self.assigned_ids[track_id] = [team]
            frames_ids.append(frame_ids)
        track_id_to_team_map = {k : max(v, key=v.count) for k, v in self.assigned_ids.items()}
        for frame in frames_ids:
            teams.append([track_id_to_team_map[track_id] for track_id in frame])
        return teams

    def get_teams(self):
        shirt_colors, rgb_colors = self.get_shirt_colors()
        self.train_kmeans(shirt_colors)
        teams_list = self.calculate_teams(shirt_colors)
        return teams_list

    def annotate_frames(self, team_colors: tuple) -> list[np.ndarray]:
        shirt_colors, rgb_colors = self.get_shirt_colors()
        self.train_kmeans(shirt_colors)
        teams_list = self.calculate_teams(shirt_colors)
        for frame, truck, shirts, teams in zip(self.video, self.trucks, rgb_colors, teams_list):
            for bbox, tracker_id, color, team in zip(truck.xyxy, truck.tracker_id, shirts, teams):
                x1, y1, x2, y2 = map(int, bbox)
                team_color = team_colors[team]
                cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
                cv2.putText(frame, str(tracker_id), (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, team_color, 4)
                cv2.rectangle(frame, (x2, y1), (x2 + 15, y1 - 15), color, -1)
        return self.video

    def save_video(self, path: str, fps: int = 25, resolution: tuple = (1280, 720), team_colors: tuple = ((255, 255, 255), (0, 0, 0))):
        annotated_video = self.annotate_frames(team_colors)
        if not path.endswith('.mp4'):
            path += '.mp4'
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
        for frame in annotated_video:
            out.write(frame)
        out.release()