from shirt_color import ShirtColor
from tracker import Tracker
import numpy as np
import cv2
from sklearn.cluster import KMeans
from supervision import Detections


class TeamAffiliation:
    def __init__(self, trucks: list[Detections], video: list[np.ndarray]):
        self.trucks = trucks
        self.video = video
        self.kmeans = KMeans(n_clusters=2)
        self.assigned_ids = {}
        self._team_colors = None

    def get_team_colors(self) -> tuple[list, list]:
        return self._team_colors

    @staticmethod
    def filter_non_players(color: list, rbg_color: list, class_id: list):
        colors = list()
        rgb_colors = list()
        for c, rgb, c_id in zip(color, rbg_color, class_id):
            if c_id == 0:
                colors.append(c)
                rgb_colors.append(rgb)
        return colors, rgb_colors

    def get_shirt_colors(self) -> tuple[list[list], list[list]]:
        shirt_colors = []
        rgb_shirt_colors = []
        teams = []
        for truck, frame in zip(self.trucks, self.video):
            shirt_color = ShirtColor(frame, truck.xyxy.tolist())
            color = shirt_color.get_shirt_color()
            rgb_color = shirt_color.get_rgb_shirt_color()
            class_id = truck.class_id.tolist()
            color, rgb_color = self.filter_non_players(color, rgb_color, class_id)
            shirt_colors.append(color)
            rgb_shirt_colors.append(rgb_color)
        return shirt_colors, rgb_shirt_colors

    def train_kmeans(self, shirt_colors: list[list]):
        color_values = [color for sublsit in shirt_colors for color in sublsit]
        color_values = np.array(color_values).reshape(-1, 1)
        self.kmeans.fit(color_values)

    def train_kmeans_rgb(self, shirt_colors: list[list]):
        color_values = [color for sublsit in shirt_colors for color in sublsit]
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
        track_id_to_team_map = {k: max(v, key=v.count) for k, v in self.assigned_ids.items()}
        for frame in frames_ids:
            teams.append([track_id_to_team_map[track_id] for track_id in frame])
        return teams

    def calculate_rgb_teams(self, shirt_colors: list) -> list:
        teams = []
        frames_ids = []
        for colors, tracks in zip(shirt_colors, self.trucks):
            frame_ids = []
            for color, track_id in zip(colors, tracks.tracker_id):
                team = self.kmeans.predict(color)[0]
                frame_ids.append(track_id)
                if track_id in self.assigned_ids.keys():
                    self.assigned_ids[track_id].append(team)
                else:
                    self.assigned_ids[track_id] = [team]
            frames_ids.append(frame_ids)
        track_id_to_team_map = {k: max(v, key=v.count) for k, v in self.assigned_ids.items()}
        for frame in frames_ids:
            teams.append([track_id_to_team_map[track_id] for track_id in frame])
        return teams

    def get_teams(self) -> list:
        shirt_colors, rgb_colors = self.get_shirt_colors()
        self.train_kmeans(shirt_colors)
        teams_list = self.calculate_teams(shirt_colors)
        return teams_list

    def get_rgb_teams(self) -> list:
        shirt_colors, rgb_colors = self.get_shirt_colors()
        self.train_kmeans_rgb(rgb_colors)
        teams_list = self.calculate_rgb_teams(rgb_colors)
        return teams_list

    def get_average_team_color(self, teams: list, rgb_colors: list) -> tuple[list, list]:
        team_1_color: list = []
        team_2_color: list = []
        for team, rgb in zip(teams, rgb_colors):
            for t, color in zip(team, rgb):
                if t == 0:
                    team_1_color.append(color)
                else:
                    team_2_color.append(color)
        team_1_color = [int(c) for c in np.mean(team_1_color, axis=0)]
        team_2_color = [int(c) for c in np.mean(team_2_color, axis=0)]
        self._team_colors = (team_1_color, team_2_color)
        return team_1_color, team_2_color

    def annotate_frames(self, mode: str = 'gray') -> list[np.ndarray]:
        if mode not in ['gray', 'rgb']:
            raise ValueError("Mode must be 'gray' or 'rgb'")
        shirt_colors, rgb_colors = self.get_shirt_colors()
        if mode == 'rgb':
            self.train_kmeans(rgb_colors)
            teams_list = self.calculate_teams(rgb_colors)
        else:
            self.train_kmeans(shirt_colors)
            teams_list = self.calculate_teams(shirt_colors)
        team_colors = self.get_average_team_color(teams_list, rgb_colors)
        for frame, truck, shirts, teams in zip(self.video, self.trucks, rgb_colors, teams_list):
            for bbox, tracker_id, color, team in zip(truck.xyxy, truck.tracker_id, shirts, teams):
                x1, y1, x2, y2 = map(int, bbox)
                team_color = team_colors[team]
                cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
                # cv2.putText(frame, str(tracker_id), (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, team_color, 4)
                cv2.rectangle(frame, (x2, y1), (x2 + 15, y1 - 15), color, -1)
        return self.video

    def save_video(self, path: str, fps: int = 25, mode = 'gray'):
        annotated_video = self.annotate_frames(mode)
        h, w, _ = annotated_video[0].shape
        if not path.endswith('.mp4'):
            path += '.mp4'
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in annotated_video:
            out.write(frame)
        out.release()
