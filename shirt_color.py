import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2


class ShirtColor:
    def __init__(self, frame: np.ndarray, boxes: list[np.ndarray], true_values: list[int] = None):
        self.frame = frame
        self.boxes = boxes
        self.true_values = true_values
        self.crop_factor = 0.25
        self.colors = list()
        self.labels = list()

    def set_frame(self, frame: np.ndarray):
        self.frame = frame

    def set_boxes(self, boxes: list[np.ndarray]):
        self.boxes = boxes

    def set_crop_factor(self, crop_factor: float):
        self.crop_factor = crop_factor

    def set_true_values(self, true_values: list[int]):
        self.true_values = true_values

    def _get_players(self) -> list[np.ndarray]:
        players = []
        for bbox in self.boxes:
            x, y, x2, y2 = map(lambda elem: int(elem), bbox)
            player = self.frame[y:y2, x:x2]
            players.append(player)
        return players

    def _get_kits(self) -> list[np.ndarray]:
        kits = []
        players = self._get_players()
        for player in players:
            kit = player[player.shape[0] // 6:player.shape[0] // 2,
                  int(player.shape[1] * self.crop_factor):int(player.shape[1] * (1 - self.crop_factor))]
            kits.append(kit)
        return kits

    def get_shirt_color(self) -> list[np.float64]:
        mean_colors: list[int] = []
        kits = self._get_kits()
        for kit in kits:
            im = Image.fromarray(kit)
            im = im.convert('P', colors=128)
            im = np.array(im)
            mean_color = np.mean(im, axis=(0, 1))
            mean_colors.append(mean_color)
        self.colors = mean_colors
        return mean_colors

    def get_rgb_shirt_color(self) -> list[np.float64]:
        mean_colors: list[int] = []
        kits = self._get_kits()
        for kit in kits:
            mean_color = np.mean(kit, axis=(0, 1))
            mean_colors.append(mean_color)
        self.rgb_colors = mean_colors
        return mean_colors

    def predict_team(self) -> np.array:
        if len(self.colors) == 0:
            return
        color_values = np.array(self.colors).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(color_values)
        self.labels = kmeans.labels_
        return kmeans.labels_

    def get_accuracy(self) -> np.float64 | None:
        if len(self.labels) == 0:
            return
        return np.mean(self.labels == self.true_values)

    def run_prediction(self) -> np.float64 | None:
        self.get_shirt_color()
        self.predict_team()
        return self.get_accuracy()

    def get_predicted_colors(self) -> tuple:
        colors = self.get_shirt_color()
        teams = self.predict_team()
        return colors, teams

    def plot(self, c: list, return_plot: bool = False):
        plt.scatter(range(1, len(self.colors) + 1), self.colors, c=c)
        plt.xlabel('Player')
        plt.ylabel('Mean color')
        plt.legend(['Team1', 'Team2'])
        if not return_plot:
            plt.show()

    def plot_predicted_colors(self, return_plot: bool = False):
        if len(self.colors) == 0 or len(self.labels) == 0:
            return
        self.plot(self.labels, return_plot)

    def plot_true_colors(self, return_plot: bool = False):
        if len(self.colors) == 0:
            return
        self.plot(self.true_values, return_plot)
