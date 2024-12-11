import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
import cv2


class ShirtColor:
    def __init__(self, frame: np.ndarray, boxes: list[np.ndarray], true_values: list[int] = None):
        self.frame = frame
        self.boxes = boxes
        self.true_values = true_values
        self.crop_factor = 0.25
        self.colors = list()
        self.labels = list()
        self.rgb_colors = list()
        self._kits = None

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

    def get_shirt_color(self) -> list[int]:
        mean_colors: list[int] = []
        kits = self._get_kits() if self._kits is None else self._kits
        for kit in kits:
            im = Image.fromarray(kit)
            im = im.convert('P', colors=128)
            im = np.array(im)
            mean_color = np.mean(im, axis=(0, 1))
            mean_colors.append(mean_color)
        self.colors = mean_colors
        return mean_colors

    def get_rgb_shirt_color(self) -> list[np.array]:
        mean_colors: list[int] = []
        kits = self._get_kits() if self._kits is None else self._kits
        for kit in kits:
            mean_color = np.mean(kit, axis=(0, 1))
            mean_colors.append(mean_color)
        self.rgb_colors = mean_colors
        return mean_colors

    def get_lab_shirt_color(self) -> list[int]:
        rgb_colors: list[np.array] = self.get_rgb_shirt_color()
        mean_colors = []
        for color in rgb_colors:
            color = color / 255
            color_lab = rgb2lab(color.reshape(1, 1, 3))[0][0]
            mean_colors.append(color_lab)
        return mean_colors

    def predict_team(self) -> np.array:
        if len(self.colors) == 0:
            return
        color_values = np.array(self.colors).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(color_values)
        self.labels = kmeans.labels_
        return kmeans.labels_

    def predict_team_with_rgb(self) -> np.array:
        if len(self.rgb_colors) == 0:
            return
        color_values = self.rgb_colors
        kmeans = KMeans(n_clusters=2, random_state=0).fit(color_values)
        self.labels = kmeans.labels_
        return kmeans.labels_

    def predict_team_with_lab(self) -> np.array:
        color_values = self.get_lab_shirt_color()
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

    def run_prediction_with_rgb(self) -> np.float64 | None:
        self.get_rgb_shirt_color()
        self.predict_team_with_rgb()
        return self.get_accuracy()

    def run_prediction_with_lab(self) -> np.float64 | None:
        self.predict_team_with_lab()
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

    def plot_kits(self, path: str = None, show: bool = False):
        kits = self._get_kits() if self._kits is None else self._kits
        total_width = sum(kit.shape[1] for kit in kits)
        max_height = max(kit.shape[0] for kit in kits)

        output_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

        current_x = 0
        for kit in kits:
            h, w = kit.shape[:2]
            output_image[0:h, current_x:current_x + w] = kit
            current_x += w
        if path is not None:
            cv2.imwrite(path, output_image)
        if show:
            plt.imshow(output_image)

    def plot_average_color(self, path: str = None, show: bool = False):
        kits = self._get_kits()
        shirt_colors = self.get_shirt_color()
        rgb_colors = self.get_rgb_shirt_color()
        total_width = sum(kit.shape[1] for kit in kits)
        class_colors = ((255, 255, 255), (0, 0, 0))

        max_kit_height = max(kit.shape[0] for kit in kits)
        swatch_height = max_kit_height // 10
        total_height = max_kit_height + 3 * swatch_height
        output_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        current_x = 0
        for i, kit in enumerate(kits):
            h, w = kit.shape[:2]
            avg_color = shirt_colors.pop(0)
            avg_rgb_color = rgb_colors.pop(0)
            class_color = class_colors[int(self.labels[i])]
            output_image[0:h, current_x:current_x + w] = kit
            color_swatch = np.full((swatch_height, w, 3), avg_color, dtype=np.uint8)
            rgb_swatch = np.full((swatch_height, w, 3), avg_rgb_color, dtype=np.uint8)
            class_swatch = np.full((swatch_height, w, 3), class_color, dtype=np.uint8)

            output_image[total_height  - 3 * swatch_height:total_height - 2 * swatch_height,
            current_x:current_x + w] = color_swatch
            output_image[total_height - 2 * swatch_height:total_height - swatch_height, current_x:current_x + w] = rgb_swatch
            output_image[total_height - swatch_height:total_height, current_x:current_x + w] = class_swatch

            current_x += w

        if path is not None:
            cv2.imwrite(path, output_image)
        if show:
            plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
