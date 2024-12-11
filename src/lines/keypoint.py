import numpy as np
from .ellipse import Ellipse
from tqdm import tqdm
import pandas as pd
import random
from ultralytics import YOLO


class Keypoint:
    _line_name_mapping = {
        'Circle central': 0,
        # lewe pole karne
        'Big rect. left bottom': 1,
        'Big rect. left main': 2,
        'Big rect. left top': 3,
        'Small rect. left bottom': 4,
        'Small rect. left main': 5,
        'Small rect. left top': 6,
        # prawe pole karne
        'Big rect. right bottom': 7,
        'Big rect. right main': 8,
        'Big rect. right top': 9,
        'Small rect. right bottom': 10,
        'Small rect. right main': 11,
        'Small rect. right top': 12,
        # linie
        'Middle line': 13,
        'Side line bottom': 14,
        'Side line left': 15,
        'Side line right': 16,
        'Side line top': 17
    }
    _real_pitch_crossing_coordinates = {
        (0, -1): (52.5, 34),
        (0, 13): ((52.5, 24.85), (52.5, 43.15)),
        (13, 14): (52.5, 0),
        (13, 17): (52.5, 68),

        (1, 15): (0, 13),
        (2, 15): (0, 34),
        (3, 15): (0, 55),
        (4, 15): (0, 24.5),
        (5, 15): (0, 34),
        (6, 15): (0, 43.5),

        (7, 16): (105, 13),
        (8, 16): (105, 34),
        (9, 16): (105, 55),
        (10, 16): (105, 24.5),
        (11, 16): (105, 34),
        (12, 16): (105, 43.5),

        (14, 15): (0, 0),
        (15, 17): (0, 68),
        (14, 16): (105, 0),
        (16, 17): (105, 68)
    }


    def __init__(self, model: YOLO):
        self.model = model
        self._lines = {}
        self._get_crossing_matrix()
        self._get_real_pitch_coordinates()

    def _get_crossing_matrix(self):
        crossing_matrix = np.zeros((18, 18))
        crossing_matrix[0, 13] = 1
        crossing_matrix[13, 0] = 1
        crossing_matrix[13, 14] = 1
        crossing_matrix[14, 13] = 1
        crossing_matrix[13, 17] = 1
        crossing_matrix[17, 13] = 1

        crossing_matrix[1, 15] = 1
        crossing_matrix[15, 1] = 1
        crossing_matrix[1, 2] = 1
        crossing_matrix[2, 1] = 1
        crossing_matrix[3, 2] = 1
        crossing_matrix[2, 3] = 1
        crossing_matrix[3, 15] = 1
        crossing_matrix[15, 3] = 1

        crossing_matrix[4, 15] = 1
        crossing_matrix[15, 4] = 1
        crossing_matrix[4, 5] = 1
        crossing_matrix[5, 4] = 1
        crossing_matrix[6, 5] = 1
        crossing_matrix[5, 6] = 1
        crossing_matrix[6, 15] = 1
        crossing_matrix[15, 6] = 1

        crossing_matrix[14, 15] = 1
        crossing_matrix[15, 14] = 1
        crossing_matrix[15, 17] = 1
        crossing_matrix[17, 15] = 1

        crossing_matrix[7, 16] = 1
        crossing_matrix[16, 7] = 1
        crossing_matrix[7, 8] = 1
        crossing_matrix[8, 7] = 1
        crossing_matrix[9, 8] = 1
        crossing_matrix[8, 9] = 1
        crossing_matrix[9, 16] = 1
        crossing_matrix[16, 9] = 1

        crossing_matrix[10, 16] = 1
        crossing_matrix[16, 10] = 1
        crossing_matrix[10, 11] = 1
        crossing_matrix[11, 10] = 1
        crossing_matrix[12, 11] = 1
        crossing_matrix[11, 12] = 1
        crossing_matrix[12, 16] = 1
        crossing_matrix[16, 12] = 1

        crossing_matrix[14, 16] = 1
        crossing_matrix[16, 14] = 1
        crossing_matrix[16, 17] = 1
        crossing_matrix[17, 16] = 1

        self._crossing_matrix = crossing_matrix

    def _get_real_pitch_coordinates(self):
        pitch_length = 105
        pitch_width = 68
        corner_to_box_edge = 13.85
        box_edge_to_goal = 16.5
        goal_width = 7.32
        inside_box_edge_to_goal = 5.5
        box_length = 16.5
        inside_box_length = 5.5
        circle_radius = 9.15
        big_box_edge_to_inside_box_edge = box_edge_to_goal - inside_box_edge_to_goal

        real_pitch_coordinates_full = np.array([
            [0.0, 0.0],
            [0.0, corner_to_box_edge],
            [0.0, corner_to_box_edge + big_box_edge_to_inside_box_edge],
            [0.0, 68.0 - corner_to_box_edge - big_box_edge_to_inside_box_edge],
            [0.0, 68.0 - corner_to_box_edge],
            [0.0, 68.0],

            [inside_box_length, corner_to_box_edge + big_box_edge_to_inside_box_edge],
            [inside_box_length, 68 - corner_to_box_edge - big_box_edge_to_inside_box_edge],

            [11.0, pitch_width / 2],

            [box_length, corner_to_box_edge],
            [box_length, 34.0 - 7.66],
            [box_length, 34.0 + 7.66],
            [box_length, 68 - corner_to_box_edge],

            [pitch_length / 2 - circle_radius, pitch_width / 2],

            [pitch_length / 2, 0.0],
            [pitch_length / 2, pitch_width / 2 - circle_radius],
            [pitch_length / 2, pitch_width / 2 + circle_radius],
            [pitch_length / 2, 68.0],

            [pitch_length / 2 + circle_radius, pitch_width / 2],

            [pitch_length - box_length, corner_to_box_edge],
            [pitch_length - box_length, 34.0 - 7.66],
            [pitch_length - box_length, 34.0 + 7.66],
            [pitch_length - box_length, 68 - corner_to_box_edge],

            [pitch_length - 11.0, pitch_width / 2],

            [pitch_length - inside_box_length, corner_to_box_edge + big_box_edge_to_inside_box_edge],
            [pitch_length - inside_box_length, 68 - corner_to_box_edge - big_box_edge_to_inside_box_edge],

            [pitch_length, 0.0],
            [pitch_length, corner_to_box_edge],
            [pitch_length, corner_to_box_edge + big_box_edge_to_inside_box_edge],
            [pitch_length, 68.0 - corner_to_box_edge - big_box_edge_to_inside_box_edge],
            [pitch_length, 68.0 - corner_to_box_edge],
            [pitch_length, 68.0]
        ])
        self._real_pitch_coordinates = real_pitch_coordinates_full

    @staticmethod
    def get_line_pattern(points: list) -> tuple[np.float64, np.float64]:
        p1, p2 = points
        a = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p2[1] - a * p2[0]
        return np.float64(a), np.float64(b)

    @staticmethod
    def get_circle_pattern(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> tuple[float, float, float]:
        A = np.array([
            [x1, x2, 1],
            [x2, y2, 1],
            [x3, y3, 1]
        ])

        B = np.array([
            [-(x1 ** 2 + y1 ** 2)],
            [-(x2 ** 2 + y2 ** 2)],
            [-(x3 ** 2 + y3 ** 2)]
        ])

        d, e, f = np.linalg.solve(A, B).flatten()

        a = -d / 2
        b = -e / 2
        r = np.sqrt(pow(a, 2) + pow(b, 2) - f)

        return float(a), float(b), float(r)

    def get_func_pattern(self, df: pd.DataFrame) -> dict[dict]:
        df_list: list[pd.DataFrame] = [image_id for _, image_id in df.groupby('image_id')]
        image_dict = {key: {} for key in df['image_id'].unique()}

        for df_of_image in tqdm(df_list):
            image_id: int = int(df_of_image.iloc[0]['image_id'])
            points_dict: dict[str, list] = {key: [] for key in df_of_image['line_type'].unique()}
            for i, row in df_of_image.iterrows():
                points_dict[row['line_type']].append((row['x'], row['y']))
            for line in points_dict.keys():
                if 'Circle central' == line and len(points_dict[line]) >= 5:
                    points: list[tuple[float, float]] = random.sample(points_dict[line], 5)
                    points = [(x * 1920, y * 1080) for x, y in points]
                    ellipse = Ellipse(points)
                    line_id = self._get_line_name_mapping(line)
                    image_dict[image_id][line_id] = ellipse
                else:
                    points = random.sample(points_dict[line], 2)
                    points = [(x * 1920, y * 1080) for x, y in points]
                    func_pattern = self.get_line_pattern(points)
                    line_id = self._get_line_name_mapping(line)
                    image_dict[image_id][line_id] = func_pattern
        self._lines = image_dict
        return image_dict

    @staticmethod
    def get_linear_crossing_points(f1: tuple, f2: tuple, im_width: int = 1920, im_height: int = 1080) -> tuple | None:
        if len(f1) != len(f2) != 2:
            raise ValueError('Functions are not linear')
        x: float = (f2[1] - f1[1]) / (f1[0] - f2[0])
        y: float = f1[0] * x + f1[1]
        if x < 0 or x > im_width or y < 0 or y > im_height:
            return None
        return x, y

    @staticmethod
    def get_linear_with_ellipse_crossing(f1: tuple, ellipse: Ellipse, im_width: int = 1920,
                                         im_height: int = 1080) -> list | None:
        if len(f1) != 2:
            raise ValueError('Not right function type')
        # Ellipse like Ax^2 + Bxy + Cy^2 + Dx + Ey = 1
        A, B, C, D, E = ellipse.get_equation_parameters()
        # Linear like y = a1 * x + b1
        a1: float = f1[0]
        b1: float = f1[1]

        a: float = A + B * a1 + C * a1 ** 2
        b: float = B * b1 + 2 * C * b1 * a1 + D + E * a1
        c: float = C * pow(b1, 2) + E * b1 - 1

        delta: float = b ** 2 - 4 * a * c

        if delta < 0:
            return None

        delta_sqrt: float = np.sqrt(delta)
        x1: float = (-b + delta_sqrt) / (2 * a)
        y1: float = a1 * x1 + b1
        x2: float = (-b - delta_sqrt) / (2 * a)
        y2: float = a1 * x2 + b1

        x1 = x1 if 0 <= x1 <= im_width else None
        y1 = y1 if 0 <= y1 <= im_height else None
        x2 = x2 if 0 <= x2 <= im_width else None
        y2 = y2 if 0 <= y2 <= im_height else None

        to_return = list()

        center_x, center_y = ellipse.get_center()
        center_x = center_x if 0 <= center_x <= im_width else None
        center_y = center_y if 0 <= center_y <= im_height else None

        if x1 is not None and y1 is not None:
            to_return.append((x1, y1))

        if x2 is not None and y2 is not None:
            to_return.append((x2, y2))

        if center_x is not None and center_y is not None:
            to_return.append((center_x, center_y))

        if len(to_return) > 0:
            return to_return

        return None

    def get_all_keypoints_by_id(self, img_id: int) -> list:
        lines = self._lines[img_id]
        matches = list()
        keypoints = list()
        for line in lines.keys():
            possible_crossings = np.where(self._crossing_matrix[line] == 1)
            possible_crossings = possible_crossings[0].tolist()
            print(line)
            print(possible_crossings)
            for crossing in possible_crossings:
                if crossing in lines.keys():
                    if line < crossing:
                        matches.append((line, crossing))
                    else:
                        matches.append((crossing, line))
        matches = set(matches)
        for match in matches:
            if match[0] == 0:
                keypoints += self.get_linear_with_ellipse_crossing(lines[match[1]], lines[0])
            else:
                keypoints.append(self.get_linear_crossing_points(lines[match[0]], lines[match[1]]))

        return keypoints

    def get_all_crossing_points(self) -> list:
        pass

    def _get_line_name_mapping(self, name: str):
        return self._line_name_mapping[name]

    def create_yolo_training_file(self):
        flip_indx = [0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 14, 16, 15, 17]
        pass
