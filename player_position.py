import numpy as np
import cv2
from supervision import Detections

class PlayerPosition:
    def __init__(self):
        self._get_real_pitch_coordinates()

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

            [pitch_length / 2, 0.0],
            [pitch_length / 2, pitch_width / 2 - circle_radius],
            [pitch_length / 2, pitch_width / 2 + circle_radius],
            [pitch_length / 2, 68.0],

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
            [pitch_length, 68.0],

            [pitch_length / 2 - circle_radius, pitch_width / 2],
            [pitch_length / 2 + circle_radius, pitch_width / 2]
        ])
        self._real_pitch_coordinates = real_pitch_coordinates_full

    def _get_homography_matrix(self, keypoints: list[float]) -> np.array:
        keypoints = np.array(keypoints)
        mask = ~np.all(keypoints == [0.0, 0.0], axis=1)
        keypoints = keypoints[mask]
        pitch_coordinates = self._real_pitch_coordinates[mask]
        homography_matrix, _ = cv2.findHomography(keypoints, pitch_coordinates)
        return homography_matrix

    @staticmethod
    def _calculate_players_position(predictions: Detections, homography_matrix: np.ndarray) -> np.array:
        image_position = list()
        predictions_list = predictions.xyxy.tolist()
        for prediction in predictions_list:
            x1, y1, x2, y2 = prediction
            player_x = (x1 + x2) / 2
            player_y = y2
            image_position.append([player_x, player_y])
        players_position = cv2.perspectiveTransform(np.array(image_position).reshape(-1, 1, 2), homography_matrix)
        players_position = players_position.reshape(-1, 2)
        players_position[:, 1] = 68 - players_position[:, 1]
        return players_position

    def get_players_position(self, keypoints_all_frames: list[list], predictions_all_frames: list[Detections]) -> list:
        players_position = list()
        for keypoints, predictions in zip(keypoints_all_frames, predictions_all_frames):
            homography_matrix = self._get_homography_matrix(keypoints)
            players_position.append(self._calculate_players_position(predictions, homography_matrix))
        return players_position