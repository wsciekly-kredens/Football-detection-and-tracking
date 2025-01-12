import numpy as np
import cv2
import json
import pandas as pd
# from supervision import Detections


def load_frames_from_video(source_path: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = list()
    cap = cv2.VideoCapture(source_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_frames_from_images(source_path: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = list()
    for i in range(750):
        frame = cv2.imread(f'{source_path}/{i + 1:06}.jpg')
        if frame is None:
            break
        frames.append(frame)
    return frames


def tracks_to_list(tracks: list) -> list:
    detections = []
    for frame_id, result in enumerate(tracks):
        detections.append({
            "frame_id": frame_id + 1,
            "bbox": result.xyxy.tolist(),
            "confidence": result.confidence.tolist(),
            "class_id": result.class_id.tolist(),
            "tracker_id": result.tracker_id.tolist()
        })
    return detections


def save_tracks_as_json(tracks: list, path: str) -> None:
    detections = tracks_to_list(tracks)
    with open(path, 'w') as f:
        json.dump(detections, f, indent=4)


def keypoints_to_list(keypoints: list) -> list:
    keypoints_list = []
    for i, keypoint in enumerate(keypoints):
        keypoints_list.append(keypoint.keypoints.xy[0].tolist())
    return keypoints_list


def keypoints_to_json(keypoints: list, path: str) -> None:
    keypoints = keypoints_to_list(keypoints)
    keypoints_json = []
    for i, keypoint in enumerate(keypoints):
        keypoints_json.append({
            "frame_id": i + 1,
            "keypoints": keypoint
        })
    with open(path, 'w') as f:
        json.dump(keypoints_json, f, indent=1)


def load_keypoints_from_json(path: str) -> dict:
    with open(path, 'r') as f:
        keypoints_json = json.load(f)
    keypoints = {'frame_id': [], 'keypoints': []}
    for frame in keypoints_json:
        keypoints['frame_id'].append(frame['frame_id'])
        keypoints['keypoints'].append(frame['keypoints'])
    return keypoints


def load_tracks_from_json(path: str) -> dict:
    detections = {'frame_id': [], 'bbox': [], 'confidence': [], 'class_id': []}
    with open(path, 'r') as f:
        data = json.load(f)

    for frame in data:
        detections['frame_id'].append(frame['frame_id'])
        detections['bbox'].append(frame['bbox'])
        detections['confidence'].append(frame['confidence'])
        detections['class_id'].append(frame['class_id'])
    return detections


# def get_tracker_ids(tracks: Detections) -> list:
def get_tracker_ids(tracks) -> list:
    tracker_ids = []
    for tracker_id, class_id in zip(tracks.tracker_id.tolist(), tracks.class_id.tolist()):
        if class_id == 0:
            tracker_ids.append(tracker_id)
    return tracker_ids



def parse_player_data(player_positions: list[np.ndarray], tracks: list, player_teams: list) -> pd.DataFrame:
    dict_to_save = {'frame': [], 'track_id': [], 'x': [], 'y': [], 'team': []}
    frame_id = 1
    for frame_positions, frame_tracks, frame_teams in zip(player_positions, tracks, player_teams):
        dict_to_save['frame'] += [frame_id for _ in frame_positions]
        dict_to_save['track_id'] += get_tracker_ids(frame_tracks)
        dict_to_save['x'] += frame_positions[:, 0].tolist()
        dict_to_save['y'] += frame_positions[:, 1].tolist()
        dict_to_save['team'] += frame_teams
        frame_id += 1
    df = pd.DataFrame(dict_to_save)
    return df
