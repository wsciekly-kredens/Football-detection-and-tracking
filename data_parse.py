import pandas as pd
import json
import os
from tqdm import tqdm
import time

class DataParser:
    
    def __init__(self, directory: str):
        self.directory = directory
        self._data = self._load_data()

    def _load_data(self) -> list:
        data = list()
        print('Wczytywanie danych...')
        for folder_name in tqdm(os.listdir(self.directory)):
            folder_full_path = os.path.join(self.directory, folder_name)
            if os.path.isdir(folder_full_path):
                with open(os.path.join(self.directory, folder_name, 'Labels-GameState.json')) as f:
                    d = json.load(f)
                    data.append(d['annotations'])
        return data
    
    
    def parse_data_to_df(self) -> pd.DataFrame:
        start = time.time()
        print('Tworzenie DataFrame...')
        flat_dict = dict()
        for annotations in tqdm(self._data):
            data_image = filter(lambda x: x['supercategory'] == 'object' 
                                          and x['bbox_pitch_raw'] is not None, annotations)
            for img in data_image:
                for key in img:
                    if isinstance(img[key], dict):
                        for key2 in img[key]:
                            if key + '_' + key2 not in flat_dict:
                                flat_dict[key + '_' + key2] = []
                            flat_dict[key + '_' + key2].append(img[key][key2])
                    else:
                        if key not in flat_dict:
                            flat_dict[key] = []
                        flat_dict[key].append(img[key])
        df = pd.DataFrame(flat_dict, index=flat_dict['id'])
        df = df.drop(columns=['id'])
        self._parsed_df = df
        return df

    def _flatten_data(self) -> list:
        flat_list = [y for x in self._data for y in x]
        return flat_list

    def get_pitch_data_df(self) -> pd.DataFrame:
        data = self._flatten_data()
        pitch_data_dict = {'id': [], 'image_id': [], 'line_type': [], 'x': [], 'y': []}
        allowed_lines = ['Circle central', 'Big rect. left bottom', 'Big rect. left main', 'Big rect. left top',
                         'Small rect. left bottom', 'Small rect. left main', 'Small rect. left top',
                         'Big rect. right bottom', 'Big rect. right main', 'Big rect. right top',
                         'Small rect. right bottom', 'Small rect. right main', 'Small rect. right top', 'Middle line',
                         'Side line bottom', 'Side line left', 'Side line right', 'Side line top']
        data = filter(lambda x: x['supercategory'] == 'pitch', data)
        for anotation in tqdm(data):
            lines = anotation['lines']
            lines = {k: v for k, v in lines.items() if k in allowed_lines}
            for line in lines:
                for data_point in anotation['lines'][line]:
                    pitch_data_dict['id'].append(int(anotation['id']))
                    pitch_data_dict['image_id'].append(int(anotation['image_id']))
                    pitch_data_dict['line_type'].append(line)
                    pitch_data_dict['x'].append(data_point['x'])
                    pitch_data_dict['y'].append(data_point['y'])
        pitch_df = pd.DataFrame(pitch_data_dict)
        return pitch_df

    def get_shirts_data(self, image_id: str = None) -> pd.DataFrame:
        if not hasattr(self, '_parsed_df'):
            raise AttributeError('DataFrame not created yet. Run parse_data_to_df method first.')

        if image_id is not None:
            df: pd.DataFrame =  self._parsed_df[self._parsed_df['image_id'] == image_id]
        else:
            df: pd.DataFrame = self._parsed_df
        df = df[df['attributes_role'] == 'player']
        df = df[['image_id', 'bbox_image_x', 'bbox_image_y', 'bbox_image_w', 'bbox_image_h', 'attributes_team']]

        return df


    def get_data(self) -> list:
        return self._data

    def get_parsed_df(self) -> pd.DataFrame:
        if hasattr(self, '_parsed_df'):
            return self._parsed_df
        else:
            self.parse_data_to_df()
            return self._parsed_df
