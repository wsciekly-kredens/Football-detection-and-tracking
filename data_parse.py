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
        raise AttributeError('DataFrame not created yet. Run parse_data_to_df method first.')
