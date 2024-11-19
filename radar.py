import pandas as pd
from scipy.signal import savgol_filter
from mplsoccer import Pitch
import matplotlib.pyplot as plt

class Radar:
    def __init__(self, players_df: pd.DataFrame, smoothing_window: int = 5):
        self.players_df = players_df
        self.smoothing_window = smoothing_window
        self.n_frames = players_df['frame'].nunique()

    def apply_rolling_window(self) -> pd.DataFrame:
        df: pd.DataFrame = self.players_df.copy()
        for track_id in df['track_id'].unique():
            df_one_player = df[df['track_id'] == track_id]
            rolling_x = df_one_player['x'].rolling(window=self.smoothing_window, center=True).mean()
            rolling_y = df_one_player['y'].rolling(window=self.smoothing_window, center=True).mean()
            df.loc[df['track_id'] == track_id, 'x_smooth'] = rolling_x
            df.loc[df['track_id'] == track_id, 'y_smooth'] = rolling_y
        return df

    def apply_savgol_filter(self, df: pd.DataFrame = None) -> pd.DataFrame:
        # savgol = Savitzky-Golay filter
        df: pd.DataFrame = df if df is not None else self.players_df.copy()
        for track_id in df['track_id'].unique():
            df_one_player = df[df['track_id'] == track_id]
            x_smooth = savgol_filter(df_one_player['x'].values, self.smoothing_window, 2)
            y_smooth = savgol_filter(df_one_player['y'].values, self.smoothing_window, 2)
            df.loc[df['track_id'] == track_id, 'x_smooth'] = x_smooth
            df.loc[df['track_id'] == track_id, 'y_smooth'] = y_smooth
        return df

    def draw_radar(self, frame_num: int = 1, axis: bool = True, label: bool = True):
        if frame_num > self.n_frames:
            raise ValueError(f'frame_num greater than number of frames: {frame_num} > {self.n_frames}')
        pitch = Pitch(pitch_type='custom', axis=axis, label=axis, pitch_length=105, pitch_width=68)
        fig, ax = pitch.draw()
        df = self.players_df[self.players_df['frame'] == frame_num]
        pitch.scatter(df.x, df.y, ax=ax, c=df.team, marker='o', edgecolors='black')
        plt.show()
