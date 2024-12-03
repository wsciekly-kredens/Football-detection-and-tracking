import pandas as pd
from scipy.signal import savgol_filter
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import random
from matplotlib import animation
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width


class Radar:
    def __init__(self, players_df: pd.DataFrame, team_colors: tuple, smoothing_window: int = 5):
        self.players_df = players_df
        self.team_colors = self._parse_colors(team_colors)
        self.smoothing_window = smoothing_window
        self.n_frames = players_df['frame'].nunique()
        self.track_id_color = dict()
        self._generate_colors()
        self.savgol_df = self.apply_savgol_filter()

    def _generate_colors(self):
        track_ids = self.players_df['track_id'].unique()
        for track_id in track_ids:
            self.track_id_color[track_id] = (random.random(), random.random(), random.random())

    @staticmethod
    def _parse_colors(colors: tuple) -> tuple:
        color_1 = tuple(map(lambda x: x / 255, colors[0]))
        color_2 = tuple(map(lambda x: x / 255, colors[1]))
        return color_1, color_2

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
            if df_one_player.shape[0] > self.smoothing_window:
                x_smooth = savgol_filter(df_one_player['x'].values, self.smoothing_window, 2)
                y_smooth = savgol_filter(df_one_player['y'].values, self.smoothing_window, 2)
            else:
                x_smooth = df_one_player['x'].values
                y_smooth = df_one_player['y'].values
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

    # TODO dopasować do formatu danych w klasie
    @staticmethod
    def animate_radar(self, player_positions: list, player_teams: list, path: str, fps: int = 30,
                      show: bool = False, ffmpeg_path: str = '') -> None:
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        pitch = Pitch(pitch_type='custom', axis=True, label=True, pitch_length=105, pitch_width=68)
        fig, ax = pitch.draw()
        marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
        away, = ax.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)
        home, = ax.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)

        def animate(i):
            frame_postions = player_positions[i]
            fame_teams = player_teams[i]
            home_positions = frame_postions[fame_teams == np.int32(0)]
            away_positions = frame_postions[fame_teams == np.int32(1)]
            home.set_data(home_positions[:, 0], home_positions[:, 1])
            away.set_data(away_positions[:, 0], away_positions[:, 1])
            return home, away

        ani = animation.FuncAnimation(fig, animate, frames=len(player_positions), interval=100, blit=True)
        if show:
            plt.show()
        ani.save(path, fps=fps)

    def draw_trace(self, n_frame: int, n_frames_back: int = 45, draw: bool = True) -> tuple:
        df = self.savgol_df
        last_frame: int = n_frame - n_frames_back if n_frame - n_frames_back > 0 else 0
        df_frame = df[df['frame'] == n_frame]
        track_ids = df_frame['track_id'].unique()
        df_range = df[df['frame'].isin(range(last_frame, n_frame + 1))]
        color_map = [self.team_colors[team] for team in df_frame.team]

        plt.figure(figsize=(8, 8))
        pitch = Pitch(pitch_type='custom', pitch_length=105, pitch_width=68)
        fig, ax = pitch.draw()
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))
        for track_id in track_ids:
            df_track = df_range[df_range['track_id'] == track_id]
            pitch.plot(df_track.x_smooth, df_track.y_smooth, ax=ax, c=self.track_id_color[track_id], linewidth=4.0,
                       zorder=1)
        pitch.scatter(df_frame.x_smooth, df_frame.y_smooth, ax=ax, c=color_map, marker='o', edgecolors='black',
                      s=220, zorder=2)
        if draw:
            plt.show()
        plt.close()
        return fig, ax
