import tkinter as tk
from tkinter import filedialog
import pandas as pd
import math
import os
from sklearn.mixture import GaussianMixture
import numpy as np

def select_files():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder")
    file_paths = []

    for root_dir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("df.csv"):
                file_paths.append(os.path.join(root_dir, file))

    return file_paths

def process_csv(file_path):
    tr = pd.read_csv(file_path)
    '''
    first_seens = tr.groupby('particle')['frame'].min().sort_values(ascending=False)
    last_seens = tr.groupby('particle')['frame'].max().sort_values(ascending=False)
    for tail_idx, tail in first_seens.items():
        for head_idx, head in last_seens.items():
            if head < tail:
                frames_tail = tr[tr['particle'] == tail_idx]['frame'].unique()
                frames_head = tr[tr['particle'] == head_idx]['frame'].unique()
                frames_intersect = set(frames_tail) & set(frames_head)
                if not frames_intersect:
                    tr.loc[tr['particle'] == tail_idx, 'particle'] = head_idx
                    break
    '''
    return tr


def calculate_distance(segment):
    x_diff = segment['x'].iloc[-1] - segment['x'].iloc[0]
    y_diff = segment['y'].iloc[-1] - segment['y'].iloc[0]
    distance = (x_diff ** 2 + y_diff ** 2) ** 0.5
    return distance


def calculate_climb(segment):
    y_diff = segment['y'].iloc[0] - segment['y'].iloc[-1]
    if y_diff < 0:
        return -y_diff
    else:
        return 0


def get_stats(longest_trajectories, max_frame_number=1799):
    t = longest_trajectories[['x', 'y', 'frame', 'particle']]
    view = 10;
    dt = 1 / 60;
    ds = 9.5 / (longest_trajectories['y'].max() + longest_trajectories['y'].min())

    # Change if needed
    height_line = (longest_trajectories['y'].max() + longest_trajectories['y'].min()) * (1.5 / 9.5)
    average_velocities = {}
    velocity_per_segment = {}
    time_to_top = {}

    for particle_id, group in t.groupby('particle'):
        # Fill the holes in detection
        full_frame_particle = pd.DataFrame(
            [(frame, particle_id) for frame in range(0, max_frame_number + 1)],
            columns=['frame', 'particle'])

        group = pd.merge(full_frame_particle, group, on=['frame', 'particle'], how='left')
        group['x'].ffill(inplace=True)
        group['y'].bfill(inplace=True)

        # Frame u kome je dodjeno do vrha
        min_frame = group[(group['y'] < height_line) & (group['frame'] >= 20)]['frame'].min()
        # Kada se gledaju samo one koje su stigle do vrha odkomentarisati
        if math.isnan(min_frame):
            min_frame = group[(group['y'] < height_line)]['frame'].min()
            if math.isnan(min_frame):
                min_frame = max_frame_number

        group = group[group['frame'] <= min_frame]
        time_to_top[particle_id] = min_frame * dt
        velocity_per_segment[particle_id] = {}
        average_velocities[particle_id] = 0
        if min_frame <= 20:
            average_velocities[particle_id] = -1
            velocity_per_segment[particle_id][0] = -1
            continue
        # Distance MA, no overlap
        segments = [group.iloc[i - view:i] for i in range(view, min_frame, view)]
        for i, segment in enumerate(segments):
            l = calculate_climb(segment) * ds / (view * dt)
            if not isinstance(l, float) or math.isnan(l): l = 0
            velocity_per_segment[particle_id][i] = l
            if velocity_per_segment[particle_id][i] > 3: velocity_per_segment[particle_id][i] = 1
            average_velocities[particle_id] += velocity_per_segment[particle_id][i]
        average_velocities[particle_id] /= (np.floor(min_frame/view))
        print(average_velocities[particle_id])

    velocity_per_segment = pd.DataFrame(velocity_per_segment).fillna(0)

    max_velocities = {}
    for particle_id, group in t.groupby('particle'):
        min_frame = group[(group['y'] < height_line) & (group['frame'] >= 20)]['frame'].min()
        if math.isnan(min_frame):
            min_frame = group[(group['y'] < height_line)]['frame'].min()
            if math.isnan(min_frame):
                min_frame = max_frame_number
        if min_frame < 20:
            max_distance = -1
        else:
            try:
                max_distance = max([velocity_per_segment[particle_id][i] for i in velocity_per_segment.T])
            except:
                max_distance = 0
        max_velocities[particle_id] = max_distance

    max_velocities_df = pd.DataFrame(max_velocities.items(), columns=['particle', 'max_velocity'])
    average_velocities_df = pd.DataFrame(average_velocities.items(), columns=['particle', 'average_velocity'])
    time_to_top_df = pd.DataFrame(time_to_top.items(), columns=['particle', 'time_to_top'])
    # Merge DataFrames
    result_df = pd.merge(max_velocities_df, average_velocities_df, on='particle')
    result_df = pd.merge(result_df, time_to_top_df, on='particle')

    return result_df


def main():
    file_paths = select_files()
    if file_paths:
        for file_path in file_paths:
            print("Processing:", file_path)
            longest_trajectories = process_csv(file_path)
            #Count crosses
            stats = get_stats(longest_trajectories, max_frame_number=1799)
            directory, filename = os.path.split(file_path)
            name, extension = os.path.splitext(filename)
            new_filename = f"{name}_stats.csv"
            new_file_path = os.path.join(directory, new_filename)
            stats.to_csv(new_file_path, index=False)



    else:
        print("No files selected.")


if __name__ == "__main__":
    main()
