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
            if file.endswith("tracks.csv"):
                file_paths.append(os.path.join(root_dir, file))

    return file_paths

def process_csv(file_path):
    tr = pd.read_csv(file_path)

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
    return tr


def calculate_distance(segment):
    x_diff = segment['x'].iloc[-1] - segment['x'].iloc[0]
    y_diff = segment['y'].iloc[-1] - segment['y'].iloc[0]
    distance = (x_diff ** 2 + y_diff ** 2) ** 0.5
    return distance


def calculate_climb(segment):
    y_diff = segment['y'].iloc[0] - segment['y'].iloc[-1]
    if y_diff > 0:
        return y_diff
    else:
        return 0


def get_stats(longest_trajectories, max_frame_number=1799):
    t = longest_trajectories[['x', 'y', 'frame', 'particle']]
    jump = 10;
    view = 50;
    dt = 1 / 60;
    ds = 9.5 / (longest_trajectories['y'].max() + longest_trajectories['y'].min())

    # Change if needed
    height_line = (longest_trajectories['y'].max() + longest_trajectories['y'].min()) * 0.5#(1.5 / 9.5)
    average_velocities = {}
    summed_distances = {}
    distance_per_segment = {}
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
        min_frame = group[group['y'] < height_line]['frame'].min()

        # Kada se gledaju samo one koje su stigle do vrha odkomentarisati
        if math.isnan(min_frame): min_frame = max_frame_number
        if min_frame < 50: continue
        group = group[group['frame'] <= min_frame]
        time_to_top[particle_id] = min_frame

        distance_per_segment[particle_id] = {}
        summed_distances[particle_id] = 0

        # Speed MA
        segments = [group.iloc[i - view:i] for i in range(view, min_frame, jump)]
        for i, segment in enumerate(segments):
            distance_per_segment[particle_id][i] = calculate_climb(segment)#calculate_distance(segment)

        # Distance MA, no overlap
        segments = [group.iloc[i - jump:i] for i in range(jump, min_frame, jump)]
        for i, segment in enumerate(segments):
            l = calculate_climb(segment)#calculate_distance(segment)
            if isinstance(l, float) and not math.isnan(l):
                summed_distances[particle_id] += l * ds
        average_velocities[particle_id] = summed_distances[particle_id] / (dt * (max_frame_number + 1))

    distance_per_segment = pd.DataFrame(distance_per_segment).fillna(0)

    max_distances = {}
    max_velocities = {}
    for particle_id in t['particle'].unique():
        min_frame = group[group['y'] < height_line]['frame'].min()
        if min_frame < 50: continue
        try:
            max_distance = max([distance_per_segment[particle_id][i] for i in distance_per_segment.T])
        except:
            max_distance = 0
        max_distances[particle_id] = max_distance * ds
        max_velocities[particle_id] = max_distances[particle_id] / (dt * view)

    max_distances_df = pd.DataFrame(max_distances.items(), columns=['particle', 'max_distance'])
    max_velocities_df = pd.DataFrame(max_velocities.items(), columns=['particle', 'max_velocity'])
    summed_distances_df = pd.DataFrame(summed_distances.items(), columns=['particle', 'summed_distance'])
    average_velocities_df = pd.DataFrame(average_velocities.items(), columns=['particle', 'average_velocity'])
    time_to_top_df = pd.DataFrame(time_to_top.items(), columns=['particle', 'time_to_top'])

    # Merge DataFrames
    result_df = pd.merge(max_distances_df, max_velocities_df, on='particle')
    result_df = pd.merge(result_df, summed_distances_df, on='particle')
    result_df = pd.merge(result_df, average_velocities_df, on='particle')
    result_df = pd.merge(result_df, time_to_top_df, on='particle')

    sorted_tr = result_df.sort_values(by='summed_distance', ascending=False)
    n = min(10, result_df['particle'].nunique())
    longest_particle_ids = sorted_tr['particle'].head(n).tolist()
    longest_trajectories = result_df[result_df['particle'].isin(longest_particle_ids)]

    return longest_trajectories


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
