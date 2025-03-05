import imageio
import skimage
from skimage import color, io, filters, measure, feature, draw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import trackpy as tp
import pandas as pd
import numpy as np
from scipy.ndimage import convolve
import tkinter as tk
from tkinter import filedialog
import os


def frame_cut_and_deflicker(video_reader, first_frame_no, duration, x, y, w, h):
    image_stack = []
    br = 0
    a = 0.9

    for frame in video_reader:

        # Crop and cut
        if br < first_frame_no:
            br += 1
            continue
        if br == first_frame_no + duration:
            break
        br += 1
        cropped_frame = color.rgb2gray(frame[y:y+h, x:x+w])

        # Deflicker
        luminance_avg = np.mean(cropped_frame)

        if len(image_stack) > 0:
            prev_frame = image_stack[-1]
            if abs(luminance_avg - prev_luminance_avg) > (1 - a) * luminance_avg:
                prev_luminance_avg = luminance_avg
            else:
                prev_luminance_avg = luminance_avg * a + (1 - a) * prev_luminance_avg
                scale = prev_luminance_avg / luminance_avg
                adjusted_frame = np.clip(cropped_frame * scale, 0, 1)

                soft_low  = 0.05 * luminance_avg
                soft_high = 0.10 * luminance_avg
                diff = abs(prev_frame - adjusted_frame)
                mask = np.logical_and(diff > soft_low, diff < soft_high)
                final = adjusted_frame.copy()

                final[mask] = (adjusted_frame[mask] * 2 + prev_frame[mask]) / 3

                mask = (diff <= soft_low)
                final[mask] = prev_frame[mask]

                cropped_frame = final.copy()
        else:
            prev_luminance_avg = luminance_avg

        image_stack.append(cropped_frame)

    return np.stack(image_stack)


def find_movement(image_stack):

    median_frame = np.median(image_stack, axis=0)
    max_frame = np.max(image_stack, axis=0)
    mask = median_frame < np.percentile(median_frame, 5)
    median_frame = np.where(mask, max_frame, median_frame)

    blob_data = pd.DataFrame(columns=['x', 'y', 'area', 'eccentricity', 'frame'])

    for c, current_frame in enumerate(image_stack):
        movement_frame = current_frame - median_frame
        movement_frame = 1 - np.abs(np.clip(movement_frame, None, 0))

        binMovement = skimage.filters.median(movement_frame, mode='nearest', behavior='ndimage')
        binMovement = binMovement > filters.threshold_otsu(binMovement)

        labeled_blobs = measure.label(binMovement, background=1, connectivity=2)
        regions = measure.regionprops(labeled_blobs)

        frame_data = pd.DataFrame({
            'x': [region.centroid[1] for region in regions],
            'y': [region.centroid[0] for region in regions],
            'area': [region.area for region in regions],
            'eccentricity': [region.eccentricity for region in regions],
            'frame': c})
        blob_data = pd.concat([blob_data, frame_data], ignore_index=True)

    return blob_data


def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title="Select Files")
    return file_paths


duration = 30
# Start of the selection
selected_files = select_files()
if not selected_files:
    print("No files selected. Aborting.")
    exit()

# Separate CSV and Video files
csv_files = sorted([file for file in selected_files if file.lower().endswith('.csv')])
video_files = sorted([file for file in selected_files if file.lower().endswith(('.mp4', '.avi', '.mkv'))])

if len(csv_files) != len(video_files):
    print("Number of CSV and video files does not match. Aborting.")
    exit()

for csv_file, video_file in zip(csv_files, video_files):
    # Create a directory with the same name as the CSV file (without extension)
    folder_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = os.path.join("image_stacks", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    roi_data = pd.read_csv(csv_file)
    first_frame_no = roi_data['First frame'][0]
    fly_count = roi_data['Fly Number'][0]
    if np.isnan(fly_count):
        fly_count = 10

    # Read video
    video_reader = imageio.get_reader(video_file)
    first_frame = video_reader.get_data(first_frame_no)
    duration = video_reader.get_meta_data()['fps'] * 30

    print(f"Processing {csv_file} and {video_file}")

    for index, row in roi_data.iterrows():
        x_roi = round(row['Top Left X'])
        y_roi = round(row['Top Left Y'])
        width_roi = round(row['Bottom Right X'] - x_roi)
        height_roi = round(row['Bottom Right Y'] - y_roi)

        # Process
        image_stack = frame_cut_and_deflicker(video_reader, first_frame_no, duration, x_roi, y_roi, width_roi, height_roi)
        blob_data = find_movement(image_stack)

        filename = os.path.join(output_dir, f"vial_{index}.csv")
        blob_data.to_csv(filename, index=False)

print("Processing completed.")




