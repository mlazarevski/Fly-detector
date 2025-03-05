import imageio
import skimage
from skimage import color, filters, measure
import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import os
import sys
import heapq
from skimage import morphology
from scipy.ndimage import median_filter
import copy

def frame_cut(video_reader, first_frame_no, duration, x, y, w, h):
    image_stack = []
    for br, frame in enumerate(video_reader):
        if br < first_frame_no:
            continue
        if br == first_frame_no + duration:
             break
        cropped_frame = color.rgb2gray(frame[y:y+h, x:x+w]) * 255
        image_stack.append(cropped_frame.astype(np.uint8))
    return np.array(image_stack)

def region_growing_bfs(image, height, width, seed_point, threshold):
    visited = np.zeros_like(image, dtype=bool)
    segmented_image = np.zeros_like(image)
    queue = [(image[seed_point], seed_point)]
    heapq.heapify(queue)
    segmented_image[seed_point] = 255
    visited[seed_point] = True

    while queue:
        _, current_point = heapq.heappop(queue)
        y, x = current_point

        neighbors = [(y + dy, x + dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if (dy != 0 or dx != 0)]

        neighbor_queue = [(image[ny, nx], (ny, nx)) for ny, nx in neighbors if
                            0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and (
                            image[ny, nx] > image[y, x] or image[ny, nx] <= threshold)]
        heapq.heapify(neighbor_queue)

        while neighbor_queue:
            _, neighbor_point = heapq.heappop(neighbor_queue)
            ny, nx = neighbor_point
            if not visited[ny, nx]:
                segmented_image[ny, nx] = 255
                visited[ny, nx] = True
                heapq.heappush(queue, (image[ny, nx], (ny, nx)))  # Add to main queue for next depth level

        return segmented_image

def find_movement(image_stack):
    median_frame = np.median(image_stack, axis=0)
    max_frame = np.max(image_stack, axis=0)

    diff_image = max_frame - median_frame
    mask = diff_image > 70
    background = copy.deepcopy(median_frame)
    background[mask] = max_frame[mask]
    height, width = background.shape

    blob_data = pd.DataFrame(columns=['x', 'y', 'area', 'eccentricity', 'frame'])

    for c, current_frame in enumerate(image_stack):
        movement_frame = current_frame - background
        movement_frame = 255 - np.abs(np.clip(movement_frame, None, 0))
        movement_frame = median_filter(movement_frame, size=3)

        min_val = np.min(movement_frame)
        max_val = np.max(movement_frame)
        movement_frame = (movement_frame - min_val) / (max_val - min_val) * 255

        step = 5
        for y in range(0, height, step):
            strip = movement_frame[y:y + step, :]
            if strip.size == 0: continue
            threshold = np.percentile(strip, 5)
            saturated_strip = np.where(strip >= threshold, 255, strip)

            movement_frame[y:y + step, :] = saturated_strip

        threshold = np.percentile(movement_frame, 1)
        movement_frame = np.where(movement_frame > threshold, 255, movement_frame)

        labeled_image = np.zeros_like(movement_frame, dtype=int)
        label = 1
        cnt = 0
        while cnt < 10:
            cnt = cnt + 1
            if np.min(movement_frame) >= threshold: break
            min_coords = np.unravel_index(np.argmin(movement_frame), movement_frame.shape)

            segmented_region = region_growing_bfs(movement_frame, height, width, min_coords, threshold)

            num_pixels = np.count_nonzero(segmented_region)
            movement_frame[segmented_region == 255] = 255
            if num_pixels <= 50 or num_pixels >= 2500:
                cnt = cnt - 1
                continue

            labeled_image[segmented_region == 255] = label
            label += 1

        regions = measure.regionprops(labeled_image)

        frame_data = pd.DataFrame({
            'x': [region.centroid[1] for region in regions],
            'y': [region.centroid[0] for region in regions],
            'area': [region.area for region in regions],
            'eccentricity': [region.eccentricity for region in regions],
            'frame': c
        })

        blob_data = pd.concat([blob_data, frame_data], ignore_index=True)

    return blob_data

file_name = "Nova tura Marko Medicinski"
selected_files = []

for root, dirs, files in os.walk(file_name):
    for file in files:
        # Add full file path
        selected_files.append(os.path.join(root, file))


csv_files = sorted([file for file in selected_files if file.lower().endswith('.csv')])
video_files = sorted([file for file in selected_files if file.lower().endswith(('.mp4', '.avi', '.mkv'))])


for csv_file, video_file in zip(csv_files, video_files):
    # Create a directory with the same name as the CSV file (without extension)
    folder_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = os.path.join("deadline", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    roi_data = pd.read_csv(csv_file)
    first_frame_no = roi_data['First frame'][0]

    # Read video
    video_reader = imageio.get_reader(video_file)
    duration = video_reader.get_meta_data()['fps'] * 30

    print(f"Processing {csv_file} and {video_file}")

    for index, row in roi_data.iterrows():
        x_roi = round(row['Top Left X'])
        y_roi = round(row['Top Left Y'])
        width_roi = round(row['Bottom Right X'] - x_roi)
        height_roi = round(row['Bottom Right Y'] - y_roi)

        # Process
        image_stack = frame_cut(video_reader, first_frame_no, duration, x_roi, y_roi, width_roi, height_roi)
        blob_data = find_movement(image_stack)

        filename = os.path.join(output_dir, f"detections_{index}.csv")
        blob_data.to_csv(filename, index=False)



