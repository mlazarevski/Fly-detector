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


class VideoProcessor(QThread):
    finished = pyqtSignal()
    warning = pyqtSignal(str)

    def __init__(self, file_paths, duration=30):
        super().__init__()
        self.duration = duration
        self.file_paths = file_paths

    def frame_cut_and_deflicker(self, video_reader, first_frame_no, duration, x, y, w, h):
        duration = int(duration)
        image_stack = []
        br = 0
        for frame in video_reader:
            # Crop and cut
            if br < first_frame_no:
                br += 1
                continue
            if br == first_frame_no + duration:
                break
            br += 1
            cropped_frame = color.rgb2gray(frame[y:y+h, x:x+w])
            image_stack.append(cropped_frame)
        image_stack = np.array(image_stack)

        # Deflicker
        a = 0.9
        prev_luminance_avg = np.mean(image_stack[0, :, :])
        for br in range(1, duration):
            frame = image_stack[br, :, :]
            prev_frame = image_stack[br - 1, :, :]
            luminance_avg = np.mean(frame)

            if abs(luminance_avg - prev_luminance_avg) > (1 - a) * luminance_avg:
                prev_luminance_avg = luminance_avg
            else:
                prev_luminance_avg = luminance_avg * a + (1 - a) * prev_luminance_avg
                scale = prev_luminance_avg / luminance_avg
                adjusted_frame = np.clip(cropped_frame * scale, 0, 1)

                soft_low = 0.05 * luminance_avg
                soft_high = 0.10 * luminance_avg
                diff = abs(prev_frame - adjusted_frame)
                mask = np.logical_and(diff > soft_low, diff < soft_high)
                final = adjusted_frame.copy()

                final[mask] = (adjusted_frame[mask] * 2 + prev_frame[mask]) / 3
                mask = (diff <= soft_low)
                final[mask] = prev_frame[mask]

                image_stack[br, :, :] = final
        return np.uint8(np.stack(image_stack)*255)

    def region_growing_bfs(self, image, seed_point, threshold):
        height, width = image.shape
        visited = np.zeros_like(image, dtype=bool)
        segmented_image = np.zeros_like(image)
        queue = [(image[seed_point], seed_point)]
        heapq.heapify(queue)
        segmented_image[seed_point] = 255
        visited[seed_point] = True

        while queue:
            _, current_point = heapq.heappop(queue)
            y, x = current_point

            # Explore neighbors in all 8 directions
            neighbors = [(y + dy, x + dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if (dy != 0 or dx != 0)]

            # Create a priority queue for neighbors based on pixel value
            neighbor_queue = [(image[ny, nx], (ny, nx)) for ny, nx in neighbors if
                              0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and (
                                          image[ny, nx] > image[y, x] or image[ny, nx] < threshold)]
            heapq.heapify(neighbor_queue)

            while neighbor_queue:
                _, neighbor_point = heapq.heappop(neighbor_queue)
                ny, nx = neighbor_point
                if not visited[ny, nx]:
                    segmented_image[ny, nx] = 255
                    visited[ny, nx] = True
                    heapq.heappush(queue, (image[ny, nx], (ny, nx)))  # Add to main queue for next depth level

        return segmented_image


    def find_movement(self, image_stack):
        print('putanjica')
        median_frame = np.median(image_stack, axis=0)
        max_frame = np.max(image_stack, axis=0)

        diff_image = max_frame - median_frame
        mask = diff_image > 70
        median_frame[mask] = max_frame[mask]

        blob_data = pd.DataFrame(columns=['x', 'y', 'area', 'eccentricity', 'frame'])

        for c, current_frame in enumerate(image_stack):
            print("Trazim kretanja u ")
            print(c)
            movement_frame = current_frame - median_frame
            movement_frame = 255 - np.abs(np.clip(movement_frame, None, 0))

            # Apply median filter
            movement_frame = median_filter(movement_frame, size=3)

            # Initialize labeled image
            labeled_image = np.zeros_like(movement_frame, dtype=int)
            label = 1
            cnt = 0
            # Perform region growing on the 10 darkest spots
            while cnt < 10:
                cnt = cnt + 1
                if np.min(movement_frame) == 255: break
                min_coords = np.unravel_index(np.argmin(movement_frame), movement_frame.shape)

                segmented_region = self.region_growing_bfs(movement_frame, min_coords, 220)

                num_pixels = np.count_nonzero(segmented_region)
                movement_frame[segmented_region == 255] = 255
                if num_pixels <= 50 or num_pixels >= 2500:
                    cnt = cnt - 1
                    continue

                labeled_image[segmented_region == 255] = label
                label += 1

            regions = measure.regionprops(labeled_image)

            # Collect blob data
            frame_data = pd.DataFrame({
                'x': [region.centroid[1] for region in regions],
                'y': [region.centroid[0] for region in regions],
                'area': [region.area for region in regions],
                'eccentricity': [region.eccentricity for region in regions],
                'frame': c
            })

            # Append the data to the blob_data DataFrame
            blob_data = pd.concat([blob_data, frame_data], ignore_index=True)

        return blob_data

    def process_videos(self):
        selected_files = self.file_paths
        if not selected_files:
            self.warning.emit("No files selected. Aborting.")
            self.finished.emit()
            return

        # Separate CSV and Video files
        csv_files = sorted([file for file in selected_files if file.lower().endswith('.csv')])
        video_files = sorted([file for file in selected_files if file.lower().endswith(('.mp4', '.avi', '.mkv'))])

        if len(csv_files) != len(video_files):
            self.warning.emit("Number of CSV and video files does not match. Aborting.")
            self.finished.emit()
            return

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
            duration = video_reader.get_meta_data()['fps'] * self.duration

            print(f"Processing {csv_file} and {video_file}")

            for index, row in roi_data.iterrows():
                x_roi = round(row['Top Left X'])
                y_roi = round(row['Top Left Y'])
                width_roi = round(row['Bottom Right X'] - x_roi)
                height_roi = round(row['Bottom Right Y'] - y_roi)

                # Process
                image_stack = self.frame_cut_and_deflicker(video_reader, first_frame_no, duration, x_roi, y_roi, width_roi, height_roi)
                blob_data = self.find_movement(image_stack)

                filename = os.path.join(output_dir, f"detections_{index}.csv")
                blob_data.to_csv(filename, index=False)

        print("Processing completed.")
        self.finished.emit()

    def run(self):
        self.process_videos()

