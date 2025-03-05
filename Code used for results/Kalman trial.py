import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Constants
df = pd.read_csv('c:/Users/marko/Desktop/Farmako mafija/Petka rok/k1mM_S4_2024-04-24 12-47-05/detections_0.csv')

ds = 9.5/(int(df['x'].max()) - int(df['x'].min()))
dt = 1/60
print(ds/dt)
distance_threshold = 50

frame_width, frame_height = int(df['x'].max()), int(df['y'].max())
frame_count = df['frame'].max()
output_video = cv2.VideoWriter('detections_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

# Kalman tracking classes
class KalmanFilter:
    def __init__(self):
        self.state = np.zeros(5)  # [x, y, vx, vy, area]

        self.F = np.array([
            [1, 0, dt, 0, 0],
            [0, 1, 0, dt, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])

        self.Q = 0.01 * np.eye(5)

        self.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])

        self.R = np.eye(3) * 1e6
        self.P = np.eye(5) * 1e6

    def predict(self):
        self.state = self.F.dot(self.state)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q


    def update(self, measurement):
        prev_estimate = self.state[:2]
        current_measurement = np.array(measurement[:2])
        distance = np.linalg.norm(current_measurement - prev_estimate) * ds / dt * frame/frame_count
        if distance < 10: w = 0.01
        elif distance < distance_threshold:
            w = 0.05
        elif distance < 2*distance_threshold:
            w = 0.1
        else:
            w = 0.05*distance/distance_threshold
        distance = np.linalg.norm(current_measurement - prev_estimate) * ds**2
        if distance > 0.05: w *= 10
        self.R = np.eye(3) * w

        # Kalman gain
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        # Measurement residual
        z = np.array(measurement)  # Measurement [x, y, area]
        y = z - self.H.dot(self.state)

        # Update state and covariance
        self.state = self.state + K.dot(y)
        self.P = (np.eye(len(self.P)) - K.dot(self.H)).dot(self.P)

        self.state[0] = np.clip(self.state[0], 0, frame_width)
        self.state[1] = np.clip(self.state[1], 0, frame_height)


class MHTTracker:
    def __init__(self, num_objects=10, max_hypotheses=4, collapse_threshold=10):
        self.num_objects = num_objects
        self.max_hypotheses = max_hypotheses
        self.collapse_threshold = collapse_threshold
        self.hypotheses = [{}]
        self.frame_count = 0
        self.trajectories = {i: [] for i in range(num_objects)}
        self.reading = {i: [] for i in range(num_objects)}
        self.filters = [KalmanFilter() for _ in range(num_objects)]
        self.initialized = False

    def initialize_filters(self, detections):
        for i, detection in enumerate(detections):
            if i < self.num_objects:
                self.filters[i].state[0:2] = detection[0:2]
                self.filters[i].state[2:4] = np.zeros(2)
                self.filters[i].state[4] = detection[2]
        for j in range(len(detections), self.num_objects):
            self.filters[j].state[0:2] = np.zeros(2)
            self.filters[j].state[2:4] = np.zeros(2)
            self.filters[j].state[4] = np.mean(detections[:, 2])
        self.initialized = True

    def predict(self):
        for f in self.filters:
            f.predict()

    def associate_detections(self, detections):
        cost_matrix = np.full((self.num_objects, len(detections)), 10e6)

        for i, kalman_filter in enumerate(self.filters):
            predicted_state = kalman_filter.state
            for j, detection in enumerate(detections):
                predicted_pos = predicted_state[0:2]
                detection_pos = detection[0:2]
                distance = np.linalg.norm(predicted_pos - detection_pos) * ds / dt
                cost_matrix[i, j] = 10e6 ** (distance/distance_threshold) if distance < distance_threshold else 10e6

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        detections_for_objects = [None] * self.num_objects
        matched_detections = set()
        costs_for_objects = [10e6] * self.num_objects

        # First pass: accept matches based on linear assignment
        for obj_idx, det_idx in zip(row_ind, col_ind):
            if cost_matrix[obj_idx, det_idx] <= 10e6:
                detections_for_objects[obj_idx] = detections[det_idx]
                costs_for_objects[obj_idx] = cost_matrix[obj_idx, det_idx]
                if detections[det_idx][-1] < self.filters[obj_idx].state[-1] * 1.5:
                    matched_detections.add(det_idx)

        # Second pass: nearest-neighbor matching for unmatched objects
        for obj_idx in range(self.num_objects):
            if detections_for_objects[obj_idx] is None:
                min_distance = 10e6
                closest_detection = None
                closest_det_idx = None

                for det_idx, detection in enumerate(detections):
                    if det_idx not in matched_detections:
                        if cost_matrix[obj_idx, det_idx] <= min_distance:
                            min_distance = cost_matrix[obj_idx, det_idx]
                            closest_detection = detection
                            closest_det_idx = det_idx

                if closest_detection is not None:
                    detections_for_objects[obj_idx] = closest_detection
                    costs_for_objects[obj_idx] = min_distance
                    matched_detections.add(closest_det_idx)

        return detections_for_objects, costs_for_objects

    def generate_hypotheses(self, detections):
        new_hypotheses = []
        if not self.initialized:
            self.initialize_filters(detections)
            return
        #Test robust
        #detections = np.delete(detections, np.random.choice(len(detections)), axis=0)

        detections, costs = self.associate_detections(detections)
        high_cost_indices = [i for i, cost in enumerate(costs) if cost > 1000]
        if high_cost_indices:
            permuted_detections = detections.copy()
            for i in high_cost_indices:
                permuted_detections[i] = [self.filters[i].state[:2], self.filters[i].state[-1]]


        for i in range(self.num_objects):
            if detections[i] is None: continue
            self.filters[i].update(detections[i])
            self.trajectories[i].append(self.filters[i].state[:])
            self.reading[i].append(detections[i])

        # Store the hypothesis after updating
        current_hypothesis = {i: f.state for i, f in enumerate(self.filters)}
        new_hypotheses.append(current_hypothesis)

        if high_cost_indices:
            permuted_hypothesis = {i: (self.filters[i].state if i not in high_cost_indices else permuted_detections[i])
                                   for i in range(self.num_objects)}
            new_hypotheses.append(permuted_hypothesis)
        self.hypotheses = new_hypotheses[:self.max_hypotheses]

    def resolve_hypotheses(self):
        if self.frame_count >= self.collapse_threshold:
            best_hypothesis = max(self.hypotheses, key=self.evaluate_hypothesis)
            self.hypotheses = [best_hypothesis]
            self.frame_count = 0  # Reset frame count after collapse

    # entropy
    def evaluate_hypothesis(self, hypothesis):
        variances = []
        for obj_id, state in hypothesis.items():
            trajectory = np.array(self.trajectories[obj_id])

            if len(trajectory) > 1:
                pos_diffs = np.diff(trajectory, axis=0)
                variance = np.var(pos_diffs, axis=0).sum()
                variances.append(variance)

        total_entropy = sum(variances)
        return 1 / (total_entropy + 1e-6)

    def track(self, detections):
        self.predict()
        self.generate_hypotheses(detections)
        self.resolve_hypotheses()
        self.frame_count += 1
        return self.hypotheses[0]

    def plot_trajectories(self):
        return self.trajectories, self.reading


num_objects = 10  # Assume 10 objects to track
tracker = MHTTracker(num_objects)

# Process each frame and track objects
for frame in df['frame'].unique():
    print(frame)
    detections = df[df['frame'] == frame][['x', 'y', 'area']].to_numpy()
    most_probable_hypothesis = tracker.track(detections)
    #if frame > 900: break
tr, rd = tracker.plot_trajectories()

import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")  # "tab10" provides 10 distinct colors
colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_objects)]


for frame_num in range(frame_count + 1):
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2), (0, 255, 0), 2)
    detections = df[df['frame'] == frame_num][['x', 'y', 'area']].to_numpy()

    # Draw detections in red
    for x, y, area in detections:
        dot_radius = int(np.sqrt(area) / 2)
        color = (0, 0, 255)
        cv2.circle(frame, (int(x), int(y)), dot_radius, color, -1)  # -1 fills the circle

    # Draw trackers in different colors based on their ID
    for obj_id, trajectory in tracker.trajectories.items():
        if frame_num < len(trajectory):
            x, y, _, _, area = trajectory[frame_num]
            tracker_color = colors[obj_id % len(colors)]
            dot_radius = int(np.sqrt(area) / 2)
            cv2.circle(frame, (int(x), int(y)), dot_radius, tracker_color, -1)

    # Display the frame
    cv2.imshow("Detections & Tracking", frame)
    output_video.write(frame)  # Save frame to video

    # Wait for a short period and exit on 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video writer and close windows
output_video.release()
cv2.destroyAllWindows()
