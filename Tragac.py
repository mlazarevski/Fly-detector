import sys
import os
import numpy as np
import pandas as pd
import trackpy as tp
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.mixture import GaussianMixture


class ParticleTracker(QThread):
    finished = pyqtSignal()
    warning = pyqtSignal(str)

    def __init__(self, file_paths, fly_count=10):
        super().__init__()
        self.fly_count = fly_count
        self.file_paths = file_paths

    def process_detections(self, csv_files):
        for csv_file in csv_files:
            try:
                all_tracks = pd.read_csv(csv_file)
                all_tracks = all_tracks[(all_tracks['area'] >= 30) &
                                             (all_tracks['area'] <= 150)]
                all_tracks = all_tracks[(all_tracks['eccentricity'] >= 0.5) &
                                             (all_tracks['eccentricity'] <= 1)]
                # Fit Gaussian Mixture Models
                gmm_ecc = GaussianMixture(n_components=2, covariance_type='full').fit(all_tracks[['eccentricity']])
                gmm_area = GaussianMixture(n_components=1, covariance_type='full').fit(all_tracks[['area']])

                # Calculate detection probabilities
                all_tracks['prob_area'] = np.exp(gmm_area.score_samples(all_tracks['area'].values.reshape(-1, 1)))
                all_tracks['prob_ecc'] = np.exp(gmm_ecc.score_samples(all_tracks['eccentricity'].values.reshape(-1, 1)))
                all_tracks['probability'] = all_tracks['prob_area'] * all_tracks['prob_ecc']
                # Select top detections
                top_detections = all_tracks.groupby('frame').apply(lambda x: x.nlargest(self.fly_count, 'probability')).reset_index(drop=True)
                # Link particles across frames
                tr = tp.link_df(top_detections.iloc[:, 0:5], search_range=30, memory=300)#, adaptive_stop=20, adaptive_step=20)
                #tr = tp.link_df(all_tracks.iloc[:, 0:5], search_range=120, memory=100)#, adaptive_stop=20, adaptive_step=20)

                # Compile results
                results = []
                for particle_id, particle_track in tr.groupby('particle'):
                    distances = np.sqrt(np.diff(particle_track['x']) ** 2 + np.diff(particle_track['y']) ** 2)
                    total_distance = np.sum(distances)

                    particle_results = {
                        'Particle_ID': particle_id,
                        'Track_Length': len(particle_track),
                        'First_Seen': particle_track['frame'].min(),
                        'Last_Seen': particle_track['frame'].max(),
                        'First_X': particle_track.loc[particle_track['frame'].idxmin(), 'x'],
                        'First_Y': particle_track.loc[particle_track['frame'].idxmin(), 'y'],
                        'Last_X': particle_track.loc[particle_track['frame'].idxmax(), 'x'],
                        'Last_Y': particle_track.loc[particle_track['frame'].idxmax(), 'y'],
                        'Max_X': particle_track['x'].max(),
                        'Min_X': particle_track['x'].min(),
                        'Max_Y': particle_track['y'].max(),
                        'Min_Y': particle_track['y'].min(),
                        'Area': particle_track['area'].median(),
                        'Eccentricity': particle_track['eccentricity'].median(),
                        'Total_Distance': total_distance
                    }
                    results.append(particle_results)

                results_df = pd.DataFrame(results)

                # Save results to CSV
                output_csv_stats = os.path.splitext(csv_file)[0] + "_statistics.csv"
                results_df.to_csv(output_csv_stats, index=False)

                print(f"Particle properties saved as {output_csv_stats}")

                output_csv_tracks = os.path.splitext(csv_file)[0] + "_tracks.csv"
                tr.to_csv(output_csv_tracks, index=False)
            except Exception as e:
                error_message = f"An error occurred while processing {csv_file}: {str(e)}"
                self.warning.emit(error_message)
                print(error_message)

        self.finished.emit()

    def run(self):
        self.process_detections(self.file_paths)
