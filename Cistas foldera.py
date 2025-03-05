import os
import glob
from PyQt5.QtWidgets import QApplication, QFileDialog

def delete_non_track_stats_csv(folder_path):
    # Traverse all subfolders and delete csv files that don't match "....track_stats.csv"
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv") and not file.endswith("tracks_stats.csv"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

def select_folder_and_clean_csv():
    app = QApplication([])
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
    if folder_path:
        delete_non_track_stats_csv(folder_path)
    else:
        print("No folder selected!")

if __name__ == "__main__":
    select_folder_and_clean_csv()
