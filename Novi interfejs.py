import sys
import cv2
import numpy as np
import csv
import subprocess
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QMenuBar, QAction, QSlider, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QColor
from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer
from video_processor import VideoProcessor
from novi_tragac import ParticleTracker


class ImageMarkerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Detector")

        # Get the screen's dimensions
        screen_rect = QApplication.desktop().screenGeometry()
        screen_height = screen_rect.height()
        screen_width = screen_rect.width()

        # Calculate desired dimensions
        desired_height = int(0.75 * screen_height)
        desired_width = int(0.75 * screen_width)

        # Set the geometry of the window
        self.setGeometry((screen_width-desired_width)//2, (screen_height-desired_height)//2,
                         desired_width, desired_height)

        self.image = None
        self.rect_start = None
        self.rect_end = None
        self.rectangles = []
        self.selected_rectangle = None
        self.resizing = False
        self.scale_factor = 1
        self.video_capture = None
        self.total_frames = None
        self.frame_begin = 1
        self.frame_end = 1
        self.isTest = False
        self.file_path = None
        self.candidates = None
        self.video_processor = None
        self.tracker = None

        # Initialize the timer
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(10)  # ~30 FPS (1/30th of a second)
        self.update_timer.timeout.connect(self.refresh_frame)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.graphics_view = QGraphicsView(self)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setStyleSheet("background-color: gray;")
        layout.addWidget(self.graphics_view)

        # Horizontal layout for frame rate and scale
        info_layout = QHBoxLayout()
        self.frame_rate_label = QLabel("Frame Rate: N/A")
        info_layout.addWidget(self.frame_rate_label)
        self.scale_label = QLabel("Scale: N/A")
        info_layout.addWidget(self.scale_label)
        self.current_frame_label = QLabel("Frame Number: N/A")
        info_layout.addWidget(self.current_frame_label)
        self.first_frame_label = QLabel("First Frame: N/A")
        info_layout.addWidget(self.first_frame_label)
        self.last_frame_label = QLabel("Last Frame: N/A")
        info_layout.addWidget(self.last_frame_label)
        layout.addLayout(info_layout)

        controls_layout = QVBoxLayout()
        layout.addLayout(controls_layout)

        open_button = QPushButton("Import Video")
        open_button.clicked.connect(self.open_image)
        controls_layout.addWidget(open_button)

        # Create a horizontal layout for the fly number and numbering
        fly_no_layout = QHBoxLayout()
        fly_no_label = QLabel("Number of flies per test:")
        fly_no_layout.addWidget(fly_no_label)
        self.fly_no_entry = QLineEdit()
        fly_no_layout.addWidget(self.fly_no_entry)
        self.show_number_checkbox = QCheckBox("Show Numbers")
        self.show_number_checkbox.stateChanged.connect(self.change_numbering)
        fly_no_layout.addWidget(self.show_number_checkbox)

        layout.addLayout(fly_no_layout)
        #controls_layout.addLayout(fly_no_layout)

        frame_slider_layout = QHBoxLayout()
        frame_slider_label = QLabel("Current Frame:")
        frame_slider_layout.addWidget(frame_slider_label)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.valueChanged.connect(self.on_slider_moved)
        frame_slider_layout.addWidget(self.frame_slider)
        controls_layout.addLayout(frame_slider_layout)

        first_frame_button = QPushButton("First")
        first_frame_button.clicked.connect(self.set_first)
        controls_layout.addWidget(first_frame_button)

        last_frame_button = QPushButton("Undo")
        last_frame_button.clicked.connect(self.undo_rectangle)
        controls_layout.addWidget(last_frame_button)

        mark_button = QPushButton("Mark")
        mark_button.clicked.connect(self.process)
        controls_layout.addWidget(mark_button)

        # Menu bar
        menu_bar = self.menuBar()
        options_menu = menu_bar.addMenu("Options")
        help_menu = menu_bar.addMenu("Help")
        load_menu = menu_bar.addMenu("Load")

        test_action = QAction("Test", self, checkable=True)
        test_action.triggered.connect(self.toggle_test)
        options_menu.addAction(test_action)

        track_action = QAction("Track", self)
        track_action.triggered.connect(self.track)
        options_menu.addAction(track_action)

        count_action = QAction("Count", self)
        count_action.triggered.connect(self.decide)
        options_menu.addAction(count_action)

        clear_action = QAction("Clear Rectangles", self)
        clear_action.triggered.connect(self.clear_rectangles)
        options_menu.addAction(clear_action)

        # Load menu action
        load_action = QAction("Load markings", self)
        load_action.triggered.connect(self.load_rectangles_from_csv)
        load_menu.addAction(load_action)
        # Load tracking
        load_action = QAction("Load detections", self)
        load_action.triggered.connect(self.load_detections_from_csv)
        load_menu.addAction(load_action)

        # Help menu action
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        self.graphics_view.setMouseTracking(True)
        self.graphics_view.mousePressEvent = self.on_mouse_press
        self.graphics_view.mouseMoveEvent = self.on_mouse_drag
        self.graphics_view.mouseReleaseEvent = self.on_mouse_release

        self.setLayout(layout)

    # Open video, count frames, set first and last, display first frame
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.file_path = file_path
            self.video_capture = cv2.VideoCapture(file_path)
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_end = self.total_frames
            ret, frame = self.video_capture.read()
            if ret:
                # Set slider attributes
                self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.frame_slider.setMaximum(self.total_frames)
                # Display image
                self.display_image(frame)

    def update_info_labels(self):
        if self.video_capture:
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.frame_rate_label.setText(f"Frame Rate: {int(fps)} FPS")
            self.first_frame_label.setText(f"First Frame: {self.frame_begin}")
            self.last_frame_label.setText(f"Last Frame: {self.total_frames}")
        else:
            self.frame_rate_label.setText("Frame Rate: N/A")
            self.first_frame_label.setText("First Frame: N/A")
            self.last_frame_label.setText("Last Frame: N/A")
        self.scale_label.setText(f"Scale: {self.scale_factor:.2f}")
        self.current_frame_label.setText(f"Frame: {self.frame_slider.value()}")

    def display_image(self, frame):
        self.image = frame
        height, width, _ = frame.shape
        view_height = self.graphics_view.height()
        view_width = self.graphics_view.width()
        self.scale_factor = min(view_width / width, view_height / height)

        # Resize and draw
        resized_image = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_AREA)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        try:
            for i, rect in enumerate(self.rectangles):
                # Scale rectangle coordinates
                scaled_rect = QRectF(
                    rect.left() * self.scale_factor,
                    rect.top() * self.scale_factor,
                    rect.width() * self.scale_factor,
                    rect.height() * self.scale_factor
                )
                # Convert QRectF to integer coordinates for OpenCV
                top_left = (int(scaled_rect.left()), int(scaled_rect.top()))
                bottom_right = (int(scaled_rect.right()), int(scaled_rect.bottom()))
                # Draw the rectangle on the image
                cv2.rectangle(resized_image, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle with thickness 2
                if self.show_number_checkbox.isChecked():
                    cv2.putText(resized_image, f"Vial {i + 1}", (top_left[0], top_left[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if self.candidates is not None:
                current_frame = int(self.frame_slider.value())
                frame_candidates = self.candidates[self.candidates['frame'] == current_frame]
                for _, candidate in frame_candidates.iterrows():
                    x = int(candidate['x'] * self.scale_factor)
                    y = int(candidate['y'] * self.scale_factor)
                    area = candidate['area']
                    radius = int((area / np.pi) ** 0.5)  # Assuming area = πr², so r = √(area/π)
                    cv2.circle(resized_image, (x, y), radius, (255, 0, 0), -1)  # Red dot with filled circle

        except Exception as e:
            print(f"Error while drawing rectangles: {e}")
        # Show the final image
        q_image = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0], resized_image.strides[0],
                         QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.graphics_scene.clear()
        pixmap_item = self.graphics_scene.addPixmap(pixmap)
        self.graphics_scene.setSceneRect(QRectF(pixmap_item.pixmap().rect()))

        # Info bar update
        self.update_info_labels()

    def set_first(self):
        self.frame_begin = int(self.frame_slider.value())
        self.update_info_labels()

    def on_slider_moved(self):
        self.update_timer.start()

    def refresh_frame(self):
        try:
            frame_number = int(self.frame_slider.value())
            if 1 <= frame_number <= self.total_frames:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                ret, frame = self.video_capture.read()
                if ret:
                    self.display_image(frame)
            else:
                QMessageBox.warning(self, "Error", f"Invalid frame number. Please enter a number between 1 and {self.total_frames}.")
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid frame number. Please enter a valid integer.")
        finally:
            self.update_timer.stop()

    def on_mouse_press(self, event):
        pos = self.graphics_view.mapToScene(event.pos())
        x, y = pos.x(), pos.y()
        self.rect_start = (x, y)
        self.rect_end = None

    def on_mouse_drag(self, event):
        if self.rect_start:
            pos = self.graphics_view.mapToScene(event.pos())
            x, y = pos.x(), pos.y()
            self.rect_end = (x, y)
            self.draw_rectangle()
            self.rectangles.pop()

    def draw_rectangle(self):
        if self.rect_start and self.rect_end:
            rect = QRectF(
                int(self.rect_start[0] / self.scale_factor),
                int(self.rect_start[1] / self.scale_factor),
                int((self.rect_end[0] - self.rect_start[0]) / self.scale_factor),
                int((self.rect_end[1] - self.rect_start[1]) / self.scale_factor)
            )
            self.rectangles.append(rect)
            self.graphics_scene.clear()
            self.display_image(self.image)  # Redraw the image

    def on_mouse_release(self, event):
        if self.rect_start and self.rect_end:
            self.draw_rectangle()
            self.rect_start = None
            self.rect_end = None

    def undo_rectangle(self):
        self.rectangles.pop()
        self.graphics_scene.clear()
        self.display_image(self.image)

    def clear_rectangles(self):
        self.rectangles = []
        self.graphics_scene.clear()
        self.display_image(self.image)

    def delete_selected_rectangle(self):
        if self.selected_rectangle:
            self.graphics_scene.removeItem(self.selected_rectangle)
            self.rectangles.remove(self.selected_rectangle)
            self.selected_rectangle = None

    def process(self):
        try:
            if self.video_capture:
                self.video_capture.release()
        except Exception as e:
            print(f"Error releasing video capture: {e}")
        try:
            fly_no = self.fly_no_entry.text()
            if not fly_no:
                fly_no = "NaN"
            if not self.file_path:
                raise ValueError("File is not set")

            csv_filename = os.path.splitext(self.file_path)[0] + ".csv"
            with open(csv_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Write header
                csvwriter.writerow(
                    ["First frame", "Last frame", "Fly Number", "Top Left X", "Top Left Y", "Bottom Right X",
                     "Bottom Right Y"])
                # Write rectangle data
                for rect in self.rectangles:
                    x1, y1 = rect.left(), rect.top()
                    x2, y2 = (rect.left() + rect.width()), (
                                rect.top() + rect.height())
                    csvwriter.writerow([self.frame_begin, self.frame_end, fly_no, x1, y1, x2, y2])
        except IOError as e:
            print(f"Error writing to CSV file: {e}")

        self.image = None
        self.rect_start = None
        self.rect_end = None
        self.rectangles = []
        self.selected_rectangle = None
        self.resizing = False
        self.scale_factor = 1
        self.video_capture = None
        self.total_frames = None
        self.frame_begin = 1
        self.frame_end = 1
        self.isTest = False
        self.file_path = None

        self.graphics_scene.clear()

    def toggle_test(self):
        self.isTest = not self.isTest

    def track(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Files")
        self.video_processor = VideoProcessor(file_paths)
        self.video_processor.finished.connect(self.on_processing_finished)
        self.video_processor.warning.connect(self.show_warning_message)
        self.video_processor.start()

    def on_processing_finished(self):
        QMessageBox.warning(self, "Done", "Video processing finished")

    def show_warning_message(self, message):
        QMessageBox.warning(self, "Warning", message)

    def decide(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Files")
        if not file_paths:  # Check if no files were selected
            return
        try:
            fly_count = int(self.fly_no_entry.text())  # Convert QLineEdit text to integer
        except ValueError:
            self.show_warning_message("Invalid input for fly count.")
            return

        self.tracker = ParticleTracker(file_paths, fly_count)
        self.tracker.finished.connect(self.on_processing_finished)
        self.tracker.warning.connect(self.show_warning_message)
        self.tracker.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image is not None:
            self.display_image(self.image)

    def change_numbering(self):
        if self.image is not None:
            self.display_image(self.image)

    def load_rectangles_from_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'r') as csvfile:
                    csvreader = csv.DictReader(csvfile)
                    required_columns = {"Top Left X", "Top Left Y", "Bottom Right X", "Bottom Right Y"}
                    if not required_columns.issubset(csvreader.fieldnames):
                        QMessageBox.warning(self, "File Corrupt", "The CSV file does not contain the required columns.")
                        return
                    self.rectangles = []  # Clear existing rectangles
                    for row in csvreader:
                        try:
                            x1 = float(row["Top Left X"])
                            y1 = float(row["Top Left Y"])
                            x2 = float(row["Bottom Right X"])
                            y2 = float(row["Bottom Right Y"])

                            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                            self.rectangles.append(rect)
                        except ValueError as e:
                            QMessageBox.warning(self, "File Corrupt", f"Error reading data from CSV: {e}")
                            return
                    if self.image is not None:
                        self.display_image(self.image)
            except IOError as e:
                QMessageBox.warning(self, "File Error", f"Error opening CSV file: {e}")

    def load_detections_from_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.candidates = pd.read_csv(file_path)
                if not all(col in self.candidates.columns for col in ["x", "y", "area", "eccentricity", "frame"]):
                    QMessageBox.warning(self, "Error", "CSV file is missing required columns.")
                    self.candidates = None
                else:
                    QMessageBox.information(self, "Success", "CSV file loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load CSV file: {e}")
                self.candidates = None

    def show_help(self):
        QMessageBox.information(self, "Help", "This is the help information for the application.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageMarkerApp()
    window.show()
    sys.exit(app.exec_())

