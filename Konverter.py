import subprocess
import tkinter as tk
from tkinter import filedialog
import os

def convert_to_mp4(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-codec', 'copy',
        output_file
    ]
    subprocess.run(command)

def select_input_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    input_file_paths = filedialog.askopenfilenames(title="Select MKV Video Files", filetypes=[("MKV files", "*.mkv")])
    return input_file_paths

def generate_output_file(input_file):
    # Remove the file extension and add .mp4
    output_file_name = os.path.splitext(input_file)[0] + ".mp4"
    return output_file_name

input_files = select_input_files()
if input_files:
    for input_file in input_files:
        output_file = generate_output_file(input_file)
        convert_to_mp4(input_file, output_file)
        print(f"Conversion completed. Output file: {output_file}")
    print("All conversions completed.")
else:
    print("No input files selected. Conversion aborted.")
