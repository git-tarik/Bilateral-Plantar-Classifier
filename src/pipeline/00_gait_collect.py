# Bilateral-Plantar-Classifier — All rights reserved.
# Copyright (c) 2025 <Your Name/Team>. See LICENSE.
# NOTE: Do not commit human-subject data. See DATA_POLICY.md.
import pandas as pd
import tkinter as tk
from tkinter import simpledialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import serial
import time
from PIL import ImageGrab

# Serial Data Logger Class
class SerialLogger:
    def __init__(self, port='COM3', baud=9600):
        self.port = port
        self.baud = baud
        self.ser = None

    def connect(self):
        """Establish serial connection."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print("✅ Serial connection established.")
        except (serial.SerialException, OSError) as e:
            messagebox.showerror("❌ Connection Error", f"Failed to connect: {e}")
            self.ser = None

    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()

    def read_data(self):
        """Read and decode serial data."""
        if self.ser and self.ser.is_open:
            try:
                data = self.ser.readline().decode().strip()
                values = [float(x) / 1024 for x in data.split("-")]
                return values if len(values) == 4 else None
            except (ValueError, UnicodeDecodeError):
                return None
        return None

# Create CSV File with Timestamp
def log_sensor_data():
    logger = SerialLogger()
    logger.connect()
    if not logger.ser:
        return None

    file_name = simpledialog.askstring("Log File Name", "Enter name for log file:") or "default_log"
    timestamp = time.strftime("%d_%B_%Y_%Hh_%Mm_%Ss", time.localtime())
    filename = f"{file_name}_{timestamp}.csv"

    print(f"📌 Logging data to: {filename}")
    start_time = time.time()
    log_entries = ["Timestamp,No,FFR1,FFR2,FFR3,HFR4"]
    count = 1

    while time.time() - start_time < 60:  # Increased from 30s to 60s
        values = logger.read_data()
        if values:
            log_time = time.strftime("%H:%M:%S", time.localtime())  # Timestamp
            log_entries.append(f"{log_time},{count},{','.join(map(str, values))}")
            print(f"🔹 Data Collected at {log_time}: {values}")
            count += 1

    logger.disconnect()

    # Save data correctly into CSV file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\n".join(log_entries))

    print(f"✅ Data successfully saved in {filename}")
    return filename

# Load CSV Data
def load_csv(file_path):
    """Load CSV file into DataFrame."""
    if not file_path or not os.path.exists(file_path):
        messagebox.showerror("❌ Error", "CSV file not found.")
        return None
    df = pd.read_csv(file_path)
    if df.empty:
        messagebox.showerror("❌ Error", "CSV file is empty.")
        return None
    return df

# Analyze Sensor Data
def analyze_data(df):
    """Compute average values for sensors."""
    sensors = ['FFR1', 'FFR2', 'FFR3', 'HFR4']
    avg_values = {sensor: df[sensor].mean() for sensor in sensors}
    return sensors, avg_values

# Determine Walking Type
def get_walking_type(avg_values):
    """Determine walking type based on average sensor values."""
    min_sensor = min(['FFR1', 'FFR2', 'FFR3'], key=lambda sensor: avg_values[sensor])
    return {"FFR1": "Toe walking", "FFR2": "Balanced walking", "FFR3": "Sideway walking"}.get(min_sensor, "Unknown")

# Determine Foot Walking Type
def get_foot_walking_type(avg_values):
    """Determine foot walking type based on all sensors."""
    return "Heel foot walking" if min(avg_values, key=avg_values.get) == 'HFR4' else "Front foot walking"

# Save Full Tkinter Window Screenshot
def save_window_screenshot(root, filename):
    """Capture and save a full screenshot of a Tkinter window using bbox."""
    root.update()
    time.sleep(1)  # Ensure window is fully rendered
    x0 = root.winfo_rootx()
    y0 = root.winfo_rooty()
    x1 = x0 + root.winfo_width()
    y1 = y0 + root.winfo_height()
    ImageGrab.grab(bbox=(x0, y0, x1, y1)).save(filename)
    print(f"✅ Screenshot saved: {filename}")

# Display Results and Graph in a Single Window
def display_results_and_graph(file_name, sensors, avg_values, walking_type, foot_walking_type):
    root = tk.Tk()
    root.title("🚶 Walking Analysis Results & Sensor Data Graph")

    # Frame for text-based results
    text_frame = tk.Frame(root)
    text_frame.pack(side=tk.LEFT, padx=20, pady=20)

    tk.Label(text_frame, text=f"🚶 Walking Type: {walking_type}", font=("Arial", 14, "bold")).pack(pady=10)
    tk.Label(text_frame, text=f"🦶 Foot Walking Type: {foot_walking_type}", font=("Arial", 14, "bold"), fg="darkblue").pack(pady=5)

    avg_text = "\n".join([f"{sensor}: {value:.2f}" for sensor, value in avg_values.items()])
    tk.Label(text_frame, text=f"📊 Average Sensor Values:\n{avg_text}", font=("Arial", 12)).pack(pady=5)

    # Frame for Graph
    graph_frame = tk.Frame(root)
    graph_frame.pack(side=tk.RIGHT, padx=20, pady=20)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(sensors, [avg_values[sensor] for sensor in sensors], color=['blue', 'green', 'red', 'purple'])
    ax.set_xlabel("Sensors")
    ax.set_ylabel("Average Sensor Values")
    ax.set_title("📊 Sensor Data Histogram")
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Save the histogram as an image (Renamed correctly)
    graph_image_path = f"graph_{file_name}.png"
    fig.savefig(graph_image_path)
    print(f"✅ Graph saved: {graph_image_path}")

    # Save the combined window screenshot
    save_window_screenshot(root, f"{file_name}_full_window.png")

    root.mainloop()

# Main Executions
file_path = log_sensor_data()
if file_path:
    df = load_csv(file_path)
    if df is not None:
        sensors, avg_values = analyze_data(df)
        walking_type = get_walking_type(avg_values)
        foot_walking_type = get_foot_walking_type(avg_values)

        display_results_and_graph(file_path, sensors, avg_values, walking_type, foot_walking_type)
def save_graph(figure, filename="analysis_graph.png"):
    try:
        figure.savefig(filename, bbox_inches='tight')
        messagebox.showinfo("✅ Screenshot Saved", f"Graph saved as {filename}")
    except Exception as e:
        messagebox.showerror("❌ Save Failed", f"Could not save graph: {e}")
