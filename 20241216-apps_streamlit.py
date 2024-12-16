import os
os.environ['NUM_WORKERS'] = '0'
import torch
import streamlit as st
import cv2
import streamlink
from ultralytics import YOLO
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import datetime, timedelta
from collections import defaultdict
import random
import pytz
from datetime import datetime
import supervision as sv
from math import sqrt

# Initialize the YOLO model with GPU support
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Initialize the YOLO model
model_path = '/Users/salwainasshabira/Documents/CBI/Product/CustomerDetector-2/Model/best_custDetect.pt'
model = YOLO(model_path)
model.to(device)  # Move model to GPU

utc_timezone = pytz.timezone('UTC')
datetime_utc = datetime.now(utc_timezone)
wib_timezone = pytz.timezone('Asia/Jakarta')
dateNow = datetime_utc.astimezone(wib_timezone)

dateSimple = dateNow.strftime("%A, %d %b %Y")
timeNow = dateNow.strftime("%H:%M WIB")
yearNow = dateNow.strftime("%Y")

# Define the folder containing videos
def list_videos_in_directory(directory_path):
    # Supported video file extensions
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.mpv'}

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        st.error("The provided path is not a valid directory.")
        return []

    # List video files
    video_files = [f for f in os.listdir(directory_path) if os.path.splitext(f)[1].lower() in video_extensions]
    return video_files


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def calculate_centroid(box):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return x_center, y_center

def euclidean_distance(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def filter_detections(results, threshold=0.1, iou_threshold=0.5, position_threshold=50):
    filtered_boxes = []
    for result in results:
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:  # Apply confidence threshold
            box = [x1, y1, x2, y2]
            centroid = calculate_centroid(box)
            is_duplicate = False

            for kept_box in filtered_boxes:
                kept_centroid = calculate_centroid(kept_box[:4])
                if (calculate_iou(box, kept_box[:4]) > iou_threshold or
                    euclidean_distance(centroid, kept_centroid) < position_threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_boxes.append([x1, y1, x2, y2, score, class_id])

    return filtered_boxes


def process_video(source, folder_path=None, additional_input=None):
    #--------------------------- FILE VIDEO ------------------------------#
    if source == "Video File (.mp4, .avi, .mkv)":
        if folder_path is None or additional_input is None:
            st.error("⚠️ Error: Missing folder path or video file name.")
            return
        
        video_path = os.path.join(folder_path, additional_input)
        if not os.path.exists(video_path):
            st.error(f"⚠️ Error: Video file {additional_input} not found in {folder_path}.")
            return
    
        if torch.cuda.is_available():
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            gpu_frame = cv2.cuda_GpuMat()  # GPU Mat for faster processing
        else:
            cap = cv2.VideoCapture(video_path)

        # Initialize variables for tracking and visualization
        label_counts = defaultdict(list)
        timestamps = []
        colors = {}
        heatmap_resolution = (360)
        heatmap_data = np.zeros((heatmap_resolution, 0))  # Initialize empty heatmap (rows=y-coordinates, columns=time)

            
        frame_placeholder = st.empty()
        row1_col1, row1_col2 = st.columns(2)  # Row 1: Line chart and box plot
        row2 = st.container()  # Row 2: Heatmap

        # Placeholders for graphs
        graph_placeholder_time_series = row1_col1.empty()
        graph_placeholder_bar = row1_col2.empty()
        graph_placeholder_heatmap = row2.empty()

        last_plot_time = time.time()
        plot_interval = 2  # Update graphs every 2 seconds
        threshold = 0.1  # Detection threshold
        last_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.sidebar.error("Failed to grab frame.")
                    break

                # Optimize frame processing with GPU if available
                if torch.cuda.is_available():
                    gpu_frame.upload(frame)
                    frame = cv2.cuda.resize(gpu_frame, (640, 360))
                    frame = frame.download()
                else:
                    frame = cv2.resize(frame, (640, 360))

                # Run inference on the model
                with torch.cuda.amp.autocast():  # Enable mixed precision
                    results = model(frame, device=device)[0]

                current_time = time.time()
                if current_time - last_time < 0.5:  # Process frame every 0.5 seconds
                    continue
                last_time = current_time

                labels_this_frame = {}
                column_data = np.zeros(heatmap_resolution)  # New column for the heatmap
                
                for result in results.boxes.data.tolist():
                    # Apply filtering for duplicate detections
                    filtered_results = filter_detections(results.boxes.data.tolist(), threshold=0.1, iou_threshold=0.5, position_threshold=50)

                    for result in filtered_results:
                        x1, y1, x2, y2, score, class_id = result
                        label = results.names[int(class_id)].upper()
                        labels_this_frame[label] = labels_this_frame.get(label, 0) + 1
                        
                        # Update heatmap
                        y_center = int((y1 + y2) / 2)
                        if 0 <= y_center < heatmap_resolution:
                            column_data[y_center] += 1  # Increment heatmap value at y_center

                        # Drawing rectangles and labels
                        if torch.cuda.is_available():
                            cv2.cuda.rectangle(gpu_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                            frame = gpu_frame.download()
                            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                
                # Append the new column to the heatmap
                heatmap_data = np.hstack((heatmap_data, column_data.reshape(-1, 1)))
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame)

                if labels_this_frame and current_time - last_plot_time > plot_interval:
                    last_plot_time = current_time
                    timestamp = datetime.now() + timedelta(hours=7)  # Adjusting time to GMT+7
                    timestamps.append(timestamp)
                    for label, count in labels_this_frame.items():
                        label_counts[label].append(count)
                        if label not in colors:
                            colors[label] = f'rgba({random.randint(100, 200)}, {random.randint(100, 200)}, {random.randint(100, 200)}, 0.5)'

                    update_plots(
                        timestamps, 
                        label_counts, 
                        colors, 
                        graph_placeholder_time_series, 
                        graph_placeholder_bar, 
                        graph_placeholder_heatmap, 
                        heatmap_data
                    )

                # Update plots
                if labels_this_frame and current_time - last_plot_time > plot_interval:
                    last_plot_time = current_time
                    timestamp = datetime.now() + timedelta(hours=7)  # Adjusting time to GMT+7
                    timestamps.append(timestamp)
                    for label, count in labels_this_frame.items():
                        label_counts[label].append(count)
                        if label not in colors:
                            colors[label] = f'rgba({random.randint(100, 200)}, {random.randint(100, 200)}, {random.randint(100, 200)}, 0.5)'

                    update_plots(
                        timestamps, 
                        label_counts, 
                        colors, 
                        graph_placeholder_time_series, 
                        graph_placeholder_bar, 
                        graph_placeholder_heatmap, 
                        heatmap_data
                    )
   
                # Monitor GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # Convert to MB
                    st.sidebar.text(f'GPU Memory Used: {memory_allocated:.0f} MB')

        finally:
            cap.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory   
    #--------------------- LINK STREAMING YOUTUBE CCTV --------------------# 
    elif source == "Video Live":
        if additional_input is None:
            st.sidebar.error("⚠️ Additional input is required (YouTube URL and Quality).")
            return
        
        youtube_url = additional_input.get("url")
        quality = additional_input.get("quality")

        if not youtube_url or not quality:
            st.sidebar.error("⚠️ Please provide a valid YouTube URL and select quality.")
            return

        # Get the stream URL based on selected quality
        try:
            streams = streamlink.streams(youtube_url)
            quality_dict = {'Low': '480p', 'Medium': '720p', 'High': '1080p'}
            stream_url = streams.get(quality_dict[quality], streams['best']).url
        except Exception as e:
            st.error(f"Error fetching stream: {e}")
            return

        # Enable CUDA backend for OpenCV if available
        if torch.cuda.is_available():
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            gpu_frame = cv2.cuda_GpuMat()  # GPU Mat for faster processing
        else:
            cap = cv2.VideoCapture(stream_url)

        # Initialize variables for tracking and visualization
        label_counts = defaultdict(list)
        timestamps = []
        colors = {}

        frame_placeholder = st.empty()
        graph_col1, graph_col2 = st.columns(2)
        graph_placeholder_time_series = graph_col1.empty()
        graph_placeholder_bar = graph_col2.empty()

        last_plot_time = time.time()
        plot_interval = 2  # Update graphs every 2 seconds
        threshold = 0.1  # Detection threshold
        last_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.sidebar.error("Failed to grab frame.")
                    break

                # Optimize frame processing with GPU if available
                if torch.cuda.is_available():
                    gpu_frame.upload(frame)
                    frame = cv2.cuda.resize(gpu_frame, (640, 360))
                    frame = frame.download()
                else:
                    frame = cv2.resize(frame, (640, 360))

                # Run inference on the model
                with torch.cuda.amp.autocast():  # Enable mixed precision
                    results = model(frame, device=device)[0]

                current_time = time.time()
                if current_time - last_time < 0.5:  # Process frame every 0.5 seconds
                    continue
                last_time = current_time

                labels_this_frame = {}

                for result in results.boxes.data.tolist():
                    # Apply filtering for duplicate detections
                    filtered_results = filter_detections(results.boxes.data.tolist(), threshold=0.1, iou_threshold=0.5, position_threshold=50)

                    for result in filtered_results:
                        x1, y1, x2, y2, score, class_id = result
                        label = results.names[int(class_id)].upper()
                        labels_this_frame[label] = labels_this_frame.get(label, 0) + 1

                        # Drawing rectangles and labels
                        if torch.cuda.is_available():
                            cv2.cuda.rectangle(gpu_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                            frame = gpu_frame.download()
                            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                            
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame)

                if labels_this_frame and current_time - last_plot_time > plot_interval:
                    last_plot_time = current_time
                    timestamp = datetime.now() + timedelta(hours=7)  # Adjusting time to GMT+7
                    timestamps.append(timestamp)
                    for label, count in labels_this_frame.items():
                        label_counts[label].append(count)
                        if label not in colors:
                            colors[label] = f'rgba({random.randint(100, 200)}, {random.randint(100, 200)}, {random.randint(100, 200)}, 0.5)'

                    update_plots(timestamps, label_counts, colors, graph_placeholder_time_series, graph_placeholder_bar)
   
                # Monitor GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # Convert to MB
                    st.sidebar.text(f'GPU Memory Used: {memory_allocated:.0f} MB')

        finally:
            cap.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory
                
                
# ------------------------------------------PROCESS-------------------------------------------                
def main():
    st.title('CtrendVision Person Detection')
    st.sidebar.image('/Users/salwainasshabira/Documents/CBI/Product/Logo-CtrendVision.png')
    st.sidebar.text(f"Today\t: {dateSimple}")
    st.sidebar.text(f"Time\t: {timeNow}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.text(f'Using device: {device}')
    st.sidebar.title('Settings')
    
    with st.sidebar:
        SOURCE_VIDEO = st.selectbox("Select Source:", ("Video File (.mp4, .avi, .mkv)", "Video Live"), index=0)

        # Handle "Video File" option
        if SOURCE_VIDEO == "Video File (.mp4, .avi, .mkv)":
            st.sidebar.text('Example Path : /open/your/path/file/')

            folder_path = st.sidebar.text_input("Enter the path to the video file:")

            if folder_path:
                video_list = list_videos_in_directory(folder_path)
                if video_list:
                    selected_video = st.sidebar.selectbox("Select a video to process:", video_list, index=0)
                else:
                    st.warning("No video files found in the specified directory.")

        elif SOURCE_VIDEO == "Video Live":
                youtube_url = st.sidebar.text_input("Enter YouTube URL:")
                quality = st.sidebar.selectbox("Select stream quality:", ("Low", "Medium", "High"), index=0)
    
    # ------------------------------------------Process Button-------------------------------------------            
    if SOURCE_VIDEO == "Video File (.mp4, .avi, .mkv)":
        #st.sidebar.text('Example Path: /open/your/path/file/')
        #folder_path = st.sidebar.text_input("Enter the path to the video file:")

        if st.sidebar.button("Start Processing", key="start_video_file"):
            if folder_path.strip():
                if selected_video.strip():
                    process_video(SOURCE_VIDEO, folder_path, selected_video)
                else:
                    st.sidebar.error("⚠️ Please select a valid video file.")
            else:
                st.sidebar.error("⚠️ Please provide a valid video file path.")
                                    
    elif SOURCE_VIDEO == "Video Live":
        if st.sidebar.button("Start Processing", key="start_video_live"):
            if youtube_url.strip():
                additional_input = {"url": youtube_url, "quality": quality}
                process_video(SOURCE_VIDEO, None, additional_input)
            else:
                st.sidebar.error("⚠️ Please provide a valid YouTube URL.")

# ------------------------------------------REPORT OF RESULT (GRAPH)-------------------------------------------                 
def update_plots(timestamps, label_counts, colors, graph_placeholder_time_series, graph_placeholder_bar, graph_placeholder_heatmap, heatmap_data):
    # Line chart: Time series
    time_series = go.Figure()
    for label, counts in label_counts.items():
        time_series.add_trace(go.Scatter(x=timestamps, y=counts, mode='lines+markers', name=label, line=dict(color=colors[label])))
    time_series.update_layout(title='Detection Count Over Time', xaxis_title='Time (GMT+7)', yaxis_title='Count')

    # Box plot: Bar chart
    bar_plot = go.Figure()
    for label, color in colors.items():
        total = sum(label_counts[label])
        bar_plot.add_trace(go.Bar(x=[label], y=[total], name=label, marker_color=color))
    bar_plot.update_layout(title='Total Count of Labels', xaxis_title='Labels', yaxis_title='Total Count')

    # Heatmap plot: Time vs Y-coordinate
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[t.strftime('%H:%M:%S') for t in timestamps],  # Convert timestamps to readable format
        y=list(range(heatmap_data.shape[0])),
        colorscale='Jet'
    ))
    heatmap_fig.update_layout(title='Heatmap: Time vs Y-Coordinates', xaxis_title='Time (GMT+7)', yaxis_title='Y Coordinate')

    # Display plots
    graph_placeholder_time_series.plotly_chart(time_series, use_container_width=True)
    graph_placeholder_bar.plotly_chart(bar_plot, use_container_width=True)
    graph_placeholder_heatmap.plotly_chart(heatmap_fig, use_container_width=True)

if __name__ == '__main__':
    main()

# ------------------------------------------FOOTER-------------------------------------------  
st.divider()
st.caption(f'©️{yearNow} Cybertrend Intrabuana. All rights reserved.')
    
 # ------------------------------------------DOWNLOAD MENU------------------------------------------- 
# # Display charts
# col1, col2 = st.columns(2)
# with col1:
#         st.write("**Unique ID Counts per Class**")
#         st.bar_chart(unique_id_chart)
        
# with col2:
#         st.write("**Count Over Time per Class**")
#         st.line_chart(count_over_time_chart)

# st.markdown("Download processed files:")

# colA, colB = st.columns(2)

# # Download button for video
# with colA:
#     download_button("Download Video", "Output/video-output.mp4", "video")

# # Download button for CSV
# with colB:
#     download_button("Download CSV", "Output/detection.csv", "csv")
    
   
 # ------------------------------------------FOOTER-------------------------------------------  
# st.divider()
# st.caption(f'©️{yearNow} Cybertrend Intrabuana. All rights reserved.')



