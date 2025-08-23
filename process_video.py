# process_video.py
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from norfair import Tracker, Detection

# UPDATED function signature to accept a confidence threshold
def analyze_video(video_path, conf_threshold):
    def to_norfair(detections, scale):
        # ... (this nested function remains the same)
        norfair_detections = []
        for det in detections:
            box_in_small_frame = det.xyxy[0].cpu().numpy()
            score = det.conf[0].cpu().numpy()
            box_in_big_frame = box_in_small_frame / scale
            points = np.array([[box_in_big_frame[0], box_in_big_frame[1]], [box_in_big_frame[2], box_in_big_frame[3]]])
            scores = np.array([score, score])
            norfair_detections.append(Detection(points=points, scores=scores))
        return norfair_detections

    OUTPUT_VIDEO_PATH = "output.avi"
    HEATMAP_OUTPUT_PATH = "heatmap.png"
    PROCESSING_WIDTH = 640

    model = YOLO('yolov8n.pt')
    source_points = np.array([[0, 43], [1185, 41], [1449, 735], [0, 629]], dtype=np.float32)
    destination_points = np.array([[0, 4], [277, 0], [280, 174], [1, 175]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    tracker = Tracker(distance_function="iou", distance_threshold=0.7)
    player_paths = {}

    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        scale = PROCESSING_WIDTH / frame.shape[1]
        processing_height = int(frame.shape[0] * scale)
        resized_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height))
        # UPDATED model call to use the new parameter
        yolo_results = model(resized_frame, classes=[0], conf=conf_threshold, verbose=False)[0]
        norfair_detections = to_norfair(yolo_results.boxes, scale)
        tracked_objects = tracker.update(detections=norfair_detections)
        for obj in tracked_objects:
            points = obj.estimate
            x1, y1 = points[0]; x2, y2 = points[1]
            obj_id = obj.id
            center_x, bottom_y = int((x1 + x2) / 2), int(y2)
            player_point = np.array([[[center_x, bottom_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(player_point, matrix)[0][0]
            if obj_id not in player_paths:
                player_paths[obj_id] = []
            player_paths[obj_id].append(transformed_point)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"P-{obj_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        video_writer.write(frame)

    video_capture.release()
    video_writer.release()

    field_map = cv2.imread("field_map.png")
    if field_map is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(field_map, cv2.COLOR_BGR2RGB))
        for player_id, path in player_paths.items():
            if len(path) < 2: continue
            df = pd.DataFrame(path, columns=['x', 'y'])
            sns.kdeplot(x=df['x'], y=df['y'], cmap="Reds", fill=True, thresh=0.05, ax=ax)
        ax.set_xlim(0, field_map.shape[1]); ax.set_ylim(field_map.shape[0], 0)
        ax.axis('off')
        plt.savefig(HEATMAP_OUTPUT_PATH, bbox_inches='tight', pad_inches=0)
    
    return OUTPUT_VIDEO_PATH, HEATMAP_OUTPUT_PATH