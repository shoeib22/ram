import cv2
from ultralytics import YOLO

model_instance = YOLO('yolov8n.pt')

video_source_path = 'input_video.mp4'
video_capture = cv2.VideoCapture(video_source_path)

while video_capture.isOpened():
    success, frame_data = video_capture.read()

    if not success:
        break

    detection_results = model_instance(frame_data)
    
    annotated_frame = detection_results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()