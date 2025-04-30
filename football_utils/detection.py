import torch
import cv2
import numpy as np

# Set device to MPS if available (for Apple M3), otherwise CPU/GPU.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load YOLO11 model (ensure 'yolo11n.pt' is in the models folder)
model_path = "models/yolo11n.pt"
model = torch.hub.load('ultralytics/ultralytics', 'custom', path=model_path)
model.to(device)
model.eval()

def detect_objects(frame):
    """
    Perform object detection on a frame using YOLO.
    Returns detection results in a dictionary.
    """
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    # Convert the results to the same format as YOLOv11
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    # Each detection includes: xmin, ymin, xmax, ymax, confidence, class, name
    return detections