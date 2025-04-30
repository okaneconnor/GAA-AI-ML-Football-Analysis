import torch
import cv2
import numpy as np
from ultralytics import YOLO # <<< Import the YOLO class
import os # <<< Import os for path handling

# --- Device Selection (Corrected for Colab/CUDA) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("[DETECTION] Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[DETECTION] Using MPS GPU")
else:
    device = torch.device("cpu")
    print("[DETECTION] Using CPU")

# --- Load YOLOv11 Model using ultralytics API ---
# Use the absolute path from your Colab environment
# IMPORTANT: Ensure this path matches where Cell 3 downloaded the model
model_abs_path = "/content/models/yolo11n.pt" # <<< Use absolute path

if os.path.exists(model_abs_path):
    print(f"[DETECTION] Loading model from: {model_abs_path}")
    try:
        model = YOLO(model_abs_path) # <<< Use the YOLO class
        model.to(device) # Move model to the selected device
        # No explicit model.eval() needed, YOLO handles modes internally
        print(f"[DETECTION] Model loaded successfully onto {device}.")
    except Exception as e:
        print(f"[DETECTION] Error loading model: {e}")
        model = None # Set model to None if loading failed
else:
    print(f"[DETECTION] Error: Model file not found at {model_abs_path}")
    model = None

# --- Detect Objects Function (Updated for ultralytics Results) ---
def detect_objects(frame):
    """
    Perform object detection on a frame using the loaded YOLO model.
    Returns detection results as a list of dictionaries compatible with the old format.
    """
    detections = [] # Initialize empty list

    if model is None:
        print("[DETECTION] Error: Model not loaded, cannot perform detection.")
        return detections # Return empty list if model failed to load

    # Perform inference using the ultralytics model
    # No need to convert color (YOLO handles BGR/RGB automatically)
    # Set verbose=False to avoid excessive console output from predict
    try:
        results = model.predict(source=frame, device=device, verbose=False)
    except Exception as e:
        print(f"[DETECTION] Error during model prediction: {e}")
        return detections # Return empty list on prediction error


    # Process results (results is a list, usually with one element for one image)
    if results and results[0].boxes:
        boxes = results[0].boxes # Get the Boxes object

        # Iterate through detected boxes
        for i in range(len(boxes)):
            # Extract data (accessing attributes of the Boxes object)
            # .data gives a tensor with [xmin, ymin, xmax, ymax, conf, cls]
            box_data = boxes.data[i].cpu().numpy() # Get as numpy array on CPU
            xmin, ymin, xmax, ymax, conf, cls_id = box_data

            # Get class name from model metadata
            class_name = model.names[int(cls_id)]

            # Append detection in the desired dictionary format
            detections.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'confidence': conf,
                'class': int(cls_id), # Keep class ID as int
                'name': class_name  # Class name string
            })

    return detections