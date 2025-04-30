# football_utils/detection.py

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
import traceback

# --- Device Selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("[DETECTION] Using CUDA GPU")
elif torch.backends.mps.is_available(): # Keep MPS check for broader compatibility
    device = torch.device("mps")
    print("[DETECTION] Using MPS GPU")
else:
    device = torch.device("cpu")
    print("[DETECTION] Using CPU")

# --- Load YOLOv11 Model ---
model_identifier = "/content/models/yolo11n.pt" # Or just "yolo11n.pt"
print(f"[DETECTION] Attempting to load/download model: {model_identifier}")
model = None
try:
    model = YOLO(model_identifier)
    model.to(device)
    print(f"[DETECTION] Model '{model_identifier}' loaded successfully onto {device}.")
    if not hasattr(model, 'names') or not model.names:
         print("[DETECTION] Warning: model.names not found after loading.")

except Exception as e:
    print(f"[DETECTION] Error loading model '{model_identifier}': {e}")
    print(traceback.format_exc()) # Ensure errors during load are printed

# --- Detect Objects Function ---
def detect_objects(frame):
    detections = []
    if model is None:
        print("[DETECTION] Error: Model is not loaded.")
        return detections
    if not hasattr(model, 'names') or not model.names:
        print("[DETECTION] Error: Model names missing.")
        return detections

    try:
        # *** Use half=True for potential FP16 speedup on compatible GPUs (like L4/T4) ***
        results = model.predict(source=frame, device=device, verbose=False, half=True)
    except Exception as e:
        print(f"[DETECTION] Error during model prediction: {e}")
        print(traceback.format_exc())
        return detections

    if results and results[0].boxes:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            try:
                box_data = boxes.data[i].cpu().numpy()
                xmin, ymin, xmax, ymax, conf, cls_id_float = box_data
                cls_id = int(cls_id_float)
                if 0 <= cls_id < len(model.names): class_name = model.names[cls_id]
                else: class_name = "unknown"; print(f"[DETECTION] Warning: Invalid class ID {cls_id}")

                detections.append({
                    'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                    'confidence': conf, 'class': cls_id, 'name': class_name
                })
            except Exception as proc_err:
                 print(f"[DETECTION] Error processing detection index {i}: {proc_err}")
    return detections