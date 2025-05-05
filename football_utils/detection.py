# football_utils/detection.py

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv # Make sure this is installed!
import os
import traceback

# --- Device Selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("[DETECTION] Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("[DETECTION] Using CPU")

# --- Load YOLOv11 Model (Module Level - Using Name for Auto-Download) ---
# Use the model NAME - the library will download it if not cached.
# Ensure internet connection is available in your environment for first run.
model_identifier = "yolo11n.pt" # Use the NAME, not a specific path
print(f"[DETECTION] Requesting model: {model_identifier} (will download if needed)")
model = None # Initialize model variable globally within this module
try:
    # Pass the NAME to YOLO() - it handles download/cache check
    model = YOLO(model_identifier)
    model.to(device)
    print(f"[DETECTION] Model '{model_identifier}' loaded/downloaded successfully onto {device}.")
    # Basic check for model usability
    if not hasattr(model, 'predict'):
         print("[DETECTION] Warning: Loaded model might not have a 'predict' method.")
         model = None # Invalidate model

except Exception as e:
    print(f"[DETECTION] Error loading/downloading model '{model_identifier}': {e}")
    print(traceback.format_exc())
    model = None # Ensure model is None if loading fails

# --- Detect Objects Function (Uses Module-Level Model) ---
def detect_objects(frame, confidence_threshold=0.3):
    """
    Detects objects (specifically class 0: person) in a frame using the module-level YOLO model.

    Args:
        frame: The input video frame (NumPy array).
        confidence_threshold (float): Minimum confidence score for detections.

    Returns:
        supervision.Detections: Detected objects filtered for class 0 and confidence,
                                or sv.Detections.empty() if no valid detections or error.
    """
    global model # Access the module-level model variable
    if model is None:
        print("[DETECTION] Error: Model is not loaded (was None after initialization attempt).")
        return sv.Detections.empty()

    try:
        # Perform inference
        results = model(frame, classes=[0], conf=confidence_threshold, device=device, verbose=False)

        if results and results[0].boxes:
             detections = sv.Detections.from_ultralytics(results[0])
             # Ensure required attributes exist after conversion
             if detections.xyxy is None: detections.xyxy = np.empty((0, 4))
             if detections.confidence is None: detections.confidence = np.empty((0,))
             if detections.class_id is None: detections.class_id = np.empty((0,))
             return detections
        else:
            return sv.Detections.empty()

    except Exception as e:
        print(f"[DETECTION] Error during model prediction or conversion: {e}")
        return sv.Detections.empty()