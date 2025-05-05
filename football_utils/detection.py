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

# --- Load YOLOv11 Model (Original Logic - Module Level) ---
# IMPORTANT: This path is hardcoded based on your original script.
# Ensure your 'yolo11n.pt' file is EXACTLY at '/content/models/yolo11n.pt'
model_identifier = "/content/models/yolo11n.pt"
print(f"[DETECTION] Attempting to load model directly: {model_identifier}")
model = None # Initialize model variable globally within this module
try:
    if not os.path.exists(model_identifier):
         raise FileNotFoundError(f"Model file not found at the required path: {model_identifier}")

    model = YOLO(model_identifier)
    model.to(device)
    print(f"[DETECTION] Model '{model_identifier}' loaded successfully onto {device}.")
    # Basic check for model usability (optional but good)
    if not hasattr(model, 'predict'):
         print("[DETECTION] Warning: Loaded model might not have a 'predict' method.")
         model = None # Invalidate model if predict is missing
    # Check for names attribute needed later (optional, handled in detect_objects)
    # if not hasattr(model, 'names') or not model.names:
    #       print("[DETECTION] Warning: model.names not found after loading.")

except Exception as e:
    print(f"[DETECTION] Error loading model '{model_identifier}': {e}")
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
    # Use the 'model' variable defined globally in this module
    global model
    if model is None:
        print("[DETECTION] Error: Model is not loaded (was None at module level).")
        return sv.Detections.empty()

    try:
        # Perform inference, explicitly asking for class 0 (person)
        # Note: Ensure your yolo11n model has 'person' as class 0
        results = model(frame, classes=[0], conf=confidence_threshold, device=device, verbose=False)

        if results and results[0].boxes:
             # Convert results directly to supervision.Detections format
             detections = sv.Detections.from_ultralytics(results[0])
             # Ensure required attributes exist after conversion
             if detections.xyxy is None: detections.xyxy = np.empty((0, 4))
             if detections.confidence is None: detections.confidence = np.empty((0,))
             if detections.class_id is None: detections.class_id = np.empty((0,))
             return detections
        else:
            return sv.Detections.empty() # No detections found

    except Exception as e:
        print(f"[DETECTION] Error during model prediction or conversion: {e}")
        # print(traceback.format_exc()) # Optional: uncomment for detailed error
        return sv.Detections.empty()