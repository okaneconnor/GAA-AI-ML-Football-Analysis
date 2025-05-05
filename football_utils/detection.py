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

# --- Load YOLO Model Function ---
def load_yolo_model(model_path='yolo11n.pt'): # Default back to yolo11n
    """Loads the specified YOLO model onto the selected device."""
    # If using Colab-like path, adjust as needed or pass full path
    # Example: model_full_path = os.path.join('/content/models', model_path)
    model_full_path = model_path # Assume full path or relative path is given

    print(f"[DETECTION] Attempting to load model: {model_full_path}")
    model = None
    try:
        if not os.path.exists(model_full_path):
             # Try adding default model dir if simple name given
             default_model_dir = "/content/models"
             potential_path = os.path.join(default_model_dir, model_path)
             if os.path.exists(potential_path):
                 model_full_path = potential_path
                 print(f"[DETECTION] Found model at: {model_full_path}")
             else:
                 raise FileNotFoundError(f"Model file not found at {model_path} or {potential_path}")

        model = YOLO(model_full_path)
        model.to(device)
        print(f"[DETECTION] Model '{model_full_path}' loaded successfully onto {device}.")
        # Basic check for model usability
        if not hasattr(model, 'predict'):
             print("[DETECTION] Warning: Loaded model might not have a 'predict' method.")
             raise ValueError("Model loading failed or model invalid.")

    except Exception as e:
        print(f"[DETECTION] Error loading model '{model_full_path}': {e}")
        print(traceback.format_exc()) # Ensure errors during load are printed
        return None # Return None if loading fails
    return model

# --- Detect Objects Function (Returns supervision.Detections) ---
def detect_objects(model, frame, confidence_threshold=0.3):
    """
    Detects objects (specifically class 0: person) in a frame using the provided YOLO model.

    Args:
        model: The loaded YOLO model object.
        frame: The input video frame (NumPy array).
        confidence_threshold (float): Minimum confidence score for detections.

    Returns:
        supervision.Detections: Detected objects filtered for class 0 and confidence,
                                or sv.Detections.empty() if no valid detections or error.
    """
    if model is None:
        print("[DETECTION] Error: Model is not loaded.")
        return sv.Detections.empty()

    try:
        # Perform inference, explicitly asking for class 0 (person)
        # Note: Ensure your yolo11n model has 'person' as class 0
        results = model(frame, classes=[0], conf=confidence_threshold, device=device, verbose=False)

        if results and results[0].boxes:
             # Convert results directly to supervision.Detections format
             # This assumes the output format from ultralytics lib is consistent for v11
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