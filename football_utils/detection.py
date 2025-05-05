# football_utils/detection.py

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import traceback

# --- Device Selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("[DETECTION] Using CUDA GPU")
# Remove MPS check if causing issues or if CUDA/CPU are primary targets
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("[DETECTION] Using MPS GPU")
else:
    device = torch.device("cpu")
    print("[DETECTION] Using CPU")

# --- Load YOLOv8 Model Function ---
def load_yolo_model(model_name='yolov8m.pt'):
    """Loads the specified YOLO model onto the selected device."""
    print(f"[DETECTION] Attempting to load model: {model_name}")
    model = None
    try:
        model = YOLO(model_name)
        model.to(device)
        print(f"[DETECTION] Model '{model_name}' loaded successfully onto {device}.")
        # Basic check for model usability
        if not hasattr(model, 'predict'):
             print("[DETECTION] Warning: Loaded model might not have a 'predict' method.")
             raise ValueError("Model loading failed or model invalid.")

    except Exception as e:
        print(f"[DETECTION] Error loading model '{model_name}': {e}")
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
        results = model(frame, classes=[0], conf=confidence_threshold, device=device, verbose=False)

        if results and results[0].boxes:
             # Convert results directly to supervision.Detections format
             detections = sv.Detections.from_ultralytics(results[0])
             # Ensure required attributes exist
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

# --- Example Usage (Optional: for testing this file directly) ---
# if __name__ == '__main__':
#     test_model_name = 'yolov8n.pt' # Use nano for quicker testing
#     yolo_model = load_yolo_model(test_model_name)
#     if yolo_model:
#         # Create a dummy black frame for testing
#         dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
#         print("\nTesting detection on dummy frame...")
#         detections = detect_objects(yolo_model, dummy_frame, confidence_threshold=0.25)
#         print(f"Detected {len(detections)} objects (class 0, conf > 0.25).")
#         if len(detections) > 0:
#             print("Sample detection data:")
#             print(f"  Box: {detections.xyxy[0]}")
#             print(f"  Conf: {detections.confidence[0]:.2f}")
#             print(f"  Class: {detections.class_id[0]}")
#     else:
#         print("Model could not be loaded for testing.")