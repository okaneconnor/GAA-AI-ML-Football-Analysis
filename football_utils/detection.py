import torch
import cv2
import numpy as np
from ultralytics import YOLO # Import the YOLO class
import os
import traceback # For better error reporting

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
# Define the desired model identifier. YOLO() will handle download if the file
# doesn't exist in the directory specified (or default cache).
# Ensure the DIRECTORY (/content/models/) exists, which Cell 3 does.
model_identifier = "/content/models/yolo11n.pt"
# Or you could just use the name, and let ultralytics manage cache:
# model_identifier = "yolo11n.pt"

print(f"[DETECTION] Attempting to load/download model: {model_identifier}")
model = None # Initialize model to None
try:
    # *** REMOVED the os.path.exists() check ***
    # Let YOLO() handle download automatically if the file doesn't exist
    model = YOLO(model_identifier)
    model.to(device) # Move model to the selected device
    print(f"[DETECTION] Model '{model_identifier}' loaded successfully onto {device}.")
    # Verify model names attribute is loaded
    if not hasattr(model, 'names') or not model.names:
         print("[DETECTION] Warning: model.names not found after loading.")
    # else:
    #      print(f"[DETECTION] Model class names: {model.names}") # Optional: print names

except Exception as e:
    print(f"[DETECTION] Error loading model '{model_identifier}': {e}")
    print(traceback.format_exc())
    # model remains None

# --- Detect Objects Function ---
def detect_objects(frame):
    """
    Perform object detection on a frame using the loaded YOLO model.
    Returns detection results as a list of dictionaries compatible with the old format.
    """
    detections = [] # Initialize empty list

    if model is None:
        # This message should now only appear if the YOLO() call above truly failed
        print("[DETECTION] Error: Model is not loaded, cannot perform detection.")
        return detections # Return empty list

    # Check if model has class names (needed for formatting output)
    if not hasattr(model, 'names') or not model.names:
        print("[DETECTION] Error: Model loaded but class names (model.names) are missing.")
        return detections

    # Perform inference using the ultralytics model
    try:
        results = model.predict(source=frame, device=device, verbose=False)
    except Exception as e:
        print(f"[DETECTION] Error during model prediction: {e}")
        print(traceback.format_exc())
        return detections # Return empty list on prediction error

    # Process results
    if results and results[0].boxes:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            try:
                box_data = boxes.data[i].cpu().numpy()
                xmin, ymin, xmax, ymax, conf, cls_id_float = box_data
                cls_id = int(cls_id_float) # Convert class ID to integer

                # Get class name from model metadata, handle potential IndexError
                if 0 <= cls_id < len(model.names):
                     class_name = model.names[cls_id]
                else:
                     print(f"[DETECTION] Warning: Invalid class ID {cls_id} detected.")
                     class_name = "unknown"

                detections.append({
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'confidence': conf,
                    'class': cls_id,
                    'name': class_name
                })
            except IndexError:
                 print(f"[DETECTION] Error accessing detection data at index {i}. Box data: {boxes.data[i]}")
            except Exception as proc_err:
                 print(f"[DETECTION] Error processing detection at index {i}: {proc_err}")


    return detections