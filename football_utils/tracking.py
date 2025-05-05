# football_utils/tracking.py

import cv2
import numpy as np

# Note: These functions are likely NOT used by the current video_processing.py,
# which uses supervision.ByteTrack. They are kept here for potential alternative uses.

def initialize_tracker(frame, bbox):
    """
    Initialize a standard OpenCV tracker (e.g., KCF) for a given bounding box.
    """
    # Example using KCF tracker (other options: MIL, CSRT, MOSSE - legacy)
    # Note: CSRT is often better but slower. KCF is faster but can drift.
    tracker = cv2.TrackerKCF_create() # Or cv2.TrackerCSRT_create()
    # bbox format for OpenCV tracker init is (x, y, width, height)
    x1, y1, x2, y2 = map(int, bbox)
    bbox_cv = (x1, y1, x2 - x1, y2 - y1)
    try:
        success = tracker.init(frame, bbox_cv)
        if success:
            return tracker
        else:
            print("[TRACKING] OpenCV Tracker initialization failed.")
            return None
    except Exception as e:
        print(f"[TRACKING] Error initializing OpenCV tracker: {e}")
        return None

def track_objects(frame, tracker):
    """
    Update tracker for the new frame.
    Returns success (bool) and updated bbox (x, y, w, h).
    """
    if tracker is None:
        return False, None
    try:
        success, bbox = tracker.update(frame)
        return success, bbox # bbox is (x, y, w, h)
    except Exception as e:
        print(f"[TRACKING] Error updating OpenCV tracker: {e}")
        return False, None

def calculate_optical_flow(prev_frame, curr_frame):
    """
    Calculate the dense optical flow (Farneback) between two frames.
    Returns the flow field (a 2-channel array for dx, dy).
    """
    if prev_frame is None or curr_frame is None:
        return None
    try:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                              0.5, 3, 15, 3, 5, 1.2, 0)
        return flow
    except Exception as e:
        print(f"[TRACKING] Error calculating optical flow: {e}")
        return None

def apply_perspective_transform(frame, src_points, dst_points, output_size):
    """
    Apply perspective transformation to get a top-down view.
    src_points: 4 points (NumPy array [[x1,y1],[x2,y2]...]) on the input frame.
    dst_points: Corresponding 4 points on the output plane.
    output_size: (width, height) tuple for the output warped image.
    """
    if frame is None or src_points is None or dst_points is None or output_size is None:
        return None
    try:
        matrix = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
        transformed = cv2.warpPerspective(frame, matrix, output_size)
        return transformed
    except Exception as e:
        print(f"[TRACKING] Error applying perspective transform: {e}")
        return None