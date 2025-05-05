# football_utils/team_assignment.py

import cv2
import numpy as np
from sklearn.cluster import KMeans
import traceback
from collections import defaultdict, deque

# --- Configuration ---
MIN_PLAYER_HEIGHT_PIXELS = 40  # Minimum height for players (Ensure consistency with video_processing.py)

# --- Core Helper Functions for Color Analysis ---

def extract_jersey_roi(frame, detection_box):
    """Extract upper torso region for jersey color analysis."""
    # If detection_box is from supervision.Detections, it's likely already [x1, y1, x2, y2]
    if not isinstance(detection_box, (list, tuple, np.ndarray)) or len(detection_box) != 4:
         print(f"[TEAM_ASSIGN] Invalid detection_box format: {detection_box}")
         return None

    x1, y1, x2, y2 = map(int, detection_box)
    height = y2 - y1
    width = x2 - x1

    # Basic size check (use config from video_processing or define here)
    if height < MIN_PLAYER_HEIGHT_PIXELS or width < 5:
        return None

    # Calculate torso region focusing on jersey area
    torso_y1 = int(y1 + height * 0.15)  # Skip head approx
    torso_y2 = int(y1 + height * 0.45)  # Upper torso only
    torso_x1 = int(x1 + width * 0.20)   # Inner torso width
    torso_x2 = int(x2 - width * 0.20)

    # Ensure coordinates are valid and within image bounds
    frame_h, frame_w = frame.shape[:2]
    torso_y1 = max(0, min(torso_y1, frame_h - 1))
    torso_y2 = max(torso_y1 + 5, min(torso_y2, frame_h - 1)) # Ensure min height
    torso_x1 = max(0, min(torso_x1, frame_w - 1))
    torso_x2 = max(torso_x1 + 3, min(torso_x2, frame_w - 1)) # Ensure min width

    # Check if calculated ROI dimensions are valid
    if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
        return None

    # Extract ROI
    try:
        jersey_roi = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        # Final check on extracted ROI size
        if jersey_roi.size < 30: # Need at least ~10 pixels (3 channels)
            return None
        return jersey_roi
    except Exception as e:
        # print(f"[TEAM_ASSIGN] Error extracting jersey ROI: {e}") # Optional logging
        return None

def is_referee(roi):
    """Detect referees based on dark jersey color."""
    if roi is None or roi.size < 15: # Need at least 5 pixels
        return False

    try:
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Lower Max Value (brightness) threshold for very dark uniforms
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 50])  # Very dark threshold

        dark_mask = cv2.inRange(hsv_roi, lower_dark, upper_dark)
        dark_ratio = cv2.countNonZero(dark_mask) / (roi.shape[0] * roi.shape[1])

        # Return True if enough dark pixels (ratio > 70%)
        is_ref = dark_ratio > 0.70
        return is_ref
    except Exception as e:
        # print(f"[TEAM_ASSIGN] Error in is_referee check: {e}") # Optional logging
        return False

def get_representative_color(roi, k=1):
    """Get dominant color from jersey region using K-means."""
    if roi is None or roi.size < 15*3: # Need at least 15 pixels
        return None

    try:
        # Convert to HSV for better color analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv_roi.reshape(-1, 3)

        # Filter out very dark/bright pixels and low saturation pixels
        valid_mask = (
            (hsv_pixels[:, 1] > 20) &   # Min saturation
            (hsv_pixels[:, 2] > 30) &   # Min brightness
            (hsv_pixels[:, 2] < 235)    # Max brightness
        )

        # If not enough valid pixels after filtering, try fallback
        if np.sum(valid_mask) < 10:
            # Fall back to simple BGR average filtering (less robust)
            pixels = roi.reshape(-1, 3).astype(np.float32)
            # Filter BGR pixels that are not too dark or too light
            dark_light_mask = (np.all(pixels > [20, 20, 20], axis=1)) & \
                              (np.all(pixels < [240, 240, 240], axis=1))
            filtered_pixels = pixels[dark_light_mask]
            if len(filtered_pixels) < 5:
                return None # Not enough pixels even with fallback
            # Return average of filtered BGR pixels
            return tuple(np.mean(filtered_pixels, axis=0).astype(int))

        # Use HSV-filtered pixels, convert back to BGR for K-Means
        filtered_hsv = hsv_pixels[valid_mask]
        filtered_bgr = cv2.cvtColor(filtered_hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)

        if len(filtered_bgr) < k: # Need at least k pixels for k-means
             # If fewer pixels than clusters, return average
             return tuple(np.mean(filtered_bgr, axis=0).astype(int))

        # Apply K-means on the filtered BGR colors
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42) # Use 'auto' for n_init
        kmeans.fit(filtered_bgr)

        # Return the first cluster center (most dominant color when k=1)
        dominant_color = tuple(kmeans.cluster_centers_[0].astype(int))
        return dominant_color

    except Exception as e:
        # print(f"[TEAM_ASSIGN] Error in get_representative_color: {e}") # Optional logging
        # Fall back to simple average if clustering fails unexpectedly
        try:
             pixels = roi.reshape(-1, 3)
             return tuple(np.mean(pixels, axis=0).astype(int))
        except: return None # Final fallback

def get_smoothed_color(color_deque):
    """Get median color from history (deque) to reduce noise."""
    if not color_deque or len(color_deque) < 2:
        # Return last color if only one, or None if empty
        return color_deque[0] if color_deque else None

    colors_list = list(color_deque)
    colors_np = np.array(colors_list)

    try:
        # Use median for robustness against outliers
        median_b = int(np.median(colors_np[:, 0]))
        median_g = int(np.median(colors_np[:, 1]))
        median_r = int(np.median(colors_np[:, 2]))
        return (median_b, median_g, median_r)
    except IndexError:
        # Should not happen with the initial length check, but safeguard
        return colors_list[-1] if colors_list else None # Return last element if median fails
    except Exception as e:
        # print(f"[TEAM_ASSIGN] Error smoothing color: {e}") # Optional logging
        return colors_list[-1] if colors_list else None

def extract_color_histogram(roi, bins=8):
    """Extract color histogram features for team classification."""
    if roi is None or roi.size < 15: # Need min pixels
        return None

    try:
        # Resize to normalized size for consistent features
        # Using INTER_AREA is good for downscaling
        roi_resized = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)

        # Extract BGR histograms
        hist_features = []
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([roi_resized], [i], None, [bins], [0, 256])
            # Normalize histogram
            cv2.normalize(hist, hist)
            hist_features.extend(hist.flatten())

        return np.array(hist_features)
    except Exception as e:
        # print(f"[TEAM_ASSIGN] Error extracting histogram: {e}") # Optional logging
        return None