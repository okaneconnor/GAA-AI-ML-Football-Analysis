# football_utils/video_processing.py

import cv2
import os
import time
import numpy as np
import traceback
from collections import defaultdict, deque
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import supervision as sv

# Import necessary functions from other modules
from football_utils.detection import load_yolo_model, detect_objects
# from football_utils.tracking import calculate_optical_flow, apply_perspective_transform # Not used in current main loop
from football_utils.team_assignment import (
    extract_jersey_roi,
    is_referee,
    get_representative_color,
    get_smoothed_color,
    extract_color_histogram
)

# --- Configuration ---
VIDEO_PROCESSING_FRAME_INTERVAL = 1 # Process every frame
YOLO_MODEL_NAME = 'yolov8m.pt'      # Using YOLOv8 Medium model
CONFIDENCE_THRESHOLD = 0.30         # YOLO confidence threshold
MOVEMENT_WINDOW_SIZE = 15           # Frames for movement analysis
MOVEMENT_STATIC_THRESHOLD = 15      # Pixel distance threshold for static
TEAM_RECLUSTER_INTERVAL = 60        # Re-run K-Means every 60 frames

# Filtering Settings
MIN_PLAYER_HEIGHT_PIXELS = 40       # Minimum height for players
MAX_PLAYER_BOTTOM_Y_PERCENT = 0.86  # Ignore bottom 14% of frame (less critical if field region used)
MAX_COLOR_DISTANCE_THRESHOLD = 55   # K-Means assignment distance threshold
MIN_COLOR_SATURATION = 0.15         # Min saturation for valid team color
COLOR_CONSISTENCY_THRESHOLD = 40    # Max deviation from team avg color

# Fence Line Detection Parameters
USE_HORIZONTAL_LINE_FILTERING = True
FENCE_DETECTION_METHOD = "position_based" # "position_based" or "line_detection"
DISPLAY_FENCE_LINES = False # Set True to draw lines on output video

# Fence positions (relative Y, height, buffer above, buffer below)
FENCE_POSITIONS = [
    {'y_percent': 0.39, 'height_percent': 0.05, 'buffer_above': 0.00, 'buffer_below': 0.05},
    {'y_percent': 0.45, 'height_percent': 0.05, 'buffer_above': 0.00, 'buffer_below': 0.05},
]

# Field region definition (relative coordinates)
FIELD_REGION = {
    'top': 0.45,
    'bottom': 0.90,
    'left': 0.05,
    'right': 0.95
}

# Display Colors (BGR Format)
DISPLAY_RED = (0, 0, 255)
DISPLAY_BLUE = (255, 0, 0)
DISPLAY_GREY = (128, 128, 128)
DISPLAY_WHITE = (255, 255, 255)

# --- Helper Functions (Moved/Consolidated Here) ---

def color_distance(c1, c2):
    """Calculate Euclidean distance between two BGR colors."""
    if c1 is None or c2 is None:
        return float('inf')
    # Ensure input are numpy arrays for calculation
    c1 = np.asarray(c1, dtype=np.float32)
    c2 = np.asarray(c2, dtype=np.float32)
    # Calculate the Euclidean distance
    return np.linalg.norm(c1 - c2)

def detect_horizontal_lines(frame, threshold=100, min_line_length=100):
    """Detect horizontal lines (like fences) using Hough Transform."""
    if not USE_HORIZONTAL_LINE_FILTERING or FENCE_DETECTION_METHOD != "line_detection":
        return [] # Only run if specifically enabled

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use HoughLinesP to detect line segments
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold,
                                minLineLength=min_line_length, maxLineGap=10)

        # Filter for mostly horizontal lines
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if the vertical change is small
                if abs(y2 - y1) < 20: # Allow small slope
                    horizontal_lines.append((x1, y1, x2, y2))
        return horizontal_lines
    except Exception as e:
        # print(f"[VIDEO_PROC] Error detecting lines: {e}") # Optional logging
        return []

def is_near_fence_line(box, frame_height, frame_width, fence_positions=None, horizontal_lines=None):
    """Check if detection box bottom is near a configured or detected fence line."""
    if not USE_HORIZONTAL_LINE_FILTERING:
        return False

    x1, y1, x2, y2 = map(int, box)

    # Method 1: Position-based check (using FENCE_POSITIONS)
    if FENCE_DETECTION_METHOD == "position_based" and fence_positions:
        for fence in fence_positions:
            # Calculate absolute Y coordinates for the fence zone
            fence_y_abs = fence['y_percent'] * frame_height
            fence_height_abs = fence['height_percent'] * frame_height
            buffer_above_abs = fence['buffer_above'] * frame_height
            buffer_below_abs = fence['buffer_below'] * frame_height

            # Check if the bottom of the box (y2) falls within the fence zone
            if (y2 >= fence_y_abs - buffer_above_abs and
                y2 <= fence_y_abs + fence_height_abs + buffer_below_abs):
                return True # Near this fence position

    # Method 2: Line-based check (using detected horizontal_lines)
    elif FENCE_DETECTION_METHOD == "line_detection" and horizontal_lines:
        box_bottom_y = y2
        for x1_line, y1_line, x2_line, y2_line in horizontal_lines:
            # Average Y position of the detected line segment
            line_y_avg = (y1_line + y2_line) / 2
            # Check if the box bottom is close vertically to the line
            if abs(box_bottom_y - line_y_avg) < 20: # Proximity threshold
                return True # Near this detected line

    return False # Not near any configured or detected fence

def is_in_field_region(box, frame_height, frame_width, field_region):
    """Check if the center of the detection box is within the defined field region."""
    x1, y1, x2, y2 = box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    # Check against relative boundaries defined in FIELD_REGION
    in_field = (
        box_center_y > field_region['top'] * frame_height and
        box_center_y < field_region['bottom'] * frame_height and
        box_center_x > field_region['left'] * frame_width and
        box_center_x < field_region['right'] * frame_width
    )
    return in_field

def has_player_movement_pattern(track_history_deque, min_window=5):
    """Check track history (a deque) for player-like movement."""
    if len(track_history_deque) < min_window:
        return True  # Assume player if not enough history

    # Calculate velocities between consecutive points in the deque
    velocities = []
    track_list = list(track_history_deque) # Convert deque to list for indexing
    for i in range(1, len(track_list)):
        p1 = np.array(track_list[i-1])
        p2 = np.array(track_list[i])
        # Calculate Euclidean distance (velocity magnitude)
        velocity = np.linalg.norm(p2 - p1)
        velocities.append(velocity)

    if not velocities: # Handle case where only one point exists
        return True

    # Analyze velocity statistics
    avg_velocity = np.mean(velocities)
    velocity_std = np.std(velocities)

    # Define criteria for player-like movement (adjust as needed)
    # Not too static (avg_velocity > 0.3)
    # Not excessively fast/erratic (avg_velocity < 35)
    # Relatively consistent speed (velocity_std < 18)
    is_player_pattern = (0.3 < avg_velocity < 35) and (velocity_std < 18)
    return is_player_pattern

def is_valid_team_color(color, team_centers, color_threshold, team_colors_array):
    """Check if a color likely belongs to a team based on cluster centers and distribution."""
    if color is None or len(team_centers) != 2:
        return False, -1 # Cannot validate if color or centers missing

    # Calculate distance to both team cluster centers
    dist0 = color_distance(color, team_centers[0])
    dist1 = color_distance(color, team_centers[1])
    min_dist = min(dist0, dist1)
    closest_team_idx = 0 if dist0 <= dist1 else 1 # 0 or 1

    # --- Basic Validation ---
    # 1. Check distance to the *closest* cluster center
    if min_dist > color_threshold:
        return False, -1 # Too far from either known team color

    # 2. Check color saturation (avoid assigning grey/white/black)
    try:
        color_bgr = np.uint8([[color]]) # Reshape for cvtColor
        color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
        saturation = color_hsv[1] / 255.0 # Normalize saturation to 0-1
        if saturation < MIN_COLOR_SATURATION:
            return False, -1 # Color is not saturated enough (likely not a team color)
    except Exception:
         return False, -1 # Error during HSV conversion

    # --- Distribution Check (if enough samples exist) ---
    # Check if the color is consistent with other colors assigned to the *closest* team
    team_samples = team_colors_array[closest_team_idx]
    if team_samples and len(team_samples) >= 3: # Need min samples for meaningful mean/std
        team_colors_np = np.array(team_samples)
        color_np = np.array(color)
        team_mean = np.mean(team_colors_np, axis=0)

        # Calculate how far the current color is from the mean of its potential team
        color_deviation = np.linalg.norm(color_np - team_mean)

        if color_deviation > COLOR_CONSISTENCY_THRESHOLD:
            return False, -1 # Color deviates too much from others in the same team

    # If all checks pass, the color is considered valid for the closest team
    return True, closest_team_idx


# --- Team Assignment & Visualization Function (Handles Classifier) ---
def assign_and_visualize_teams(
    frame, detections_with_tracking, track_history, player_colors_history,
    team_assignments, is_static, current_team_centers,
    cluster_to_display_color, cluster_to_display_label, is_player_referee,
    max_color_distance_threshold, team_colors_array, fence_positions, horizontal_lines,
    team_classifier=None, classifier_trained=False # New classifier args
    ):
    """Draws ellipses and labels, using classifier first if available."""
    output_frame = frame.copy()

    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]

    # Optional: Draw fence lines for debugging
    if DISPLAY_FENCE_LINES:
        if USE_HORIZONTAL_LINE_FILTERING and FENCE_DETECTION_METHOD == "position_based":
            for fence in fence_positions:
                fence_y = int(fence['y_percent'] * frame_h)
                cv2.line(output_frame, (0, fence_y), (frame_w, fence_y), (0, 255, 255), 1)
        if USE_HORIZONTAL_LINE_FILTERING and FENCE_DETECTION_METHOD == "line_detection":
             for x1l, y1l, x2l, y2l in horizontal_lines:
                 cv2.line(output_frame, (x1l, y1l), (x2l, y2l), (0, 255, 255), 1)

    # Process each tracked detection
    if hasattr(detections_with_tracking, 'tracker_id') and detections_with_tracking.tracker_id is not None:
        for i, box in enumerate(detections_with_tracking.xyxy):
            track_id = detections_with_tracking.tracker_id[i]
            if track_id is None: continue
            track_id = int(track_id)

            # Default display settings
            display_label = f"ID:{track_id}"; display_color = DISPLAY_WHITE

            # --- Filtering Logic ---
            x1, y1, x2, y2 = map(int, box)
            height = y2 - y1
            box_center_y = (y1 + y2) / 2

            # 1. Filter spectators near fence
            if is_near_fence_line(box, frame_h, frame_w, fence_positions, horizontal_lines):
                continue # Skip visualization completely

            # 2. Filter people clearly outside/above field region
            in_field = is_in_field_region(box, frame_h, frame_w, FIELD_REGION)
            if not in_field and box_center_y < FIELD_REGION['top'] * frame_h:
                continue # Skip visualization

            # 3. Handle known static objects or referees
            if track_id in is_static:
                display_label = f"ID:{track_id} Static"; display_color = DISPLAY_GREY
            elif track_id in is_player_referee:
                display_label = f"ID:{track_id} Referee"; display_color = DISPLAY_GREY
            else: # Potential team player
                # 4. Check movement pattern
                is_moving_like_player = has_player_movement_pattern(track_history[track_id], min_window=3)
                if not is_moving_like_player:
                    display_label = f"ID:{track_id} Filtered"; display_color = DISPLAY_GREY
                else:
                    # --- Assignment Logic (Classifier First) ---
                    assigned_label = team_assignments.get(track_id, "Filtered") # Get current/previous label

                    # Attempt classification using Logistic Regression model if trained
                    classification_success = False
                    if classifier_trained and team_classifier is not None:
                        jersey_roi = extract_jersey_roi(frame, box)
                        if jersey_roi is not None:
                            hist_features = extract_color_histogram(jersey_roi)
                            if hist_features is not None:
                                try:
                                    team_pred_proba = team_classifier.predict_proba([hist_features])[0]
                                    max_proba = np.max(team_pred_proba)
                                    # Only use classifier if confident enough (e.g., > 70%)
                                    if max_proba > 0.70:
                                        team_pred_label = team_classifier.predict([hist_features])[0]
                                        if team_pred_label == 0: # Mapped to Blue Team
                                            assigned_label = "Blue Team"
                                            classification_success = True
                                        elif team_pred_label == 1: # Mapped to Red Team
                                            assigned_label = "Red Team"
                                            classification_success = True
                                        # else: # Predicted as 'Other' - keep as Filtered
                                        #     assigned_label = "Filtered"
                                        #     classification_success = True # We made a decision
                                        # Update main assignment dictionary
                                        team_assignments[track_id] = assigned_label

                                except Exception:
                                    pass # Ignore classifier errors, fallback to K-Means

                    # Fallback/Refinement using K-Means cluster distance if classifier failed/unsure
                    if not classification_success and assigned_label == "Filtered" and len(current_team_centers) == 2:
                        smoothed_color = get_smoothed_color(player_colors_history[track_id])
                        if smoothed_color:
                            is_valid, assigned_cluster = is_valid_team_color(
                                smoothed_color,
                                current_team_centers,
                                max_color_distance_threshold,
                                team_colors_array
                            )
                            if is_valid and assigned_cluster >= 0:
                                potential_label = cluster_to_display_label.get(assigned_cluster)
                                if potential_label:
                                    assigned_label = potential_label
                                    team_assignments[track_id] = assigned_label # Update assignment
                            # Optional: Force assignment for large players clearly in field?
                            # elif in_field and height >= MIN_PLAYER_HEIGHT_PIXELS * 1.2:
                            #     dist0 = color_distance(smoothed_color, current_team_centers[0])
                            #     dist1 = color_distance(smoothed_color, current_team_centers[1])
                            #     closest_team = 0 if dist0 <= dist1 else 1
                            #     assigned_label = cluster_to_display_label.get(closest_team, "Filtered")
                            #     team_assignments[track_id] = assigned_label

                    # Final display color based on assigned_label
                    if assigned_label == "Red Team": display_color = DISPLAY_RED
                    elif assigned_label == "Blue Team": display_color = DISPLAY_BLUE
                    # elif assigned_label == "Team 0": display_color = DISPLAY_BLUE # Keep consistent mapping
                    else: display_color = DISPLAY_GREY # Filtered or other

                    display_label = f"ID:{track_id} {assigned_label}"

            # --- Drawing Logic ---
            # Draw ellipse at the bottom center of the box
            center_x = int((box[0] + box[2]) / 2)
            bottom_y = int(box[3])
            ellipse_axes = (20, 6) # Width, Height of ellipse

            # Draw the ellipse
            cv2.ellipse(output_frame, (center_x, bottom_y), ellipse_axes, 0, 0, 360, display_color, 2)
            # Draw the text label below the ellipse
            cv2.putText(output_frame, display_label,
                        (center_x - 30, bottom_y + ellipse_axes[1] + 15), # Position below ellipse
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_color, 1)

    return output_frame


# ==============================================================
#           --- Main Video Processing Function ---
# ==============================================================

def process_video(input_video_path, output_folder):
    """
    Processes the input video for player detection, tracking, and team assignment.

    Args:
        input_video_path (str): Path to the input video file.
        output_folder (str): Folder where the output video will be saved.

    Returns:
        tuple: (output_path, status_message)
               output_path (str or None): Path to the processed video, or None on failure.
               status_message (str): A message indicating success or error.
    """
    log_prefix = "[PROCESS_VIDEO] "
    print(f"{log_prefix}Starting processing for: {input_video_path}")
    print(f"{log_prefix}Using YOLO Model: {YOLO_MODEL_NAME}, Confidence: {CONFIDENCE_THRESHOLD}")
    start_total_time = time.time()

    # --- Initialization ---
    model = None
    cap = None
    video_writer = None
    processed_frame_count = 0
    frame_num = -1
    output_video_path = None # Initialize output path

    # Data structures for tracking and analysis
    track_history = defaultdict(lambda: deque(maxlen=MOVEMENT_WINDOW_SIZE))
    player_colors_history = defaultdict(lambda: deque(maxlen=10)) # History of representative colors
    team_assignments = {} # Stores current team label for each track_id
    is_static = set() # Set of track_ids considered static
    is_player_referee = {} # track_id -> True if identified as referee
    current_team_centers = [] # Stores the 2 K-Means cluster centers
    team_colors_array = [[], []] # Stores BGR colors assigned to team 0 and team 1 for validation
    # Mapping from K-Means cluster index (0 or 1) to display color/label
    cluster_to_display_color = {0: DISPLAY_BLUE, 1: DISPLAY_RED}
    cluster_to_display_label = {0: "Blue Team", 1: "Red Team"}

    # Team classifier data (using Logistic Regression)
    team_training_data = [] # Stores tuples of (histogram_features, team_label [0 or 1])
    team_classifier = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42) # 'ovr' can be simpler
    classifier_trained = False

    # --- Main Try Block for resource handling ---
    try:
        # 1. Load Model
        model = load_yolo_model(YOLO_MODEL_NAME)
        if model is None:
            raise RuntimeError("Failed to load YOLO model.")

        # 2. Setup Video Capture
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps_read) if fps_read > 0 else 30 # Default to 30 FPS if read fails
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{log_prefix}Video Info: {frame_width}x{frame_height}, {fps} FPS, ~{total_frames} Frames")

        # 3. Initialize Tracker
        byte_tracker = sv.ByteTrack(frame_rate=fps) # Use supervision's ByteTrack
        print(f"{log_prefix}ByteTrack initialized.")

        # 4. Setup Video Writer
        output_filename = f"{os.path.splitext(os.path.basename(input_video_path))[0]}_output_v8.mp4"
        output_video_path = os.path.join(output_folder, output_filename)
        os.makedirs(output_folder, exist_ok=True) # Ensure output folder exists
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            raise IOError(f"Failed to open VideoWriter for path: {output_video_path}")
        print(f"{log_prefix}Output video writer ready: {output_video_path}")

        print(f"{log_prefix}Fence positions configured at Y% {[f['y_percent'] for f in FENCE_POSITIONS]}")
        print(f"{log_prefix}Field region set Y%=({FIELD_REGION['top']}-{FIELD_REGION['bottom']}), X%=({FIELD_REGION['left']}-{FIELD_REGION['right']})")
        print(f"{log_prefix}Fence lines will be {'shown' if DISPLAY_FENCE_LINES else 'hidden'}")

        # --- Frame Processing Loop ---
        while True:
            frame_num += 1
            ret, frame = cap.read()
            if not ret:
                print(f"\n{log_prefix}End of video stream reached.")
                break # Exit loop if no more frames

            # Skip frames based on interval if needed
            if frame_num % VIDEO_PROCESSING_FRAME_INTERVAL != 0:
                continue

            loop_start_time = time.time()

            # Detect horizontal lines (only if method is 'line_detection')
            horizontal_lines = detect_horizontal_lines(frame)

            # 1. Object Detection (using function from detection.py)
            # Returns sv.Detections object already filtered for class 0 and confidence
            detections_sv = detect_objects(model, frame, confidence_threshold=CONFIDENCE_THRESHOLD)

            # 1b. Filter detections based on size and fence proximity BEFORE tracking
            filtered_indices = []
            for i, box in enumerate(detections_sv.xyxy):
                height = box[3] - box[1]
                # Apply size filter
                if height < MIN_PLAYER_HEIGHT_PIXELS:
                    continue
                # Apply fence filter
                if is_near_fence_line(box, frame_height, frame_width, FENCE_POSITIONS, horizontal_lines):
                    continue
                # If passes filters, keep index
                filtered_indices.append(i)

            # Create new Detections object with only filtered items
            detections_for_tracking = detections_sv[filtered_indices]

            # 2. Tracking
            try:
                # Update tracker with the filtered detections
                detections_with_tracking = byte_tracker.update_with_detections(detections=detections_for_tracking)
            except Exception as e_trk:
                print(f"{log_prefix}Frame {frame_num}: Error during tracker update: {e_trk}")
                detections_with_tracking = sv.Detections.empty() # Continue with empty if error

            # 3. Movement Analysis & Update Colors/Features
            current_moving_tracker_ids = set()
            current_frame_track_ids = set()

            if hasattr(detections_with_tracking, 'tracker_id') and detections_with_tracking.tracker_id is not None:
                # Get anchor points (centers) for track history
                centers = detections_with_tracking.get_anchors_coordinates(anchor=sv.Position.CENTER)
                valid_track_indices = [i for i, tid in enumerate(detections_with_tracking.tracker_id) if tid is not None]

                for i in valid_track_indices:
                    track_id = int(detections_with_tracking.tracker_id[i])
                    current_frame_track_ids.add(track_id)
                    box = detections_with_tracking.xyxy[i]
                    center_point = tuple(map(int, centers[i]))

                    # Update track history (position)
                    track_history[track_id].append(center_point)

                    # Check if track is static
                    is_track_static = False
                    if len(track_history[track_id]) >= MOVEMENT_WINDOW_SIZE:
                        # Calculate displacement over the window
                        displacement = color_distance(track_history[track_id][0], track_history[track_id][-1])
                        if displacement < MOVEMENT_STATIC_THRESHOLD:
                            is_static.add(track_id) # Mark as static
                            is_track_static = True
                        else:
                            is_static.discard(track_id) # Mark as moving
                            current_moving_tracker_ids.add(track_id)
                    elif track_id not in is_static: # If not marked static yet
                         current_moving_tracker_ids.add(track_id)

                    # Only update color/features if moving and in field region
                    if not is_track_static and is_in_field_region(box, frame_height, frame_width, FIELD_REGION):
                        roi = extract_jersey_roi(frame, box)
                        if roi is not None:
                            # Check if referee
                            if is_referee(roi):
                                is_player_referee[track_id] = True
                                team_assignments[track_id] = "Referee" # Assign label
                                # Optionally add histogram features with a 'referee' label (e.g., 2)
                                # hist_features = extract_color_histogram(roi)
                                # if hist_features is not None: team_training_data.append((hist_features, 2))
                            elif track_id not in is_player_referee:
                                # Get representative color for K-Means
                                rep_color = get_representative_color(roi)
                                if rep_color is not None:
                                    player_colors_history[track_id].append(rep_color)

                                # Get histogram features for classifier
                                hist_features = extract_color_histogram(roi)
                                if hist_features is not None:
                                    # Add to training data IF we have a confident assignment from K-Means
                                    current_label = team_assignments.get(track_id)
                                    label_to_int = -1
                                    if current_label == "Blue Team": label_to_int = 0
                                    elif current_label == "Red Team": label_to_int = 1

                                    # Only add if assigned to Team A or B
                                    if label_to_int != -1:
                                         team_training_data.append((hist_features, label_to_int))
                                         # Optional: Limit training data size
                                         # if len(team_training_data) > 500:
                                         #     team_training_data.pop(0)


            # Remove data for tracks that disappeared in this frame
            all_known_ids = set(track_history.keys())
            disappeared_ids = all_known_ids - current_frame_track_ids
            for old_id in disappeared_ids:
                track_history.pop(old_id, None)
                player_colors_history.pop(old_id, None)
                team_assignments.pop(old_id, None)
                is_static.discard(old_id)
                is_player_referee.pop(old_id, None)

            # 4. Periodic Team Re-clustering and Classifier Training
            if (frame_num % TEAM_RECLUSTER_INTERVAL == 0 or frame_num < 30) and len(current_moving_tracker_ids) > 0:
                print(f"\n--- Frame {frame_num}: Re-clustering & Classifier Update ---")
                colors_for_clustering = []
                tracks_for_clustering = []

                # Reset team color arrays used for validation
                team_colors_array = [[], []]

                # Gather smoothed colors from currently valid players
                for tid in current_moving_tracker_ids:
                    # Ensure track is not referee, not static, and moving like a player
                    if (tid not in is_player_referee and
                        tid not in is_static and
                        has_player_movement_pattern(track_history[tid], min_window=3)):
                        # Check if track is currently visible and in field
                        track_found_in_frame = False
                        for i, tid_check in enumerate(detections_with_tracking.tracker_id or []):
                             if tid_check is not None and int(tid_check) == tid:
                                 box = detections_with_tracking.xyxy[i]
                                 if is_in_field_region(box, frame_height, frame_width, FIELD_REGION):
                                      track_found_in_frame = True
                                 break
                        if track_found_in_frame:
                            smoothed_color = get_smoothed_color(player_colors_history[tid])
                            if smoothed_color is not None:
                                colors_for_clustering.append(smoothed_color)
                                tracks_for_clustering.append(tid)


                print(f"Clustering {len(colors_for_clustering)} smoothed colors from valid tracks...")

                # Perform K-Means clustering if enough samples
                if len(colors_for_clustering) >= 4:
                    cluster_colors_np = np.array(colors_for_clustering).astype(np.float32)
                    unique_colors, counts = np.unique(cluster_colors_np, axis=0, return_counts=True)

                    if len(unique_colors) >= 2: # Need at least 2 unique colors
                        try:
                            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                            kmeans.fit(cluster_colors_np)
                            new_team_centers = [tuple(map(int, center)) for center in kmeans.cluster_centers_]
                            print(f"K-Means OK. New Centers: {new_team_centers}")

                            # Only update centers if they seem valid (e.g., not too close)
                            if color_distance(new_team_centers[0], new_team_centers[1]) > 30:
                                current_team_centers = new_team_centers
                            else:
                                print("Warning: K-Means centers are very close, not updating.")


                            # Determine Red/Blue mapping based on R/B ratio of centers
                            if current_team_centers: # Check if we have valid centers
                                c0, c1 = current_team_centers[0], current_team_centers[1]
                                # Simple ratio: Red channel / (Blue channel + 1) to avoid division by zero
                                ratio0 = c0[2] / (c0[0] + 1.0)
                                ratio1 = c1[2] / (c1[0] + 1.0)
                                if ratio0 > ratio1: # Center 0 is more 'Red'
                                    cluster_to_display_color = {0: DISPLAY_RED, 1: DISPLAY_BLUE}
                                    cluster_to_display_label = {0: "Red Team", 1: "Blue Team"}
                                else: # Center 1 is more 'Red' (or equal)
                                    cluster_to_display_color = {0: DISPLAY_BLUE, 1: DISPLAY_RED}
                                    cluster_to_display_label = {0: "Blue Team", 1: "Red Team"}
                                print(f"Team mapping updated: Cluster 0 -> {cluster_to_display_label[0]}, Cluster 1 -> {cluster_to_display_label[1]}")


                            # Re-assign all players based on new K-Means centers and populate team_colors_array
                            assigned_count = 0
                            if current_team_centers: # Proceed only if centers are defined
                                kmeans_labels = kmeans.predict(cluster_colors_np) # Get cluster for each color used

                                # Populate team_colors_array with high-confidence samples first
                                for i, tid in enumerate(tracks_for_clustering):
                                    assigned_cluster = kmeans_labels[i]
                                    smoothed_color = colors_for_clustering[i] # Already calculated
                                    # Check distance to the assigned cluster center
                                    dist_to_center = color_distance(smoothed_color, current_team_centers[assigned_cluster])
                                    # Only add to array if reasonably close (e.g., < 70% of threshold)
                                    if dist_to_center <= MAX_COLOR_DISTANCE_THRESHOLD * 0.7:
                                        if len(team_colors_array[assigned_cluster]) < 20: # Limit samples per team
                                            team_colors_array[assigned_cluster].append(smoothed_color)

                                # Now re-assign *all* currently moving, non-referee, in-field players
                                for tid in current_moving_tracker_ids:
                                    if tid in is_player_referee or tid in is_static: continue
                                     # Check if in field again (might have moved out briefly)
                                    in_field_now = False
                                    for i_check, tid_check in enumerate(detections_with_tracking.tracker_id or []):
                                        if tid_check is not None and int(tid_check) == tid:
                                            box_check = detections_with_tracking.xyxy[i_check]
                                            in_field_now = is_in_field_region(box_check, frame_height, frame_width, FIELD_REGION)
                                            break
                                    if not in_field_now:
                                        team_assignments[tid] = "Filtered"
                                        continue

                                    smoothed_color = get_smoothed_color(player_colors_history[tid])
                                    if smoothed_color:
                                        is_valid, closest_cluster = is_valid_team_color(
                                            smoothed_color, current_team_centers,
                                            MAX_COLOR_DISTANCE_THRESHOLD, team_colors_array
                                        )
                                        if is_valid and closest_cluster >= 0:
                                            team_assignments[tid] = cluster_to_display_label.get(closest_cluster, "Filtered")
                                            assigned_count += 1
                                        else:
                                            team_assignments[tid] = "Filtered" # Mark as filtered if color invalid
                                    else:
                                        team_assignments[tid] = "Filtered" # Mark if no color

                                print(f"Re-assigned {assigned_count} players based on K-Means distance.")

                        except Exception as e_kmeans:
                            print(f"K-Means Clustering Exception: {e_kmeans}")
                            traceback.print_exc() # Print detailed error for debugging
                    else:
                        print("Not enough unique colors for K-Means clustering.")

                    # Train/Update Logistic Regression Classifier
                    # Require a minimum number of diverse samples
                    if len(team_training_data) >= 10: # Increased min samples
                         labels_in_data = [d[1] for d in team_training_data]
                         # Check if we have samples from both teams (0 and 1)
                         if 0 in labels_in_data and 1 in labels_in_data:
                             try:
                                 print(f"Attempting to train classifier with {len(team_training_data)} samples...")
                                 X_train = np.array([data[0] for data in team_training_data])
                                 y_train = np.array(labels_in_data)
                                 team_classifier.fit(X_train, y_train)
                                 classifier_trained = True
                                 print(f"Team classifier trained successfully.")
                             except Exception as e_train:
                                 print(f"Error training team classifier: {e_train}")
                                 classifier_trained = False # Mark as not trained if error occurs
                         else: print("Not enough diversity in training data (need samples from both teams).")
                    else: print(f"Not enough training samples ({len(team_training_data)} < 10).")

                else:
                    print("Not enough valid player colors found for K-Means clustering.")

            # 5. Visualization
            annotated_frame = assign_and_visualize_teams(
                frame, detections_with_tracking, track_history, player_colors_history,
                team_assignments, is_static, current_team_centers,
                cluster_to_display_color, cluster_to_display_label, is_player_referee,
                MAX_COLOR_DISTANCE_THRESHOLD, team_colors_array, FENCE_POSITIONS, horizontal_lines,
                team_classifier, classifier_trained # Pass classifier state
            )

            # 6. Write Frame to Output Video
            if video_writer:
                 video_writer.write(annotated_frame)
            processed_frame_count += 1

            # Progress Update
            loop_time = time.time() - loop_start_time
            if frame_num % 30 == 0 or frame_num < 10:
                 elapsed_total = time.time() - start_total_time
                 est_rem_str = "N/A"
                 if total_frames > 0 and frame_num > 0:
                     est_rem_secs = (elapsed_total / (frame_num + 1)) * (total_frames - frame_num - 1)
                     est_rem_str = f"~{est_rem_secs:.0f}s"
                 # Log moving IDs count for diagnostics
                 moving_ids_count = len(current_moving_tracker_ids)
                 print(f"{log_prefix}Frame {frame_num}/{total_frames or '?'} ({loop_time:.3f}s) | Moving IDs: {moving_ids_count} | Est. Rem: {est_rem_str}   ", end='\r')


        # <<< END OF WHILE LOOP >>>
        print(f"\n{log_prefix}Video processing loop finished.")

    except Exception as e:
        print(f"\n{log_prefix}🚨 An error occurred during video processing: {e}")
        traceback.print_exc()
        status_message = f"Error during processing: {e}"
        if output_video_path and os.path.exists(output_video_path):
             # Clean up potentially incomplete output file on error
             try:
                 if video_writer and video_writer.isOpened(): video_writer.release()
                 os.remove(output_video_path)
                 print(f"{log_prefix}Removed incomplete output file: {output_video_path}")
             except Exception as e_del: print(f"Error removing file: {e_del}")
        output_video_path = None # Ensure None is returned on error
        return output_video_path, status_message

    finally:
        # --- Release Resources ---
        if cap:
            cap.release()
            print(f"{log_prefix}Video capture released.")
        if video_writer:
            video_writer.release()
            print(f"{log_prefix}Video writer released.")

        total_end_time = time.time()
        total_duration = total_end_time - start_total_time
        print(f"\n{log_prefix}Total processing time: {total_duration:.2f} seconds.")
        print(f"{log_prefix}Processed {processed_frame_count} frames.")

        # Final check on output file
        if output_video_path and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 1024:
            print(f"{log_prefix}Output video successfully saved: {output_video_path}")
            status_message = "Processing successful."
            return output_video_path, status_message
        elif output_video_path and os.path.exists(output_video_path):
             print(f"{log_prefix}Error: Output file exists but is too small/empty.")
             status_message = "Error: Output file likely corrupt."
             # Optionally remove tiny file here too
             # os.remove(output_video_path)
             return None, status_message
        else:
             print(f"{log_prefix}Error: Output file not found or processing failed before creation.")
             status_message = "Error: Output file not created."
             return None, status_message