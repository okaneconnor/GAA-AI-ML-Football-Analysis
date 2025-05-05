# football_utils/video_processing.py

import cv2
import os
import time
import numpy as np
import traceback
from collections import defaultdict, deque

# Import necessary functions
from football_utils.detection import detect_objects
from football_utils.tracking import calculate_optical_flow, apply_perspective_transform
from football_utils.team_assignment import is_referee, extract_jersey_roi, is_in_field_region
from football_utils.llm_inference import classify_output

# --- Configuration ---
# Set desired confidence threshold for 'person' detection
PERSON_CONF_THRESHOLD = 0.4 # Adjust as needed (0.0 to 1.0)

# Movement analysis settings
MOVEMENT_WINDOW_SIZE = 15
MOVEMENT_STATIC_THRESHOLD = 15

# *** IMPROVED FILTERING SETTINGS ***
MIN_PLAYER_HEIGHT_PIXELS = 40         # Minimum height for players
MAX_PLAYER_BOTTOM_Y_PERCENT = 0.86    # Ignore bottom 14% of frame 
MAX_COLOR_DISTANCE_THRESHOLD = 55     # Color distance threshold (balanced)
MIN_COLOR_SATURATION = 0.15           # Minimum color saturation
COLOR_CONSISTENCY_THRESHOLD = 40      # Color consistency threshold

# Field region definition (area where players should be detected)
FIELD_REGION = {
    'top': 0.45,        # Vertical position below fence
    'bottom': 0.90,     # Bottom of field
    'left': 0.05,       # Left edge with small margin
    'right': 0.95       # Right edge with small margin
}

# Fence line detection for spectator filtering
USE_HORIZONTAL_LINE_FILTERING = True  # Enable fence line detection
FENCE_DETECTION_METHOD = "position_based"  # "position_based" or "line_detection"
DISPLAY_FENCE_LINES = False           # Set to False to hide fence lines in output

# Set multiple fence line positions (can be fine-tuned for each video)
FENCE_POSITIONS = [
    {'y_percent': 0.39, 'height_percent': 0.05, 'buffer_above': 0.00, 'buffer_below': 0.05},  # Main fence line
    {'y_percent': 0.45, 'height_percent': 0.05, 'buffer_above': 0.00, 'buffer_below': 0.05},  # Secondary fence line
]

# Colors for visualization
DISPLAY_RED = (0, 0, 255)  # BGR format (Red)
DISPLAY_BLUE = (255, 0, 0)  # BGR format (Blue)
DISPLAY_GREY = (128, 128, 128)  # Grey
DISPLAY_WHITE = (255, 255, 255)  # White

def color_distance(c1, c2):
    """Calculate Euclidean distance between two colors in BGR space."""
    if c1 is None or c2 is None: 
        return float('inf')
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    return np.linalg.norm(c1 - c2)

def detect_horizontal_lines(frame, threshold=100, min_line_length=100):
    """Detect horizontal lines (like fences) in the frame."""
    if not USE_HORIZONTAL_LINE_FILTERING:
        return []
        
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=min_line_length, maxLineGap=10)
    
    # Filter for mostly horizontal lines
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is mostly horizontal (small vertical change)
            if abs(y2 - y1) < 20:  # Adjust this threshold as needed
                horizontal_lines.append((x1, y1, x2, y2))
    
    return horizontal_lines

def is_near_fence_line(box, frame_height, frame_width, fence_positions=None, horizontal_lines=None):
    """Check if detection is near a fence line using either predefined positions or detected lines."""
    if not USE_HORIZONTAL_LINE_FILTERING:
        return False
        
    x1, y1, x2, y2 = box
    
    # Method 1: Position-based fence detection (more reliable)
    if FENCE_DETECTION_METHOD == "position_based" and fence_positions:
        for fence in fence_positions:
            fence_y = fence['y_percent'] * frame_height
            fence_height = fence['height_percent'] * frame_height
            buffer_above = fence['buffer_above'] * frame_height
            buffer_below = fence['buffer_below'] * frame_height
            
            # Check if bottom of detection is in fence region
            if (y2 >= fence_y - buffer_above and 
                y2 <= fence_y + fence_height + buffer_below):
                return True
    
    # Method 2: Detected line-based approach
    elif FENCE_DETECTION_METHOD == "line_detection" and horizontal_lines:
        box_bottom_y = y2
        for x1_line, y1_line, x2_line, y2_line in horizontal_lines:
            # Find average y position of the line
            line_y = (y1_line + y2_line) / 2
            # Check if bottom of detection is near the line
            if abs(box_bottom_y - line_y) < 20:  # Adjust threshold as needed
                return True
    
    return False

def get_representative_color(roi, k=1):
    """Get dominant color from jersey region."""
    if roi is None or roi.size < 15*3: 
        return None
    
    # Convert to HSV for better color analysis
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv_roi.reshape(-1, 3)
    
    # Filter out very dark/bright pixels
    valid_mask = (
        (hsv_pixels[:, 1] > 20) &        # Min saturation
        (hsv_pixels[:, 2] > 30) &        # Min brightness
        (hsv_pixels[:, 2] < 235)         # Max brightness
    )
    
    if np.sum(valid_mask) < 10:
        # Fall back to simple filtering if HSV filtering removes too many pixels
        pixels = roi.reshape(-1, 3).astype(np.float32)
        dark_light_mask = (np.all(pixels > [20, 20, 20], axis=1)) & (np.all(pixels < [240, 240, 240], axis=1))
        pixels = pixels[dark_light_mask]
        if len(pixels) < 5: 
            return None
        return tuple(np.mean(pixels, axis=0).astype(int))
    
    # Use HSV-filtered pixels for clustering
    filtered_hsv = hsv_pixels[valid_mask]
    filtered_bgr = cv2.cvtColor(filtered_hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    
    # Apply K-means on the filtered BGR colors
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(filtered_bgr)
        dominant_color = tuple(kmeans.cluster_centers_[0].astype(int))
        return dominant_color
    except Exception as e:
        # Fall back to average if clustering fails
        return tuple(np.mean(filtered_bgr, axis=0).astype(int)) if len(filtered_bgr) > 0 else None

def get_smoothed_color(color_deque):
    """Get median color from history to reduce noise."""
    if not color_deque or len(color_deque) < 2: 
        return None
    
    colors_list = list(color_deque)
    colors_np = np.array(colors_list)
    
    try:
        # Use median for robustness against outliers
        median_b = int(np.median(colors_np[:, 0]))
        median_g = int(np.median(colors_np[:, 1]))
        median_r = int(np.median(colors_np[:, 2]))
        return (median_b, median_g, median_r)
    except IndexError:
        return None

def has_player_movement_pattern(track_history, min_window=5):
    """Check if movement pattern is consistent with a player."""
    if not track_history or len(track_history) < min_window:
        return True  # Not enough data
    
    # Calculate velocities between consecutive points
    velocities = []
    track_list = list(track_history)
    for i in range(1, len(track_list)):
        p1 = np.array(track_list[i-1])
        p2 = np.array(track_list[i])
        velocity = np.sqrt(np.sum((p2-p1)**2))
        velocities.append(velocity)
    
    # Player movement patterns: not too static, not erratic
    avg_velocity = np.mean(velocities)
    velocity_std = np.std(velocities)
    
    # More specific player movement pattern
    return 0.3 < avg_velocity < 35 and velocity_std < 18

def is_valid_team_color(color, team_centers, color_threshold, team_colors_array):
    """Determine if a color belongs to one of the teams based on distance and consistency."""
    if color is None or len(team_centers) != 2:
        return False, -1
    
    # Distance to both team centers
    dist0 = color_distance(color, team_centers[0])
    dist1 = color_distance(color, team_centers[1])
    min_dist = min(dist0, dist1)
    closest_team = 0 if dist0 <= dist1 else 1
    
    # Check color saturation (convert to HSV)
    color_bgr = np.uint8([[color]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    saturation = color_hsv[1] / 255.0
    
    # Basic validation based on distance and saturation
    basic_valid = min_dist <= color_threshold and saturation >= MIN_COLOR_SATURATION
    
    # If we have enough team color samples, perform distribution check
    if team_colors_array[closest_team] and len(team_colors_array[closest_team]) >= 3:
        team_colors_np = np.array(team_colors_array[closest_team])
        color_np = np.array(color)
        team_mean = np.mean(team_colors_np, axis=0)
        
        # Calculate color deviation using Euclidean distance
        color_deviation = np.linalg.norm(color_np - team_mean)
        is_within_distribution = color_deviation <= COLOR_CONSISTENCY_THRESHOLD
        
        return basic_valid and is_within_distribution, closest_team
    else:
        # If not enough team color samples, just use distance threshold
        return basic_valid, closest_team

def assign_and_visualize_teams(
    frame, detections_with_tracking, track_history, player_colors_history,
    team_assignments, is_static, current_team_centers,
    cluster_to_display_color, cluster_to_display_label, is_player_referee,
    max_color_distance_threshold, team_colors_array, fence_positions, horizontal_lines
    ):
    """
    This function handles both team assignment and visualization with ellipses instead of boxes.
    This is directly based on the example code.
    """
    output_frame = frame.copy()
    DISPLAY_RED=(0,0,255)  # BGR 
    DISPLAY_BLUE=(255,0,0)
    DISPLAY_GREY=(128,128,128)
    DISPLAY_WHITE=(255,255,255)
    
    # Get frame dimensions for calculations
    frame_h, frame_w = frame.shape[:2]
    
    # Optional: Draw fence lines for debugging
    if DISPLAY_FENCE_LINES:
        if USE_HORIZONTAL_LINE_FILTERING and FENCE_DETECTION_METHOD == "position_based":
            for fence in fence_positions:
                fence_y = int(fence['y_percent'] * frame_h)
                cv2.line(output_frame, (0, fence_y), (frame_w, fence_y), (0, 255, 255), 1)
        
        if USE_HORIZONTAL_LINE_FILTERING and FENCE_DETECTION_METHOD == "line_detection":
            for x1, y1, x2, y2 in horizontal_lines:
                cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Process each detection - use EXACTLY the ellipse drawing code from the example
    if hasattr(detections_with_tracking, 'tracker_id') and detections_with_tracking.tracker_id is not None:
        for i, box in enumerate(detections_with_tracking.xyxy):
            track_id = detections_with_tracking.tracker_id[i]
            if track_id is None: continue
            track_id = int(track_id)

            display_label = f"#{track_id}"; display_color = DISPLAY_WHITE # Defaults
            
            # Extract key information about this detection
            x1, y1, x2, y2 = map(int, box)
            height = y2 - y1
            box_center_y = (y1 + y2) / 2
            
            # 1. Check for spectators near fence line
            near_fence = is_near_fence_line(box, frame_h, frame_w, fence_positions, horizontal_lines)
            
            # 2. Check if in main playing field
            in_field = is_in_field_region(box, frame_h, frame_w, FIELD_REGION)
            
            # Filter out spectators and non-field detections
            if near_fence:
                # Skip visualization for spectators - they're completely filtered out
                continue
            elif not in_field and box_center_y < FIELD_REGION['top'] * frame_h:
                # Skip visualization for detections above the field (likely spectators)
                continue
            elif track_id in is_static:
                display_label = "Static"; display_color = DISPLAY_GREY
            elif track_id in is_player_referee:
                display_label=f"#{track_id} Referee"; display_color=DISPLAY_GREY
            else: # Potential team player
                # Check movement pattern
                player_movement_pattern = has_player_movement_pattern(track_history[track_id], min_window=3)
                
                if not player_movement_pattern:
                    display_label = "Filtered"; display_color = DISPLAY_GREY
                else:
                    # Get current assignment or default to "Filtered"
                    assigned_label = team_assignments.get(track_id, "Filtered")

                    # If classifier not used or failed, use distance-based approach
                    if assigned_label == "Filtered" and len(current_team_centers) == 2:
                        smoothed_color = get_smoothed_color(player_colors_history[track_id])
                        
                        if smoothed_color:
                            # Check if color matches a team
                            is_valid, assigned_cluster = is_valid_team_color(
                                smoothed_color, 
                                current_team_centers, 
                                max_color_distance_threshold,
                                team_colors_array
                            )
                            
                            if is_valid and assigned_cluster >= 0:
                                potential_new_label = cluster_to_display_label.get(assigned_cluster)
                                if potential_new_label: 
                                    assigned_label = potential_new_label
                                    team_assignments[track_id] = assigned_label
                            else:
                                # Only do central player check for detections that are clearly in the field
                                if in_field and height >= MIN_PLAYER_HEIGHT_PIXELS * 1.2:
                                    # If in central field area, assign to closest team
                                    dist0 = color_distance(smoothed_color, current_team_centers[0]) if smoothed_color else float('inf')
                                    dist1 = color_distance(smoothed_color, current_team_centers[1]) if smoothed_color else float('inf')
                                    closest_team = 0 if dist0 <= dist1 else 1
                                    assigned_label = cluster_to_display_label.get(closest_team, "Filtered")
                                    team_assignments[track_id] = assigned_label
                                else:
                                    assigned_label = "Filtered"
                                    team_assignments[track_id] = assigned_label

                    # Set display color based on team assignment
                    if assigned_label == "Red Team": display_color = DISPLAY_RED
                    elif assigned_label == "Blue Team": display_color = DISPLAY_BLUE
                    elif assigned_label == "Team 0": display_color = DISPLAY_BLUE
                    elif assigned_label == "Referee": display_color = DISPLAY_GREY
                    else: display_color = DISPLAY_GREY

                    display_label = f"#{track_id} {assigned_label}"

            # Draw ellipse and label - EXACTLY as in the example code
            center_x=int((box[0]+box[2])/2); bottom_y=int(box[3]); ellipse_axes=(20,6)
            cv2.ellipse(output_frame, (center_x, bottom_y), ellipse_axes, 0, 0, 360, display_color, 2)
            cv2.putText(output_frame, display_label, (center_x - 30, bottom_y + ellipse_axes[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_color, 1)

    return output_frame

def process_video(input_video_path, output_folder):
    log_prefix = "[PROCESS_VIDEO_FINAL] "
    print(f"{log_prefix}Starting for: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"{log_prefix}Error: Could not open input video.")
        return None, "Input Video Error"

    # --- Get video properties ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = int(frame_count_prop) if frame_count_prop > 0 else 0
    print(f"{log_prefix}Input Props: {width}x{height} @ {fps:.2f} FPS, Frames: ~{frame_count if frame_count>0 else 'Unknown'}")

    if width <= 0 or height <= 0 or fps <= 0:
         print(f"{log_prefix}Error: Invalid video properties.")
         cap.release(); return None, "Invalid Properties"

    # --- Setup Video Writer (Reliable MP4V) ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = f"output_final_{int(time.time())}.mp4"
    output_video_path = os.path.join(output_folder, output_filename)
    print(f"{log_prefix}Attempting to write MP4 (mp4v) to: {output_video_path}")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"{log_prefix}Error: Failed to open VideoWriter (mp4v/MP4)."); cap.release(); return None, "VideoWriter Open Error"
    else: print(f"{log_prefix}VideoWriter opened successfully (mp4v).")

    # --- Initialize data structures for tracking and team assignment ---
    track_history = defaultdict(lambda: deque(maxlen=MOVEMENT_WINDOW_SIZE))
    player_colors_history = defaultdict(lambda: deque(maxlen=10))
    team_assignments = {}
    is_static = set()
    is_player_referee = {}
    current_team_centers = []
    team_colors_array = [[], []]
    cluster_to_display_color = {0: DISPLAY_BLUE, 1: DISPLAY_RED}
    cluster_to_display_label = {0: "Blue Team", 1: "Red Team"}

    # --- Initialize processing variables ---
    frame_num = 0
    prev_frame = None
    processing_successful = True
    final_output_path = None
    write_errors = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: print(f"{log_prefix}End of stream after {frame_num} frames read."); break
            frame_num += 1
            if frame_num == 1: prev_frame = frame.copy()

            # Frame validation
            if frame is None or frame.shape[0] != height or frame.shape[1] != width or frame.dtype != np.uint8:
                print(f"{log_prefix}Invalid frame {frame_num}, skipping write."); continue

            if frame_num % 100 == 0: # Progress
                 total_frames_str = f"{frame_count}" if frame_count > 0 else "?"
                 print(f"{log_prefix}Processing frame {frame_num}/{total_frames_str}...")

            try:
                # Detect horizontal lines for spectator filtering
                horizontal_lines = detect_horizontal_lines(frame) if FENCE_DETECTION_METHOD == "line_detection" else []
                
                # 1. Object Detection
                all_detections = detect_objects(frame)

                # Import supervision as sv (do this here to avoid circular imports)
                try:
                    import supervision as sv
                    supervision_available = True
                except:
                    supervision_available = False
                    
                # Convert detections to supervision format for tracking
                if supervision_available:
                    try:
                        # Filter detections to only include people
                        detections_to_track = []
                        for i, det in enumerate(all_detections):
                            if det['name'] == 'person' and det['confidence'] >= PERSON_CONF_THRESHOLD:
                                # Extract box coordinates
                                xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                                
                                # Skip very small detections
                                if (ymax - ymin) < MIN_PLAYER_HEIGHT_PIXELS:
                                    continue
                                    
                                # Skip detections near fence line
                                if is_near_fence_line([xmin, ymin, xmax, ymax], height, width, FENCE_POSITIONS, horizontal_lines):
                                    continue
                                    
                                # Add tracking ID to detection
                                detections_to_track.append({
                                    'xyxy': [xmin, ymin, xmax, ymax],
                                    'confidence': det['confidence'],
                                    'class_id': 0  # Person class
                                })
                                
                        # Create supervision detections object
                        if detections_to_track:
                            detections_sv = sv.Detections(
                                xyxy=np.array([d['xyxy'] for d in detections_to_track]),
                                confidence=np.array([d['confidence'] for d in detections_to_track]),
                                class_id=np.array([d['class_id'] for d in detections_to_track])
                            )
                            
                            # Initialize tracker if not already done
                            if 'byte_tracker' not in locals():
                                byte_tracker = sv.ByteTrack(frame_rate=fps)
                                
                            # Update tracker with detections
                            detections_with_tracking = byte_tracker.update_with_detections(detections=detections_sv)
                        else:
                            detections_with_tracking = sv.Detections.empty()
                    except Exception as e:
                        print(f"{log_prefix}Error in tracking: {e}")
                        detections_with_tracking = sv.Detections.empty()
                else:
                    # If supervision not available, create a dummy detections object
                    class DummyDetections:
                        def __init__(self, boxes, tracker_ids):
                            self.xyxy = boxes
                            self.tracker_id = tracker_ids
                            
                    detections_with_tracking = DummyDetections([], [])

                # ------- 3. Update track history and movement analysis -------
                current_moving_tracker_ids = set()
                current_frame_track_ids = set()
                
                if hasattr(detections_with_tracking, 'tracker_id') and detections_with_tracking.tracker_id is not None:
                    centers = detections_with_tracking.get_anchors_coordinates(anchor=sv.Position.CENTER)
                    valid_track_indices = [i for i, tid in enumerate(detections_with_tracking.tracker_id) if tid is not None]
                    
                    for i in valid_track_indices:
                        track_id = int(detections_with_tracking.tracker_id[i])
                        current_frame_track_ids.add(track_id)
                        box = detections_with_tracking.xyxy[i]
                        center_point = tuple(map(int, centers[i]))
                        
                        # Skip if near fence line (secondary check)
                        if is_near_fence_line(box, height, width, FENCE_POSITIONS, horizontal_lines):
                            continue
                        
                        # Update track history for movement analysis
                        track_history[track_id].append(center_point)
                        is_track_static = False
                        
                        if len(track_history[track_id]) >= MOVEMENT_WINDOW_SIZE:
                            displacement = color_distance(track_history[track_id][0], track_history[track_id][-1])
                            if displacement < MOVEMENT_STATIC_THRESHOLD:
                                is_static.add(track_id)
                                is_track_static = True
                            else:
                                is_static.discard(track_id)
                                current_moving_tracker_ids.add(track_id)
                        elif track_id not in is_static:
                            current_moving_tracker_ids.add(track_id)
                        
                        # Only update color if moving and in field
                        if not is_track_static and is_in_field_region(box, height, width, FIELD_REGION):
                            roi = extract_jersey_roi(frame, box)
                            if roi is not None:
                                # Check referee status
                                if is_referee(roi):
                                    is_player_referee[track_id] = True
                                    team_assignments[track_id] = "Referee"
                                elif track_id not in is_player_referee:
                                    # Store color for clustering
                                    rep_color = get_representative_color(roi)
                                    if rep_color is not None:
                                        player_colors_history[track_id].append(rep_color)

                # Remove disappeared tracks
                all_known_ids = set(track_history.keys())
                disappeared_ids = all_known_ids - current_frame_track_ids
                for old_id in disappeared_ids:
                    track_history.pop(old_id, None)
                    player_colors_history.pop(old_id, None)
                    team_assignments.pop(old_id, None)
                    is_static.discard(old_id)
                    is_player_referee.pop(old_id, None)

                # ------- 4. Periodic Team Re-clustering -------
                if (frame_num % 30 == 0 or frame_num < 30) and len(current_moving_tracker_ids) > 0:
                    print(f"\n--- Frame {frame_num}: Re-clustering ---")
                    colors_for_clustering = []; tracks_for_clustering = []
                    
                    # Reset team color arrays
                    team_colors_array = [[], []]
                    
                    for tid in current_moving_tracker_ids:
                        # Use only valid player tracks for clustering that are not near fence
                        if (tid not in is_player_referee and 
                            tid not in is_static and 
                            has_player_movement_pattern(track_history[tid], min_window=3)):
                            
                            # Get track position to check if in field
                            for i, tid_check in enumerate(detections_with_tracking.tracker_id):
                                if tid_check is not None and int(tid_check) == tid:
                                    box = detections_with_tracking.xyxy[i]
                                    if is_in_field_region(box, height, width, FIELD_REGION):
                                        smoothed_color = get_smoothed_color(player_colors_history[tid])
                                        if smoothed_color is not None:
                                            colors_for_clustering.append(smoothed_color)
                                            tracks_for_clustering.append(tid)
                                    break
                    
                    print(f"Clustering {len(colors_for_clustering)} colors after filtering...")
                    
                    if len(colors_for_clustering) >= 4:  # Need minimum number for meaningful clusters
                        cluster_colors_np = np.array(colors_for_clustering).astype(np.float32)
                        unique_colors = np.unique(cluster_colors_np, axis=0)
                        
                        if len(unique_colors) >= 2:
                            try:
                                # Use K-means with better initialization
                                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                                kmeans.fit(cluster_colors_np)
                                current_team_centers = [tuple(map(int,center)) for center in kmeans.cluster_centers_]
                                print(f"K-Means OK. Centers: {current_team_centers}")
                                
                                # Determine which cluster is "Red" vs "Blue"
                                color0 = current_team_centers[0]; color1 = current_team_centers[1]
                                ratio0 = color0[2]/(color0[0]+1.0); ratio1 = color1[2]/(color1[0]+1.0)
                                
                                if ratio0 > ratio1:
                                    cluster_to_display_color = {0:DISPLAY_RED,1:DISPLAY_BLUE}
                                    cluster_to_display_label = {0:"Red Team",1:"Blue Team"}
                                else:
                                    cluster_to_display_color = {0:DISPLAY_BLUE,1:DISPLAY_RED}
                                    cluster_to_display_label = {0:"Blue Team",1:"Red Team"}
                                
                                print(f"Team mapping: {cluster_to_display_label}")
                                
                                # Assign clusters to tracks and populate team color arrays
                                labels = kmeans.labels_
                                assigned_count = 0
                                
                                # First pass: populate team color arrays with high-confidence assignments
                                for i, tid in enumerate(tracks_for_clustering):
                                    assigned_cluster = labels[i]
                                    smoothed_color = get_smoothed_color(player_colors_history[tid])
                                    
                                    if smoothed_color:
                                        # Check distance to assigned cluster center
                                        dist_to_center = color_distance(smoothed_color, current_team_centers[assigned_cluster])
                                        
                                        # Only use high-confidence colors for team arrays
                                        if dist_to_center <= MAX_COLOR_DISTANCE_THRESHOLD * 0.7:
                                            if len(team_colors_array[assigned_cluster]) < 20:
                                                team_colors_array[assigned_cluster].append(smoothed_color)
                                
                                # Second pass: assign all players based on color distance and field position
                                for tid in current_moving_tracker_ids:
                                    # Skip referees and static objects
                                    if tid in is_player_referee or tid in is_static:
                                        continue
                                    
                                    # Get track position
                                    in_field = False
                                    for i, tid_check in enumerate(detections_with_tracking.tracker_id):
                                        if tid_check is not None and int(tid_check) == tid:
                                            box = detections_with_tracking.xyxy[i]
                                            in_field = is_in_field_region(box, height, width, FIELD_REGION)
                                            break
                                    
                                    # Skip if not in field
                                    if not in_field:
                                        team_assignments[tid] = "Filtered"
                                        continue
                                            
                                    smoothed_color = get_smoothed_color(player_colors_history[tid])
                                    if smoothed_color:
                                        # Check if color is valid team color
                                        is_valid, closest_cluster = is_valid_team_color(
                                            smoothed_color,
                                            current_team_centers,
                                            MAX_COLOR_DISTANCE_THRESHOLD,
                                            team_colors_array
                                        )
                                        
                                        if is_valid and closest_cluster >= 0:
                                            team_assignments[tid] = cluster_to_display_label.get(closest_cluster, "Filtered")
                                            assigned_count += 1
                                        else:
                                            team_assignments[tid] = "Filtered"
                                    else:
                                        team_assignments[tid] = "Filtered"
                                
                                print(f"Re-assigned {assigned_count} players to teams")
                                
                            except Exception as e:
                                print(f"K-Means Exception: {e}")
                        else:
                            print("Not enough unique colors for clustering.")
                    else:
                        print("Not enough valid jersey colors for clustering.")
                        
                # ------- 5. Visualization -------
                # Use the function from the example code to visualize the teams with ellipses
                # This function draws ELLIPSES at player feet, not boxes
                annotated_frame = assign_and_visualize_teams(
                    frame, detections_with_tracking, track_history, player_colors_history,
                    team_assignments, is_static, current_team_centers,
                    cluster_to_display_color, cluster_to_display_label, is_player_referee,
                    MAX_COLOR_DISTANCE_THRESHOLD, team_colors_array, FENCE_POSITIONS, horizontal_lines
                )
                
                # ------- 6. Write Frame -------
                try:
                    out.write(annotated_frame)
                except Exception as write_err:
                    write_errors += 1
                    print(f"{log_prefix}Error writing frame {frame_num}: {write_err}")

                prev_frame = frame.copy()

            except Exception as frame_proc_err:
                print(f"{log_prefix}Error processing frame {frame_num}: {frame_proc_err}")
                traceback.print_exc()
                processing_successful = False

        # End of while loop
        print(f"{log_prefix}Finished loop. Frames read: {frame_num}, Write errors: {write_errors}")

    except Exception as loop_err:
        print(f"{log_prefix}Error in loop: {loop_err}")
        traceback.print_exc()
        processing_successful = False
    finally:
        # --- Release Resources ---
        print(f"{log_prefix}Releasing resources...")
        if cap is not None: cap.release()
        if out is not None: out.release()
        print(f"{log_prefix}Resources released.")

    # --- Check output file ---
    final_output_path = None
    if processing_successful and write_errors == 0 and os.path.exists(output_video_path):
        try:
            file_size = os.path.getsize(output_video_path)
            if file_size > 1024:
                print(f"{log_prefix}Output MP4 file successfully created: {output_video_path} ({file_size} bytes)")
                final_output_path = output_video_path
            else: print(f"{log_prefix}Error: Output file is too small/empty ({file_size} bytes).")
        except OSError as e: print(f"{log_prefix}Error checking file size: {e}")
    elif not os.path.exists(output_video_path): print(f"{log_prefix}Error: Output file not found.")
    else: print(f"{log_prefix}Output file likely corrupt due to errors (success={processing_successful}, write_errs={write_errors}).")

    if final_output_path is None and os.path.exists(output_video_path):
         print(f"{log_prefix}Deleting potentially corrupt/empty file.")
         try: os.remove(output_video_path)
         except OSError as e: print(f"Error deleting file: {e}")

    # --- LLM Classification ---
    classification_str = "N/A"
    try:
        print(f"{log_prefix}Running LLM Classification...")
        summary_text = f"Analysis complete. Processed {frame_num} frames. Detected players and assigned teams."
        classification = classify_output(summary_text)
        if classification and isinstance(classification, list) and len(classification) > 0 and isinstance(classification[0], dict):
            label = classification[0].get('label', 'N/A'); score = classification[0].get('score', 0.0)
            classification_str = f"Label: {label}, Score: {score:.4f}"
        else: classification_str = str(classification)
        print(f"{log_prefix}LLM Classification Result: {classification_str}")
    except Exception as llm_err:
         print(f"{log_prefix}Error during LLM classification: {llm_err}")
         classification_str = "Error during LLM classification"

    print(f"{log_prefix}Returning path: {final_output_path}, classification: {classification_str}")
    return final_output_path, classification_str