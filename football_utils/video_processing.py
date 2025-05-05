# football_utils/video_processing.py

import cv2
import os
import time
import numpy as np
import traceback
from collections import defaultdict, deque
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import supervision as sv # Make sure this is installed!

# Import necessary functions from other modules
# Removed load_yolo_model as it's now loaded directly in detection.py
from football_utils.detection import detect_objects
from football_utils.team_assignment import (
    extract_jersey_roi,
    is_referee,
    get_representative_color,
    get_smoothed_color,
    extract_color_histogram
)

# --- Configuration ---
VIDEO_PROCESSING_FRAME_INTERVAL = 1
# This constant is now mainly for reference/printing, as loading uses hardcoded path in detection.py
YOLO_MODEL_NAME = 'yolo11n.pt'
CONFIDENCE_THRESHOLD = 0.30
MOVEMENT_WINDOW_SIZE = 15
MOVEMENT_STATIC_THRESHOLD = 15
TEAM_RECLUSTER_INTERVAL = 60

# Filtering Settings
MIN_PLAYER_HEIGHT_PIXELS = 40
MAX_PLAYER_BOTTOM_Y_PERCENT = 0.86
MAX_COLOR_DISTANCE_THRESHOLD = 55
MIN_COLOR_SATURATION = 0.15
COLOR_CONSISTENCY_THRESHOLD = 40

# Fence Line Detection Parameters
USE_HORIZONTAL_LINE_FILTERING = True
FENCE_DETECTION_METHOD = "position_based"
DISPLAY_FENCE_LINES = False

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
# Functions color_distance, detect_horizontal_lines, is_near_fence_line,
# is_in_field_region, has_player_movement_pattern, is_valid_team_color
# remain the same as the previous version. Copied here for completeness.

def color_distance(c1, c2):
    """Calculate Euclidean distance between two BGR colors."""
    if c1 is None or c2 is None: return float('inf')
    c1 = np.asarray(c1, dtype=np.float32); c2 = np.asarray(c2, dtype=np.float32)
    return np.linalg.norm(c1 - c2)

def detect_horizontal_lines(frame, threshold=100, min_line_length=100):
    """Detect horizontal lines (like fences) using Hough Transform."""
    if not USE_HORIZONTAL_LINE_FILTERING or FENCE_DETECTION_METHOD != "line_detection": return []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=min_line_length, maxLineGap=10)
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 20: horizontal_lines.append((x1, y1, x2, y2))
        return horizontal_lines
    except Exception: return []

def is_near_fence_line(box, frame_height, frame_width, fence_positions=None, horizontal_lines=None):
    """Check if detection box bottom is near a configured or detected fence line."""
    if not USE_HORIZONTAL_LINE_FILTERING: return False
    x1, y1, x2, y2 = map(int, box)
    if FENCE_DETECTION_METHOD == "position_based" and fence_positions:
        for fence in fence_positions:
            fence_y_abs = fence['y_percent'] * frame_height; fence_height_abs = fence['height_percent'] * frame_height
            buffer_above_abs = fence['buffer_above'] * frame_height; buffer_below_abs = fence['buffer_below'] * frame_height
            if (y2 >= fence_y_abs - buffer_above_abs and y2 <= fence_y_abs + fence_height_abs + buffer_below_abs): return True
    elif FENCE_DETECTION_METHOD == "line_detection" and horizontal_lines:
        box_bottom_y = y2
        for x1_line, y1_line, x2_line, y2_line in horizontal_lines:
            line_y_avg = (y1_line + y2_line) / 2
            if abs(box_bottom_y - line_y_avg) < 20: return True
    return False

def is_in_field_region(box, frame_height, frame_width, field_region):
    """Check if the center of the detection box is within the defined field region."""
    x1, y1, x2, y2 = box; box_center_x = (x1 + x2) / 2; box_center_y = (y1 + y2) / 2
    in_field = (box_center_y > field_region['top'] * frame_height and box_center_y < field_region['bottom'] * frame_height and
                box_center_x > field_region['left'] * frame_width and box_center_x < field_region['right'] * frame_width)
    return in_field

def has_player_movement_pattern(track_history_deque, min_window=5):
    """Check track history (a deque) for player-like movement."""
    if len(track_history_deque) < min_window: return True
    velocities = []; track_list = list(track_history_deque)
    for i in range(1, len(track_list)):
        p1 = np.array(track_list[i-1]); p2 = np.array(track_list[i]); velocity = np.linalg.norm(p2 - p1); velocities.append(velocity)
    if not velocities: return True
    avg_velocity = np.mean(velocities); velocity_std = np.std(velocities)
    is_player_pattern = (0.3 < avg_velocity < 35) and (velocity_std < 18)
    return is_player_pattern

def is_valid_team_color(color, team_centers, color_threshold, team_colors_array):
    """Check if a color likely belongs to a team based on cluster centers and distribution."""
    if color is None or len(team_centers) != 2: return False, -1
    dist0 = color_distance(color, team_centers[0]); dist1 = color_distance(color, team_centers[1])
    min_dist = min(dist0, dist1); closest_team_idx = 0 if dist0 <= dist1 else 1
    if min_dist > color_threshold: return False, -1
    try:
        color_bgr = np.uint8([[color]]); color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
        saturation = color_hsv[1] / 255.0
        if saturation < MIN_COLOR_SATURATION: return False, -1
    except Exception: return False, -1
    team_samples = team_colors_array[closest_team_idx]
    if team_samples and len(team_samples) >= 3:
        team_colors_np = np.array(team_samples); color_np = np.array(color); team_mean = np.mean(team_colors_np, axis=0)
        color_deviation = np.linalg.norm(color_np - team_mean)
        if color_deviation > COLOR_CONSISTENCY_THRESHOLD: return False, -1
    return True, closest_team_idx


# --- Team Assignment & Visualization Function ---
# This function remains the same as the previous version
def assign_and_visualize_teams(
    frame, detections_with_tracking, track_history, player_colors_history,
    team_assignments, is_static, current_team_centers,
    cluster_to_display_color, cluster_to_display_label, is_player_referee,
    max_color_distance_threshold, team_colors_array, fence_positions, horizontal_lines,
    team_classifier=None, classifier_trained=False
    ):
    """Draws ellipses and labels, using classifier first if available."""
    output_frame = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    if DISPLAY_FENCE_LINES:
        # Code to draw fence lines ... (same as before)
        if USE_HORIZONTAL_LINE_FILTERING and FENCE_DETECTION_METHOD == "position_based":
            for fence in fence_positions:
                fence_y = int(fence['y_percent'] * frame_h)
                cv2.line(output_frame, (0, fence_y), (frame_w, fence_y), (0, 255, 255), 1)
        if USE_HORIZONTAL_LINE_FILTERING and FENCE_DETECTION_METHOD == "line_detection":
             for x1l, y1l, x2l, y2l in horizontal_lines:
                 cv2.line(output_frame, (x1l, y1l), (x2l, y2l), (0, 255, 255), 1)


    if hasattr(detections_with_tracking, 'tracker_id') and detections_with_tracking.tracker_id is not None:
        for i, box in enumerate(detections_with_tracking.xyxy):
            track_id = detections_with_tracking.tracker_id[i]
            if track_id is None: continue
            track_id = int(track_id)

            display_label = f"ID:{track_id}"; display_color = DISPLAY_WHITE
            x1, y1, x2, y2 = map(int, box)
            height = y2 - y1
            box_center_y = (y1 + y2) / 2

            # Filtering Logic (Fence, Out of Field, Static, Referee)
            if is_near_fence_line(box, frame_h, frame_w, fence_positions, horizontal_lines): continue
            in_field = is_in_field_region(box, frame_h, frame_w, FIELD_REGION)
            if not in_field and box_center_y < FIELD_REGION['top'] * frame_h: continue
            if track_id in is_static: display_label = f"ID:{track_id} Static"; display_color = DISPLAY_GREY
            elif track_id in is_player_referee: display_label = f"ID:{track_id} Referee"; display_color = DISPLAY_GREY
            else: # Potential team player
                is_moving_like_player = has_player_movement_pattern(track_history[track_id], min_window=3)
                if not is_moving_like_player: display_label = f"ID:{track_id} Filtered"; display_color = DISPLAY_GREY
                else:
                    # Assignment Logic (Classifier -> K-Means Fallback)
                    assigned_label = team_assignments.get(track_id, "Filtered")
                    classification_success = False
                    if classifier_trained and team_classifier is not None:
                        jersey_roi = extract_jersey_roi(frame, box)
                        if jersey_roi is not None:
                            hist_features = extract_color_histogram(jersey_roi)
                            if hist_features is not None:
                                try:
                                    team_pred_proba = team_classifier.predict_proba([hist_features])[0]
                                    max_proba = np.max(team_pred_proba)
                                    if max_proba > 0.70:
                                        team_pred_label = team_classifier.predict([hist_features])[0]
                                        if team_pred_label == 0: assigned_label = "Blue Team"; classification_success = True
                                        elif team_pred_label == 1: assigned_label = "Red Team"; classification_success = True
                                        team_assignments[track_id] = assigned_label
                                except Exception: pass

                    if not classification_success and assigned_label == "Filtered" and len(current_team_centers) == 2:
                        smoothed_color = get_smoothed_color(player_colors_history[track_id])
                        if smoothed_color:
                            is_valid, assigned_cluster = is_valid_team_color(smoothed_color, current_team_centers, max_color_distance_threshold, team_colors_array)
                            if is_valid and assigned_cluster >= 0:
                                potential_label = cluster_to_display_label.get(assigned_cluster)
                                if potential_label: assigned_label = potential_label; team_assignments[track_id] = assigned_label

                    if assigned_label == "Red Team": display_color = DISPLAY_RED
                    elif assigned_label == "Blue Team": display_color = DISPLAY_BLUE
                    else: display_color = DISPLAY_GREY
                    display_label = f"ID:{track_id} {assigned_label}"

            # Drawing Logic
            center_x = int((box[0] + box[2]) / 2); bottom_y = int(box[3]); ellipse_axes = (20, 6)
            cv2.ellipse(output_frame, (center_x, bottom_y), ellipse_axes, 0, 0, 360, display_color, 2)
            cv2.putText(output_frame, display_label, (center_x - 30, bottom_y + ellipse_axes[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_color, 1)

    return output_frame


# ==============================================================
#           --- Main Video Processing Function ---
# ==============================================================

def process_video(input_video_path, output_folder):
    """Processes the input video for player detection, tracking, and team assignment."""
    log_prefix = "[PROCESS_VIDEO] "
    print(f"{log_prefix}Starting processing for: {input_video_path}")
    print(f"{log_prefix}Using YOLO Model specified in detection.py, Confidence: {CONFIDENCE_THRESHOLD}")
    start_total_time = time.time()

    # --- Initialization --- (Same as before)
    cap = None; video_writer = None
    processed_frame_count = 0; frame_num = -1
    output_video_path = None
    track_history = defaultdict(lambda: deque(maxlen=MOVEMENT_WINDOW_SIZE))
    player_colors_history = defaultdict(lambda: deque(maxlen=10))
    team_assignments = {}
    is_static = set()
    is_player_referee = {}
    current_team_centers = []
    team_colors_array = [[], []]
    cluster_to_display_color = {0: DISPLAY_BLUE, 1: DISPLAY_RED}
    cluster_to_display_label = {0: "Blue Team", 1: "Red Team"}
    team_training_data = []
    team_classifier = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
    classifier_trained = False

    # --- Main Try Block ---
    try:
        # Model loading happens in detection.py now

        # 2. Setup Video Capture (Same as before)
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened(): raise IOError(f"Cannot open video file: {input_video_path}")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_read = cap.get(cv2.CAP_PROP_FPS); fps = int(fps_read) if fps_read > 0 else 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{log_prefix}Video Info: {frame_width}x{frame_height}, {fps} FPS, ~{total_frames} Frames")

        # 3. Initialize Tracker (Same as before)
        byte_tracker = sv.ByteTrack(frame_rate=fps)
        print(f"{log_prefix}ByteTrack initialized.")

        # 4. Setup Video Writer (Same as before)
        output_filename = f"{os.path.splitext(os.path.basename(input_video_path))[0]}_output_v11_fix1.mp4" # Updated name
        output_video_path = os.path.join(output_folder, output_filename)
        os.makedirs(output_folder, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'); video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened(): raise IOError(f"Failed to open VideoWriter for path: {output_video_path}")
        print(f"{log_prefix}Output video writer ready: {output_video_path}")
        print(f"{log_prefix}Fence positions configured at Y% {[f['y_percent'] for f in FENCE_POSITIONS]}")
        print(f"{log_prefix}Field region set Y%=({FIELD_REGION['top']}-{FIELD_REGION['bottom']}), X%=({FIELD_REGION['left']}-{FIELD_REGION['right']})")
        print(f"{log_prefix}Fence lines will be {'shown' if DISPLAY_FENCE_LINES else 'hidden'}")

        # --- Frame Processing Loop ---
        while True:
            frame_num += 1; ret, frame = cap.read()
            if not ret: print(f"\n{log_prefix}End of video stream reached."); break
            if frame_num % VIDEO_PROCESSING_FRAME_INTERVAL != 0: continue

            loop_start_time = time.time()
            horizontal_lines = detect_horizontal_lines(frame)

            # 1. Object Detection (Same as before)
            detections_sv = detect_objects(frame, confidence_threshold=CONFIDENCE_THRESHOLD)

            # 1b. Filter before tracking (Same as before)
            filtered_indices = []
            for i, box in enumerate(detections_sv.xyxy):
                 height = box[3] - box[1]
                 if height < MIN_PLAYER_HEIGHT_PIXELS: continue
                 if is_near_fence_line(box, frame_height, frame_width, FENCE_POSITIONS, horizontal_lines): continue
                 filtered_indices.append(i)
            detections_for_tracking = detections_sv[filtered_indices]

            # 2. Tracking (Same as before)
            try:
                detections_with_tracking = byte_tracker.update_with_detections(detections=detections_for_tracking)
            except Exception as e_trk:
                print(f"{log_prefix}Frame {frame_num}: Error during tracker update: {e_trk}")
                detections_with_tracking = sv.Detections.empty()

            # 3. Movement Analysis & Update Colors/Features (Same as before)
            current_moving_tracker_ids = set(); current_frame_track_ids = set()
            if hasattr(detections_with_tracking, 'tracker_id') and detections_with_tracking.tracker_id is not None:
                centers = detections_with_tracking.get_anchors_coordinates(anchor=sv.Position.CENTER)
                valid_track_indices = [i for i, tid in enumerate(detections_with_tracking.tracker_id) if tid is not None]
                for i in valid_track_indices:
                    # ... (logic for track_history, is_static, is_referee, feature collection - same as before)
                    track_id = int(detections_with_tracking.tracker_id[i]); current_frame_track_ids.add(track_id)
                    box = detections_with_tracking.xyxy[i]; center_point = tuple(map(int, centers[i]))
                    track_history[track_id].append(center_point)
                    is_track_static = False
                    if len(track_history[track_id]) >= MOVEMENT_WINDOW_SIZE:
                        displacement = color_distance(track_history[track_id][0], track_history[track_id][-1])
                        if displacement < MOVEMENT_STATIC_THRESHOLD: is_static.add(track_id); is_track_static = True
                        else: is_static.discard(track_id); current_moving_tracker_ids.add(track_id)
                    elif track_id not in is_static: current_moving_tracker_ids.add(track_id)
                    if not is_track_static and is_in_field_region(box, frame_height, frame_width, FIELD_REGION):
                        roi = extract_jersey_roi(frame, box)
                        if roi is not None:
                            if is_referee(roi): is_player_referee[track_id] = True; team_assignments[track_id] = "Referee"
                            elif track_id not in is_player_referee:
                                rep_color = get_representative_color(roi)
                                if rep_color is not None: player_colors_history[track_id].append(rep_color)
                                hist_features = extract_color_histogram(roi)
                                if hist_features is not None:
                                    current_label = team_assignments.get(track_id); label_to_int = -1
                                    if current_label == "Blue Team": label_to_int = 0
                                    elif current_label == "Red Team": label_to_int = 1
                                    if label_to_int != -1: team_training_data.append((hist_features, label_to_int))


            # Remove disappeared tracks (Same as before)
            all_known_ids = set(track_history.keys()); disappeared_ids = all_known_ids - current_frame_track_ids
            for old_id in disappeared_ids:
                 track_history.pop(old_id, None); player_colors_history.pop(old_id, None)
                 team_assignments.pop(old_id, None); is_static.discard(old_id); is_player_referee.pop(old_id, None)

            # 4. Periodic Team Re-clustering and Classifier Training
            if (frame_num % TEAM_RECLUSTER_INTERVAL == 0 or frame_num < 30) and len(current_moving_tracker_ids) > 0:
                print(f"\n--- Frame {frame_num}: Re-clustering & Classifier Update ---")
                colors_for_clustering = []; tracks_for_clustering = []
                team_colors_array = [[], []]

                # --- FIX #1 START ---
                # Gather smoothed colors from currently valid players
                for tid in current_moving_tracker_ids:
                    if (tid not in is_player_referee and tid not in is_static and
                        has_player_movement_pattern(track_history[tid], min_window=3)):

                        track_found_in_frame = False
                        # Safely check and iterate through current tracks
                        if detections_with_tracking.tracker_id is not None: # Check if None first
                            for i_f, tid_check in enumerate(detections_with_tracking.tracker_id):
                                 if tid_check is not None and int(tid_check) == tid:
                                     # Added check for xyxy length consistency
                                     if i_f < len(detections_with_tracking.xyxy):
                                         box_f = detections_with_tracking.xyxy[i_f]
                                         if is_in_field_region(box_f, frame_height, frame_width, FIELD_REGION):
                                              track_found_in_frame = True
                                     break # Found the track, exit inner loop
                                 # Else: Continue searching if tid_check is None or doesn't match

                        if track_found_in_frame:
                            smoothed_color = get_smoothed_color(player_colors_history[tid])
                            if smoothed_color is not None:
                                colors_for_clustering.append(smoothed_color)
                                tracks_for_clustering.append(tid)
                # --- FIX #1 END ---

                print(f"Clustering {len(colors_for_clustering)} smoothed colors...")

                if len(colors_for_clustering) >= 4: # K-Means logic start
                    cluster_colors_np = np.array(colors_for_clustering).astype(np.float32)
                    unique_colors = np.unique(cluster_colors_np, axis=0)
                    if len(unique_colors) >= 2:
                        try:
                            # K-Means fitting and center processing (same as before)
                            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(cluster_colors_np)
                            new_team_centers = [tuple(map(int, center)) for center in kmeans.cluster_centers_]
                            if color_distance(new_team_centers[0], new_team_centers[1]) > 30:
                                current_team_centers = new_team_centers; print(f"K-Means OK. Centers: {current_team_centers}")
                            else: print("Warning: K-Means centers too close.")
                            if current_team_centers:
                                c0, c1 = current_team_centers[0], current_team_centers[1]; ratio0 = c0[2]/(c0[0]+1.0); ratio1 = c1[2]/(c1[0]+1.0)
                                if ratio0 > ratio1: cluster_to_display_label={0:"Red Team",1:"Blue Team"}; cluster_to_display_color={0:DISPLAY_RED,1:DISPLAY_BLUE}
                                else: cluster_to_display_label={0:"Blue Team",1:"Red Team"}; cluster_to_display_color={0:DISPLAY_BLUE,1:DISPLAY_RED}
                                print(f"Team mapping: 0->{cluster_to_display_label[0]}, 1->{cluster_to_display_label[1]}")

                                # Populate team_colors_array (same as before)
                                kmeans_labels = kmeans.predict(cluster_colors_np)
                                for i_k, tid_k in enumerate(tracks_for_clustering):
                                    assigned_cluster=kmeans_labels[i_k]; smoothed_color=colors_for_clustering[i_k]; dist_to_center=color_distance(smoothed_color, current_team_centers[assigned_cluster])
                                    if dist_to_center <= MAX_COLOR_DISTANCE_THRESHOLD*0.7:
                                         if len(team_colors_array[assigned_cluster]) < 20: team_colors_array[assigned_cluster].append(smoothed_color)

                                # --- FIX #2 START ---
                                # Re-assign *all* currently moving players
                                assigned_count = 0
                                for tid_assign in current_moving_tracker_ids:
                                     if tid_assign in is_player_referee or tid_assign in is_static: continue
                                     in_field_now = False
                                     # Safely check and iterate through current tracks
                                     if detections_with_tracking.tracker_id is not None: # Check if None first
                                         for i_fc, tid_check_fc in enumerate(detections_with_tracking.tracker_id):
                                             if tid_check_fc is not None and int(tid_check_fc) == tid_assign:
                                                 # Added check for xyxy length consistency
                                                 if i_fc < len(detections_with_tracking.xyxy):
                                                     box_fc = detections_with_tracking.xyxy[i_fc]
                                                     in_field_now = is_in_field_region(box_fc, frame_height, frame_width, FIELD_REGION)
                                                 break # Found track, exit inner loop
                                             # Else: Continue searching if tid_check is None or doesn't match

                                     # --- FIX #2 END ---

                                     if not in_field_now: team_assignments[tid_assign]="Filtered"; continue # Skip if not in field
                                     s_color = get_smoothed_color(player_colors_history[tid_assign])
                                     if s_color:
                                         is_valid, c_cluster = is_valid_team_color(s_color, current_team_centers, MAX_COLOR_DISTANCE_THRESHOLD, team_colors_array)
                                         if is_valid and c_cluster >= 0: team_assignments[tid_assign]=cluster_to_display_label.get(c_cluster, "Filtered"); assigned_count+=1
                                         else: team_assignments[tid_assign]="Filtered"
                                     else: team_assignments[tid_assign]="Filtered"
                                print(f"Re-assigned {assigned_count} players via K-Means distance.")
                        except Exception as e_kmeans: print(f"K-Means Exception: {e_kmeans}"); traceback.print_exc()
                    else: print("Not enough unique colors for K-Means.")
                else: print("Not enough valid colors for K-Means.")

                # Classifier Training Logic (Same as before)
                if len(team_training_data) >= 10:
                    labels_in_data = [d[1] for d in team_training_data]
                    if 0 in labels_in_data and 1 in labels_in_data:
                        try:
                            print(f"Attempting train: {len(team_training_data)} samples..."); X_train = np.array([d[0] for d in team_training_data]); y_train = np.array(labels_in_data)
                            team_classifier.fit(X_train, y_train); classifier_trained = True; print("Classifier trained.")
                        except Exception as e_train: print(f"Error training: {e_train}"); classifier_trained = False
                    else: print("Need samples for label 0 and 1.")
                else: print(f"Need more samples ({len(team_training_data)} < 10).")


            # 5. Visualization (Same as before)
            annotated_frame = assign_and_visualize_teams(
                frame, detections_with_tracking, track_history, player_colors_history,
                team_assignments, is_static, current_team_centers,
                cluster_to_display_color, cluster_to_display_label, is_player_referee,
                MAX_COLOR_DISTANCE_THRESHOLD, team_colors_array, FENCE_POSITIONS, horizontal_lines,
                team_classifier, classifier_trained
            )

            # 6. Write Frame (Same as before)
            if video_writer: video_writer.write(annotated_frame)
            processed_frame_count += 1

            # Progress Update (Same as before)
            loop_time = time.time() - loop_start_time
            if frame_num % 30 == 0 or frame_num < 10:
                elapsed_total = time.time() - start_total_time; est_rem_str="N/A"
                if total_frames > 0 and frame_num > 0: est_rem_secs=(elapsed_total / (frame_num + 1)) * (total_frames - frame_num - 1); est_rem_str=f"~{est_rem_secs:.0f}s"
                moving_ids_count=len(current_moving_tracker_ids); print(f"{log_prefix}Frame {frame_num}/{total_frames or '?'} ({loop_time:.3f}s) | Moving IDs: {moving_ids_count} | Est. Rem: {est_rem_str}   ", end='\r')


        # <<< END OF WHILE LOOP >>>
        print(f"\n{log_prefix}Video processing loop finished.")

    # --- Exception Handling & Cleanup --- (Same as before)
    except Exception as e:
        print(f"\n{log_prefix}🚨 An error occurred: {e}")
        traceback.print_exc()
        status_message = f"Error: {e}"
        if output_video_path and os.path.exists(output_video_path):
             try:
                 if video_writer and video_writer.isOpened(): video_writer.release()
                 os.remove(output_video_path); print(f"{log_prefix}Removed incomplete file.")
             except Exception as e_del: print(f"Error removing file: {e_del}")
        output_video_path = None
        return output_video_path, status_message
    finally:
        if cap: cap.release(); print(f"{log_prefix}Capture released.")
        if video_writer: video_writer.release(); print(f"{log_prefix}Writer released.")
        total_end_time = time.time(); total_duration = total_end_time - start_total_time
        print(f"\n{log_prefix}Total time: {total_duration:.2f}s.")
        print(f"{log_prefix}Processed {processed_frame_count} frames.")
        if output_video_path and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 1024:
            print(f"{log_prefix}Output saved: {output_video_path}")
            status_message = "Processing successful."; return output_video_path, status_message
        elif output_video_path and os.path.exists(output_video_path):
             print(f"{log_prefix}Error: Output file empty/corrupt."); status_message = "Error: Output file corrupt."; return None, status_message
        else:
             print(f"{log_prefix}Error: Output file not created."); status_message = "Error: Output file not created."; return None, status_message