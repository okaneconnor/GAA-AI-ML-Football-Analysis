# football_utils/video_processing.py

import cv2
import os
import time
import numpy as np
import traceback

# Import necessary functions
from football_utils.detection import detect_objects
from football_utils.tracking import calculate_optical_flow, apply_perspective_transform
from football_utils.team_assignment import assign_teams # Import the updated function
from football_utils.llm_inference import classify_output

# --- Configuration ---
# Set desired confidence threshold for 'person' detection
PERSON_CONF_THRESHOLD = 0.4 # Adjust as needed (0.0 to 1.0)
# Simple color check for potential referees (assuming they wear black/very dark)
# Check average BGR color of the central part of the box
REFEREE_DARK_THRESHOLD = 50 # Max average BGR value to be considered 'dark'
# -------------------

def is_likely_referee(roi):
    """ Simple check based on average color being very dark """
    if roi.size == 0: return False
    avg_color = np.mean(roi.reshape(-1, 3), axis=0)
    # Check if average B, G, and R are all below the threshold
    if np.all(avg_color < REFEREE_DARK_THRESHOLD):
        # print(f"Potential referee detected, avg color: {avg_color}") # Debug
        return True
    return False

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
    # ... (get frame_count) ...
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
                # 1. Object Detection
                all_detections = detect_objects(frame)

                # --- 2. FILTER Detections ---
                player_detections = [] # Only players go here
                player_indices_map = {} # Map filtered list index back to original index
                original_indices = {}   # Map original index to filtered list index

                for i, det in enumerate(all_detections):
                     # Check it's a person above confidence threshold
                     if det['name'] == 'person' and det['confidence'] >= PERSON_CONF_THRESHOLD:
                          # --- Simple Referee Check ---
                          xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                          # Take central ROI to check color (avoid boots/head)
                          box_h = ymax - ymin; box_w = xmax - xmin
                          roi_ymin_ref = ymin + int(box_h * 0.2); roi_ymax_ref = ymax - int(box_h * 0.2)
                          roi_xmin_ref = xmin + int(box_w * 0.2); roi_xmax_ref = xmax - int(box_w * 0.2)
                          if roi_ymin_ref < roi_ymax_ref and roi_xmin_ref < roi_xmax_ref: # Check valid ROI dims
                              roi_ref = frame[roi_ymin_ref:roi_ymax_ref, roi_xmin_ref:roi_xmax_ref]
                              if not is_likely_referee(roi_ref):
                                   # If not likely a referee, add to player list
                                   filtered_idx = len(player_detections) # Index in the filtered list
                                   player_detections.append(det)
                                   player_indices_map[filtered_idx] = i # Map filtered index -> original index
                                   original_indices[i] = filtered_idx # Map original index -> filtered index
                              # else: print(f"Filtered potential referee at original index {i}") # Debug
                          else:
                              # If ROI is invalid, assume it's a player for now
                              filtered_idx = len(player_detections)
                              player_detections.append(det)
                              player_indices_map[filtered_idx] = i
                              original_indices[i] = filtered_idx

                # --------------------------

                # 3. Team Assignment (using filtered player detections)
                # assign_teams expects list of detections; returns dict mapping *filtered index* -> team_id
                team_assignments_filtered_idx = assign_teams(frame, player_detections)

                # Convert team assignment keys back to original detection indices for drawing
                team_assignments_original_idx = {
                    player_indices_map[filtered_idx]: team_id
                    for filtered_idx, team_id in team_assignments_filtered_idx.items()
                    if filtered_idx in player_indices_map # Ensure mapping exists
                }
                # print(f"Orig Idx Teams: {team_assignments_original_idx}") # Debug

                # --- 4. Optical Flow (Optional - consider disabling) ---
                flow = calculate_optical_flow(prev_frame, frame) if prev_frame is not None else None

                # 5. Perspective Transformation (Identity)
                # ... (transform code if needed) ...

                # --- 6. Overlay Filtered Detections + Teams ---
                display_frame = frame.copy() # Draw on a copy
                # Iterate through the ORIGINAL detections list
                for i, det in enumerate(all_detections):
                     # Only draw if it was classified as a player (i.e., exists in original_indices map)
                     if i in original_indices:
                         xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                         label = det['name'] # Should be 'person'
                         conf = det['confidence']
                         # Add team info using the converted original index map
                         if i in team_assignments_original_idx:
                             team_id = team_assignments_original_idx[i]
                             label += f" | Team {team_id}"
                             # Optional: color code bbox by team
                             color = (255, 0, 0) if team_id == 0 else (0, 0, 255) # Blue/Red example
                         else:
                              # Player detected but not assigned a team (maybe color extraction failed?)
                              color = (0, 255, 0) # Default green

                         cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                         cv2.putText(display_frame, f"{label} ({conf:.2f})", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # -----------------------------------------------


                # --- Write the processed frame ---
                try:
                    out.write(display_frame)
                except Exception as write_err:
                    # ... (handle write errors as before) ...
                    write_errors += 1; print(...)

                prev_frame = frame.copy()

            except Exception as frame_proc_err:
                # ... (handle frame processing errors) ...
                print(f"{log_prefix}Error processing frame {frame_num}: {frame_proc_err}")
                processing_successful = False # Mark as failed if one frame fails badly

        # End of while loop
        print(f"{log_prefix}Finished loop. Frames read: {frame_num}, Write errors: {write_errors}")

    except Exception as loop_err:
        # ... (handle outer loop errors) ...
        print(f"{log_prefix}Error in loop: {loop_err}"); processing_successful = False
    finally:
        # --- Release Resources ---
        print(f"{log_prefix}Releasing resources...")
        if cap is not None: cap.release()
        if out is not None: out.release()
        print(f"{log_prefix}Resources released.")

    # --- Check output file ---
    final_output_path = None
    # Only consider valid if processing was okay AND file exists with size > 1KB AND no write errors
    if processing_successful and write_errors == 0 and os.path.exists(output_video_path):
        try:
            file_size = os.path.getsize(output_video_path)
            if file_size > 1024 :
                print(f"{log_prefix}Output MP4 file successfully created: {output_video_path} ({file_size} bytes)")
                final_output_path = output_video_path
            else: print(f"{log_prefix}Error: Output file is too small/empty ({file_size} bytes).")
        except OSError as e: print(f"{log_prefix}Error checking file size: {e}")
    # ... (Handle other failure cases, cleanup potentially bad files) ...
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
        # Make summary more informative if possible
        summary_text = f"Analysis complete. Processed {frame_num} frames. Detected players and assigned teams."
        classification = classify_output(summary_text)
        # ... (LLM result parsing) ...
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