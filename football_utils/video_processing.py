import cv2
import os
import time
import numpy as np
import traceback

# Assume these imports work and the functions exist
# Ensure detection.py is the corrected version using the ultralytics API
from football_utils.detection import detect_objects
from football_utils.tracking import calculate_optical_flow, apply_perspective_transform
from football_utils.team_assignment import assign_teams
from football_utils.llm_inference import classify_output

def process_video(input_video_path, output_folder):
    print(f"[PROCESS_VIDEO_MP4V] Starting for: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[PROCESS_VIDEO_MP4V] Error: Could not open input video.")
        return None, "Input Video Error"

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = int(frame_count_prop) if frame_count_prop > 0 else 0
    print(f"[PROCESS_VIDEO_MP4V] Input Props: {width}x{height} @ {fps:.2f} FPS, Frames: ~{frame_count if frame_count>0 else 'Unknown'}")

    if width <= 0 or height <= 0 or fps <= 0:
         print(f"[PROCESS_VIDEO_MP4V] Error: Invalid video properties.")
         cap.release()
         return None, "Invalid Properties"

    # --- Setup Video Writer using MP4V codec ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for MP4 output
    output_filename = f"output_{int(time.time())}.mp4" # Output MP4 file
    output_video_path = os.path.join(output_folder, output_filename)
    print(f"[PROCESS_VIDEO_MP4V] Attempting to write MP4 (mp4v) to: {output_video_path}")

    # Initialize VideoWriter to None initially
    out = None
    try:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        # Check if writer opened successfully
        if not out.isOpened():
            print(f"[PROCESS_VIDEO_MP4V] Error: Failed to open VideoWriter (mp4v/MP4). Check path/codec support.")
            cap.release()
            return None, "VideoWriter Open Error"
        else:
            print(f"[PROCESS_VIDEO_MP4V] VideoWriter opened successfully (mp4v).")

        # --- Process Frames (Original Logic + Safety) ---
        frame_num = 0
        prev_frame = None
        processing_successful = True # Assume success unless error occurs
        write_errors = 0 # Count write errors

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[PROCESS_VIDEO_MP4V] End of stream after {frame_num} frames read.")
                break # End of video or read error

            frame_num += 1
            # Initialize prev_frame on first valid frame
            if frame_num == 1:
                 # Basic check for first frame
                 if frame is None or frame.shape[0] != height or frame.shape[1] != width or frame.dtype != np.uint8:
                      print(f"[PROCESS_VIDEO_MP4V] Error: Invalid first frame.")
                      processing_successful = False
                      break
                 prev_frame = frame.copy()
                 print(f"[PROCESS_VIDEO_MP4V] Processing frame {frame_num}...")
            elif frame_num % 100 == 0:
                 print(f"[PROCESS_VIDEO_MP4V] Processing frame {frame_num}/{frame_count if frame_count>0 else '?'}")


            # Basic frame validation for subsequent frames
            if frame is None or frame.shape[0] != height or frame.shape[1] != width or frame.dtype != np.uint8:
                print(f"[PROCESS_VIDEO_MP4V] Invalid frame {frame_num}, skipping write.")
                continue # Skip processing for this frame

            try:
                # --- Perform ALL original processing steps ---
                # Ensure detection.py is using the corrected YOLOv11 loader
                detections = detect_objects(frame)
                team_assignments = assign_teams(frame, detections)
                flow = calculate_optical_flow(prev_frame, frame) if prev_frame is not None else None
                # Perspective transform (as in original)
                src_points = np.float32([[0,0], [width,0], [width,height], [0,height]])
                dst_points = np.float32([[0,0], [width,0], [width,height], [0,height]])
                transformed_frame = apply_perspective_transform(frame, src_points, dst_points) # Unused?

                # --- Draw overlays ON THE ORIGINAL FRAME ('frame') ---
                # (This modifies 'frame' directly before writing, as per the original example)
                display_frame = frame
                if detections:
                     for i, det in enumerate(detections):
                         # Added safety checks here
                         if isinstance(det, dict) and all(k in det for k in ('xmin', 'ymin', 'xmax', 'ymax', 'name')):
                             xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                             label = det['name']
                             # Added check for team_assignments being a dict
                             if label == "person" and isinstance(team_assignments, dict) and i in team_assignments:
                                  label += f" | Team {team_assignments[i]}"
                             # Draw directly onto display_frame (which is 'frame')
                             cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                             cv2.putText(display_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                         else:
                              print(f"[PROCESS_VIDEO_MP4V] Warning: Skipping malformed detection in frame {frame_num}, index {i}")

                # --- Write the processed (overlayed) frame ---
                try:
                    out.write(display_frame)
                except Exception as write_err:
                    write_errors += 1
                    if write_errors == 1: # Only print first error detail
                        print(f"[PROCESS_VIDEO_MP4V] Error writing frame {frame_num}: {write_err}")
                        print(traceback.format_exc())
                    elif write_errors % 100 == 0: # Print summary periodically
                         print(f"[PROCESS_VIDEO_MP4V] Encountered {write_errors} frame write errors...")
                    # Maybe don't break, let it try subsequent frames
                    # processing_successful = False # Mark overall process as potentially flawed
                    # break

                # Update previous frame AFTER using the current one for overlays/writing
                prev_frame = frame.copy()

            except Exception as frame_proc_err:
                 print(f"[PROCESS_VIDEO_MP4V] Error processing frame {frame_num}: {frame_proc_err}")
                 print(traceback.format_exc())
                 # Mark overall process as potentially flawed if one frame fails processing
                 processing_successful = False
                 # Decide if you want to stop entirely: break

        # End of while loop
        print(f"[PROCESS_VIDEO_MP4V] Finished reading/processing loop. Total frames read: {frame_num}, Write errors encountered: {write_errors}")

    except Exception as loop_err:
        print(f"[PROCESS_VIDEO_MP4V] Error during main read/process loop: {loop_err}")
        print(traceback.format_exc())
        processing_successful = False
    finally:
        # --- Ensure resources are released ---
        print("[PROCESS_VIDEO_MP4V] Releasing video capture and writer...")
        if cap is not None: cap.release()
        if out is not None: out.release() # Essential to finalize the file
        print("[PROCESS_VIDEO_MP4V] Resources released.")

    # --- Check if the output file was actually created and seems valid ---
    final_output_path = None # Default to None (failure)
    if processing_successful and os.path.exists(output_video_path):
        try:
            # Check file size only if processing seemed okay overall
            file_size = os.path.getsize(output_video_path)
            if file_size > 1024 and write_errors == 0: # Check > 1KB and no write errors reported
                print(f"[PROCESS_VIDEO_MP4V] Output MP4 file successfully created: {output_video_path} ({file_size} bytes)")
                final_output_path = output_video_path # Set path only if valid
            elif file_size <= 1024:
                print(f"[PROCESS_VIDEO_MP4V] Error: Output MP4 file exists but is too small or empty ({file_size} bytes). Likely corrupt.")
            else: # file_size > 1024 but write_errors > 0
                 print(f"[PROCESS_VIDEO_MP4V] Warning: Output MP4 file created ({file_size} bytes), but {write_errors} frame write errors occurred. File may be incomplete/corrupt.")
                 final_output_path = output_video_path # Return path but with warning
        except OSError as e:
            print(f"[PROCESS_VIDEO_MP4V] Error checking output file size: {e}")
    elif processing_successful: # but file doesn't exist
         print(f"[PROCESS_VIDEO_MP4V] Error: Processing finished but output MP4 file not found: {output_video_path}")
    else: # Processing loop failed
         print(f"[PROCESS_VIDEO_MP4V] Error: Processing loop failed before completion.")

     # Attempt cleanup if file exists but process likely failed or file is invalid
    if final_output_path is None and os.path.exists(output_video_path):
         print(f"[PROCESS_VIDEO_MP4V] Deleting potentially corrupt/empty file: {output_video_path}")
         try: os.remove(output_video_path)
         except OSError as e: print(f"Error deleting file: {e}")

    # --- LLM Classification ---
    # (LLM logic remains the same as your original example)
    classification_str = "N/A"
    try:
        print("[PROCESS_VIDEO_MP4V] Running LLM Classification...")
        summary_text = "Analysis complete. Player detection, team assignment, and movement tracking executed."
        classification = classify_output(summary_text)
        if classification and isinstance(classification, list) and len(classification) > 0 and isinstance(classification[0], dict):
            label = classification[0].get('label', 'N/A')
            score = classification[0].get('score', 0.0)
            classification_str = f"Label: {label}, Score: {score:.4f}"
        else: classification_str = str(classification)
        print(f"[PROCESS_VIDEO_MP4V] LLM Classification Result: {classification_str}")
    except Exception as llm_err:
         print(f"[PROCESS_VIDEO_MP4V] Error during LLM classification: {llm_err}")
         classification_str = "Error during LLM classification"

    print(f"[PROCESS_VIDEO_MP4V] Returning path: {final_output_path}, classification: {classification_str}")
    # Return the path ONLY if it was verified non-empty and likely okay
    return final_output_path, classification_str