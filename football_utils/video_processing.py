# football_utils/video_processing.py

import cv2
import os
import time
import numpy as np
import traceback
import glob
import shutil # For removing temporary directory
import subprocess # For running ffmpeg

# --- Only import detection for this simplified test ---
from football_utils.detection import detect_objects
# from football_utils.tracking import calculate_optical_flow, apply_perspective_transform # SKIPPED
# from football_utils.team_assignment import assign_teams # SKIPPED
from football_utils.llm_inference import classify_output # Keep LLM part

def process_video(input_video_path, output_folder):
    log_prefix = "[PROCESS_VIDEO_FFMPEG_SIMPLE] "
    print(f"{log_prefix}Starting simplified process for: {input_video_path}")
    print(f"{log_prefix}Output folder: {output_folder}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"{log_prefix}Error: Could not open input video.")
        return None, "Error: Could not open input video"

    # --- Get video properties ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = int(frame_count_prop) if frame_count_prop > 0 else 0
    print(f"{log_prefix}Video properties: {width}x{height} @ {fps:.2f} FPS, Frames: ~{frame_count if frame_count>0 else 'Unknown'}")

    if width <= 0 or height <= 0 or fps <= 0:
         print(f"{log_prefix}Error: Invalid video properties.")
         cap.release()
         return None, "Error: Invalid video properties"

    # --- Create Temporary Directory for Frames ---
    timestamp = int(time.time())
    temp_frame_folder = os.path.join(output_folder, f"temp_frames_{timestamp}")
    try:
        # Remove existing temp folder if it exists from a failed run
        if os.path.exists(temp_frame_folder):
             print(f"{log_prefix}Removing existing temp folder: {temp_frame_folder}")
             shutil.rmtree(temp_frame_folder)
        os.makedirs(temp_frame_folder, exist_ok=True)
        print(f"{log_prefix}Created temporary frame folder: {temp_frame_folder}")
    except OSError as e:
        print(f"{log_prefix}Error creating temporary directory {temp_frame_folder}: {e}")
        cap.release()
        return None, f"Error creating temp directory: {e}"

    # --- Process Frames (Detection + Overlay ONLY) and Save ---
    frame_num = 0
    processing_successful = True
    saved_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"{log_prefix}End of stream after {frame_num} frames read.")
                break

            frame_num += 1
            # Basic frame validation
            if frame is None or frame.shape[0] != height or frame.shape[1] != width or frame.dtype != np.uint8:
                print(f"{log_prefix}Invalid frame {frame_num}, skipping save.")
                continue

            try:
                # --- Perform ONLY Detection ---
                detections = detect_objects(frame)
                # --- SKIP team_assignments ---
                # --- SKIP calculate_optical_flow ---
                # --- SKIP apply_perspective_transform ---

                # Draw overlays on a copy
                display_frame = frame.copy()
                if detections:
                     for i, det in enumerate(detections):
                         if isinstance(det, dict) and all(k in det for k in ('xmin', 'ymin', 'xmax', 'ymax', 'name')):
                             xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                             label = det['name']
                             # No team info added in this simplified version
                             cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                             cv2.putText(display_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # ------------------------------------

                # --- Save Processed Frame as Image ---
                frame_filename = os.path.join(temp_frame_folder, f"frame_{frame_num:06d}.jpg")
                success = cv2.imwrite(frame_filename, display_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    print(f"{log_prefix}Error: Failed to write frame {frame_num} to {frame_filename}")
                    # processing_successful = False # Decide if one failure stops all
                else:
                    saved_frame_count += 1
                    if saved_frame_count % 100 == 0:
                         print(f"{log_prefix}Saved frame {saved_frame_count} (Input frame {frame_num})")
                # ------------------------------------

            except Exception as frame_proc_err:
                 print(f"{log_prefix}Error processing frame {frame_num}: {frame_proc_err}")
                 print(traceback.format_exc())
                 processing_successful = False # Mark as failed if any frame error occurs

        # End of while loop

    except Exception as loop_err:
         print(f"{log_prefix}Error in main loop: {loop_err}")
         print(traceback.format_exc())
         processing_successful = False

    finally:
        # --- Release Video Capture ---
        print(f"{log_prefix}Releasing video capture...")
        if cap is not None: cap.release()
        print(f"{log_prefix}Video capture released.")
        # ---------------------------

    # --- Assemble Video using ffmpeg ---
    classification_str = "N/A" # Default
    final_output_path = None

    if processing_successful and saved_frame_count > 0:
        print(f"{log_prefix}Processing loop finished. Assembling video from {saved_frame_count} saved frames...")
        output_video_name = f"output_ffmpeg_{timestamp}.mp4" # Create final MP4
        output_video_path = os.path.join(output_folder, output_video_name)

        # Construct ffmpeg command (ensure quotes around paths)
        ffmpeg_cmd = (
            f"ffmpeg -y -framerate {fps} -start_number 1 -i \"{temp_frame_folder}/frame_%06d.jpg\" "
            f"-c:v libx264 -preset medium -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" "
            f"\"{output_video_path}\""
        )

        print(f"{log_prefix}Executing ffmpeg command:\n{ffmpeg_cmd}")
        try:
            result = subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            print(f"{log_prefix}--- ffmpeg stdout ---")
            print(result.stdout if result.stdout else "(empty)")
            print(f"{log_prefix}--- ffmpeg stderr ---")
            print(result.stderr if result.stderr else "(empty)") # ffmpeg often prints info to stderr
            print(f"{log_prefix}--- ffmpeg assembly finished ---")

            # Verify output file
            if os.path.exists(output_video_path):
                 file_size = os.path.getsize(output_video_path)
                 if file_size > 1024: # Check > 1KB
                      print(f"{log_prefix}Output MP4 video file VERIFIED: {output_video_path}, Size: {file_size} bytes")
                      final_output_path = output_video_path
                 else:
                      print(f"{log_prefix}Error: ffmpeg created an empty/tiny MP4 file ({file_size} bytes).")
            else:
                 print(f"{log_prefix}Error: Output MP4 file not found after ffmpeg execution.")

        except subprocess.CalledProcessError as e:
            print(f"{log_prefix}Error: ffmpeg command failed with exit code {e.returncode}")
            print(f"  Stderr: {e.stderr}")
        except FileNotFoundError:
             print(f"{log_prefix}Error: 'ffmpeg' command not found. Make sure ffmpeg is installed in Cell 1.")
        except Exception as e:
            print(f"{log_prefix}An unexpected error occurred during ffmpeg execution: {e}")
            print(traceback.format_exc())

    else:
        print(f"{log_prefix}Video processing loop did not complete successfully or no frames were saved to assemble.")

    # --- Cleanup Temporary Frames ---
    if os.path.exists(temp_frame_folder):
        print(f"{log_prefix}Cleaning up temporary frame directory: {temp_frame_folder}")
        try:
            shutil.rmtree(temp_frame_folder)
        except OSError as e:
            print(f"{log_prefix}Error removing temporary directory: {e}")
    # --------------------------------

    # Run LLM Classification (using simple summary)
    try:
        print(f"{log_prefix}Running LLM Classification...")
        summary_text = f"Simplified analysis complete. Detected objects in {saved_frame_count} frames."
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