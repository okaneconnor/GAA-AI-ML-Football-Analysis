import cv2
import os
import time
import traceback

def process_video(input_video_path, output_folder):
    print(f"[SIMPLIFIED_PROCESS] Starting for: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[SIMPLIFIED_PROCESS] Error: Could not open input video.")
        return None, "Input Video Error"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SIMPLIFIED_PROCESS] Input Props: {width}x{height} @ {fps:.2f} FPS")

    if width <= 0 or height <= 0 or fps <= 0:
         print(f"[SIMPLIFIED_PROCESS] Error: Invalid video properties.")
         cap.release()
         return None, "Invalid Properties"

    # Use AVI / XVID as it's often simpler/more robust
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = f"test_output_{int(time.time())}.avi"
    output_path = os.path.join(output_folder, output_filename)
    print(f"[SIMPLIFIED_PROCESS] Attempting to write to: {output_path}")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"[SIMPLIFIED_PROCESS] Error: Failed to open VideoWriter.")
        cap.release()
        return None, "VideoWriter Open Error"

    frame_count = 0
    write_success = True
    final_output_path = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame is None:
                 print(f"[SIMPLIFIED_PROCESS] Read None frame at index {frame_count}, skipping.")
                 continue

            # *** NO PROCESSING - Just write the raw frame ***
            try:
                out.write(frame)
            except Exception as e:
                 print(f"[SIMPLIFIED_PROCESS] Error writing frame {frame_count}: {e}")
                 print(traceback.format_exc())
                 write_success = False
                 break # Stop if writing fails

        print(f"[SIMPLIFIED_PROCESS] Finished reading {frame_count} frames.")

    except Exception as e:
        print(f"[SIMPLIFIED_PROCESS] Error during read/write loop: {e}")
        print(traceback.format_exc())
        write_success = False
    finally:
        print("[SIMPLIFIED_PROCESS] Releasing resources...")
        cap.release()
        out.release() # Essential to finalize the file
        print("[SIMPLIFIED_PROCESS] Resources released.")

    # Check if file was created
    if write_success and os.path.exists(output_path):
        try:
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                print(f"[SIMPLIFIED_PROCESS] Output file successfully created: {output_path} ({file_size} bytes)")
                final_output_path = output_path
            else:
                print(f"[SIMPLIFIED_PROCESS] Error: Output file exists but is empty: {output_path}")
        except OSError as e:
            print(f"[SIMPLIFIED_PROCESS] Error checking output file size: {e}")
    elif write_success:
         print(f"[SIMPLIFIED_PROCESS] Error: Loop finished but output file not found: {output_path}")
    else:
         print(f"[SIMPLIFIED_PROCESS] Error: Loop did not complete successfully or writing failed.")
         # Clean up potentially corrupt file
         if os.path.exists(output_path):
             try: os.remove(output_path)
             except OSError as e: print(f"Error removing temp file: {e}")


    # Return dummy classification for compatibility with Cell 3 structure
    return final_output_path, "Simplified Test Run"