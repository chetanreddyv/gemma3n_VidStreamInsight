import cv2
import time
import pathlib
import tempfile

def capture_frames(duration_seconds, fps):
    """
    Captures frames from the default camera for a given duration.
    Returns the temporary directory path and a list of frame file paths.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return None, []

    temp_dir = tempfile.mkdtemp(prefix="live_frames_")
    frame_paths = []
    start_time = time.time()
    frame_interval = 1.0 / fps
    next_frame_time = start_time

    print(f"Capturing video for {duration_seconds} seconds...")
    while time.time() - start_time < duration_seconds:
        if time.time() >= next_frame_time:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = pathlib.Path(temp_dir) / f"frame_{len(frame_paths):04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            next_frame_time += frame_interval
        
        time.sleep(0.01) # Prevent busy-waiting

    cap.release()
    print(f"Captured {len(frame_paths)} frames.")
    return temp_dir, frame_paths