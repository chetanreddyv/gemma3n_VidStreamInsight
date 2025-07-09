Excellent initiative. Fine-tuning a model on custom data is the most effective way to improve its performance for a specialized task like this. Here is a detailed approach for building the dataset and fine-tuning the model.

### Phase 1: Data Collection & Annotation

This is the most critical phase. The quality of your data will directly determine the quality of your fine-tuned model.

#### 1. Video & Audio Recording Strategy

*   **Perspective:** Record all videos from a **first-person point of view (FPV)**, as if you are the one walking. A chest-mounted phone or camera is ideal for stability and correct height.
*   **Environment Diversity:** Capture a wide range of scenarios:
    *   **Indoors:** Hallways, stairs, elevators, doorways, office spaces, navigating around furniture.
    *   **Outdoors:** Sidewalks, street crossings (with and without signals), parks, parking lots, building entrances.
    *   **Lighting:** Record during the day, at night, and in mixed/low-light conditions.
    *   **Weather:** Capture clear days, rainy conditions (if possible), and cloudy weather.
*   **Scenario Diversity:** Focus on recording situations that require specific instructions.
    *   **Obstacles:** People, poles, trash cans, benches, open car doors, low-hanging branches.
    *   **Hazards:** Curbs, stairs (up and down), puddles, uneven pavement, moving vehicles.
    *   **Navigation:** Turning left/right at corners, following a path, stopping at a door.
*   **Audio:** Ensure you capture clear audio. Ambient sounds (traffic, beeping crosswalks, conversations) are valuable context for the model.

#### 2. Annotation: Writing Good Instructions

Your goal is to create pairs of `(media_chunk, instruction)`. The instructions must be high-quality.

**Instruction Writing Rules:**
1.  **Be Direct & Actionable:** Start with a verb. The instruction should be something a person can do immediately.
    *   *Good:* `Stop, curb ahead.`
    *   *Bad:* `There is a curb in front of you.`
2.  **Be Concise:** Aim for **3-8 words**. A person needs to process it quickly.
    *   *Good:* `Turn right at the corner.`
    *   *Bad:* `Okay, now you should get ready to make a right turn at the upcoming corner.`
3.  **Prioritize Safety:** Always call out immediate dangers first.
    *   `Stop, car approaching from left.`
    *   `Caution, stairs going down.`
4.  **Be Consistent:** Use a consistent vocabulary.
    *   Always use "obstacle" for general blockages.
    *   Always use "curb" for curbs, not "ledge" or "edge."

#### 3. Data Structuring

Organize your data logically. Create a main `dataset` folder with the following structure:

```
dataset/
├── frames/
│   ├── video1_chunk000.jpg
│   ├── video1_chunk001.jpg
│   └── ...
├── audio/
│   ├── video1_chunk000.wav
│   ├── video1_chunk001.wav
│   └── ...
└── metadata.jsonl
```

The `metadata.jsonl` file is a text file where each line is a JSON object linking the media to its instruction.

**`metadata.jsonl` line format:**
```json
{"frame_path": "frames/video1_chunk000.jpg", "audio_path": "audio/video1_chunk000.wav", "instruction": "Continue straight, path is clear."}
{"frame_path": "frames/video1_chunk001.jpg", "audio_path": "audio/video1_chunk001.wav", "instruction": "Stop, obstacle ahead."}
```

### Phase 2: Creating the Dataset (Scripting)

You can adapt your working.py script to help you create this dataset. The idea is to have the script process a recorded video, show you the frame for each chunk, and prompt you to enter the instruction.

Here is a conceptual script for this task. You can save this as `create_dataset.py`.

````python
import os
import pathlib
import tempfile
import av
from pydub import AudioSegment
from PIL import Image
import json

# --- Configuration ---
CHUNK_DURATION_S = 2.0
DATASET_DIR = pathlib.Path("dataset")
FRAMES_DIR = DATASET_DIR / "frames"
AUDIO_DIR = DATASET_DIR / "audio"
METADATA_FILE = DATASET_DIR / "metadata.jsonl"

def setup_directories():
    """Create the necessary directories for the dataset."""
    DATASET_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)

def process_video_for_labeling(video_path: str):
    """
    Extracts frames and audio chunks from a video and prompts the user for
    instruction labels.
    """
    print(f"Processing video: {video_path}")
    video_name = pathlib.Path(video_path).stem
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    time_base = video_stream.time_base
    duration = float(video_stream.duration * time_base)
    
    # Use pydub to handle audio extraction
    full_audio = AudioSegment.from_file(video_path)

    current_time = 0.0
    chunk_id = 0
    
    with open(METADATA_FILE, "a") as f:
        while current_time < duration:
            target_time = current_time + (CHUNK_DURATION_S / 2)
            
            # --- Extract Frame ---
            frame_filename = f"{video_name}_chunk{chunk_id:04d}.jpg"
            frame_path = FRAMES_DIR / frame_filename
            
            container.seek(int(target_time / time_base))
            frame_extracted = False
            for frame in container.decode(video=0):
                timestamp = float(frame.pts * time_base)
                if abs(timestamp - target_time) < CHUNK_DURATION_S:
                    frame.to_image().save(frame_path)
                    frame_extracted = True
                    break
            
            if not frame_extracted:
                current_time += CHUNK_DURATION_S
                chunk_id += 1
                continue

            # --- Extract Audio ---
            audio_filename = f"{video_name}_chunk{chunk_id:04d}.wav"
            audio_path = AUDIO_DIR / audio_filename
            start_ms = int(current_time * 1000)
            end_ms = int((current_time + CHUNK_DURATION_S) * 1000)
            audio_chunk = full_audio[start_ms:end_ms]
            audio_chunk.export(audio_path, format="wav")

            # --- User Annotation ---
            print("\n" + "="*50)
            print(f"Chunk {chunk_id} at {current_time:.1f}s")
            # On a local machine, you can open the image automatically
            # import webbrowser
            # webbrowser.open(str(frame_path.resolve()))
            print(f"Showing frame: {frame_path}")
            
            instruction = input("Enter instruction (or 'skip'): ")
            
            if instruction.lower() == 'skip':
                print("...Skipped.")
                os.remove(frame_path)
                os.remove(audio_path)
            else:
                # Write metadata to the JSONL file
                metadata = {
                    "frame_path": str(frame_path),
                    "audio_path": str(audio_path),
                    "instruction": instruction.strip()
                }
                f.write(json.dumps(metadata) + "\n")
                print(f"Saved: {instruction}")

            current_time += CHUNK_DURATION_S
            chunk_id += 1
            
    container.close()
    print("\nFinished processing video.")

if __name__ == "__main__":
    setup_directories()
    # Replace with the path to your recorded video
    video_file_to_process = input("Enter the path to the video file you want to label: ")
    if os.path.exists(video_file_to_process):
        process_video_for_labeling(video_file_to_process)
    else:
        print("File not found.")

```

**How to use this script:**
1.  Save it as `create_dataset.py`.
2.  Run it from your terminal: `python create_dataset.py`.
3.  It will ask for the path to a video you recorded.
4.  For each 2-second chunk, it will save a frame and audio clip, print the path to the frame (which you can open to view), and wait for you to type the instruction.
5.  Your labels are saved to `dataset/metadata.jsonl`. Repeat for all your videos.

### Phase 3: Fine-Tuning (High-Level Plan)

Once you have a dataset of at least a few hundred high-quality examples, you can begin fine-tuning.

1.  **Use Hugging Face `transformers`:** The `Trainer` API is perfect for this.
2.  **Load Your Dataset:** Use the Hugging Face `datasets` library to load your `metadata.jsonl` file.
3.  **Format the Data:** You'll need a function that takes a data entry (`frame_path`, `audio_path`, `instruction`) and formats it into the chat template the model expects. The instruction you wrote will be the "assistant's" response.
4.  **Training:**
    *   Use a powerful GPU (like a T4 or A100 on Google Colab).
    *   Employ techniques like **LoRA (Low-Rank Adaptation)** to make fine-tuning much more memory-efficient.
    *   Train for a few epochs, validating regularly to prevent overfitting.
5.  **Save & Test:** Save your fine-tuned model adapter (the LoRA weights) and load them into your `working.py` script for inference.

This structured approach will give you the best chance of creating a highly accurate, custom navigation assistant. Start with a small, high-quality dataset and scale up.# filepath: create_dataset.py
import os
import pathlib
import tempfile
import av
from pydub import AudioSegment
from PIL import Image
import json

# --- Configuration ---
CHUNK_DURATION_S = 2.0
DATASET_DIR = pathlib.Path("dataset")
FRAMES_DIR = DATASET_DIR / "frames"
AUDIO_DIR = DATASET_DIR / "audio"
METADATA_FILE = DATASET_DIR / "metadata.jsonl"

def setup_directories():
    """Create the necessary directories for the dataset."""
    DATASET_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)

def process_video_for_labeling(video_path: str):
    """
    Extracts frames and audio chunks from a video and prompts the user for
    instruction labels.
    """
    print(f"Processing video: {video_path}")
    video_name = pathlib.Path(video_path).stem
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    time_base = video_stream.time_base
    duration = float(video_stream.duration * time_base)
    
    # Use pydub to handle audio extraction
    full_audio = AudioSegment.from_file(video_path)

    current_time = 0.0
    chunk_id = 0
    
    with open(METADATA_FILE, "a") as f:
        while current_time < duration:
            target_time = current_time + (CHUNK_DURATION_S / 2)
            
            # --- Extract Frame ---
            frame_filename = f"{video_name}_chunk{chunk_id:04d}.jpg"
            frame_path = FRAMES_DIR / frame_filename
            
            container.seek(int(target_time / time_base))
            frame_extracted = False
            for frame in container.decode(video=0):
                timestamp = float(frame.pts * time_base)
                if abs(timestamp - target_time) < CHUNK_DURATION_S:
                    frame.to_image().save(frame_path)
                    frame_extracted = True
                    break
            
            if not frame_extracted:
                current_time += CHUNK_DURATION_S
                chunk_id += 1
                continue

            # --- Extract Audio ---
            audio_filename = f"{video_name}_chunk{chunk_id:04d}.wav"
            audio_path = AUDIO_DIR / audio_filename
            start_ms = int(current_time * 1000)
            end_ms = int((current_time + CHUNK_DURATION_S) * 1000)
            audio_chunk = full_audio[start_ms:end_ms]
            audio_chunk.export(audio_path, format="wav")

            # --- User Annotation ---
            print("\n" + "="*50)
            print(f"Chunk {chunk_id} at {current_time:.1f}s")
            # On a local machine, you can open the image automatically
            # import webbrowser
            # webbrowser.open(str(frame_path.resolve()))
            print(f"Showing frame: {frame_path}")
            
            instruction = input("Enter instruction (or 'skip'): ")
            
            if instruction.lower() == 'skip':
                print("...Skipped.")
                os.remove(frame_path)
                os.remove(audio_path)
            else:
                # Write metadata to the JSONL file
                metadata = {
                    "frame_path": str(frame_path),
                    "audio_path": str(audio_path),
                    "instruction": instruction.strip()
                }
                f.write(json.dumps(metadata) + "\n")
                print(f"Saved: {instruction}")

            current_time += CHUNK_DURATION_S
            chunk_id += 1
            
    container.close()
    print("\nFinished processing video.")

if __name__ == "__main__":
    setup_directories()
    # Replace with the path to your recorded video
    video_file_to_process = input("Enter the path to the video file you want to label: ")
    if os.path.exists(video_file_to_process):
        process_video_for_labeling(video_file_to_process)
    else:
        print("File not found.")

```

**How to use this script:**
1.  Save it as `create_dataset.py`.
2.  Run it from your terminal: `python create_dataset.py`.
3.  It will ask for the path to a video you recorded.
4.  For each 2-second chunk, it will save a frame and audio clip, print the path to the frame (which you can open to view), and wait for you to type the instruction.
5.  Your labels are saved to `dataset/metadata.jsonl`. Repeat for all your videos.

### Phase 3: Fine-Tuning (High-Level Plan)

Once you have a dataset of at least a few hundred high-quality examples, you can begin fine-tuning.

1.  **Use Hugging Face `transformers`:** The `Trainer` API is perfect for this.
2.  **Load Your Dataset:** Use the Hugging Face `datasets` library to load your `metadata.jsonl` file.
3.  **Format the Data:** You'll need a function that takes a data entry (`frame_path`, `audio_path`, `instruction`) and formats it into the chat template the model expects. The instruction you wrote will be the "assistant's" response.
4.  **Training:**
    *   Use a powerful GPU (like a T4 or A100 on Google Colab).
    *   Employ techniques like **LoRA (Low-Rank Adaptation)** to make fine-tuning much more memory-efficient.
    *   Train for a few epochs, validating regularly to prevent overfitting.
5.  **Save & Test:** Save your fine-tuned model adapter (the LoRA weights) and load them into your `working.py` script for inference.

This structured approach will give you the best chance of creating a highly accurate, custom navigation assistant. Start with a small, high-quality dataset and scale up.