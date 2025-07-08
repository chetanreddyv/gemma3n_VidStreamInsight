import os
import pathlib
import tempfile
import time
import av
import torch
from threading import Thread
from collections import deque
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

# --- Model Initialization ---
print("Initializing model...")
model_id = "google/gemma-3n-E4B-it"

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get('HF_TOKEN')
        os.environ["HF_TOKEN"] = HF_TOKEN
        print("Hugging Face token loaded from Colab secrets.")
    except Exception:
        print("⚠️ Warning: Hugging Face token not found. Make sure you are logged in.")

processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print("✅ Model initialized.")

# --- System Configuration ---
TARGET_FPS = 3
MAX_FRAMES_IN_PROMPT = 10
MAX_INPUT_TOKENS = 8000
SYSTEM_PROMPT = (
    "You are an expert assistant for visually impaired users. "
    "Your role is to describe the user's surroundings based on a sequence of video frames. "
    "Be concise and clear. Focus on immediate obstacles, people, and important environmental features like doors, stairs, or crosswalks. "
    "Describe the scene as if you are right there with the user."
)
USER_PROMPT_WITH_AUDIO = (
    "The user has provided a spoken question. Analyze the video frames in the context of this audio question and provide a direct answer."
)

# --- Text-to-speech placeholder ---
def speak(text):
    """Simulated speech output (just print)."""
    print(f"\n🔈 ASSISTANT: {text}")

# --- Video Processing ---
def extract_frames_from_video(video_path, fps, max_frames=30):
    """Extract frames from video at target FPS using PyAV for accuracy."""
    temp_dir = tempfile.mkdtemp(prefix="video_frames_")
    frame_paths = []
    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        time_base = video_stream.time_base
        
        # Calculate the interval between frames to capture
        interval = 1.0 / fps
        target_times = [i * interval for i in range(max_frames)]
        target_index = 0

        for frame in container.decode(video=0):
            timestamp = float(frame.pts * time_base)
            
            if target_index < len(target_times) and timestamp >= target_times[target_index]:
                path = pathlib.Path(temp_dir) / f"frame_{target_index:04d}.jpg"
                frame.to_image().save(path)
                frame_paths.append(str(path))
                target_index += 1
                
                if len(frame_paths) >= max_frames:
                    break
        
        container.close()
        print(f"✅ Extracted {len(frame_paths)} frames to {temp_dir}")
    except av.AVError as e:
        print(f"Error processing video with PyAV: {e}")
        return []
    return frame_paths

# --- Prompt Builder ---
def format_prompt(text_prompt, frames, audio_path=None):
    """This function is no longer needed as arguments are passed directly."""
    # This function can be removed or left for conceptual clarity.
    # The new generate_response function takes text, frames, and audio directly.
    pass

# --- Generation (Streaming) ---
@torch.inference_mode()
def generate_response(text_prompt, frames, audio_path=None):
    """Generates a response from the model in a streaming fashion."""
    images = [Image.open(f) for f in frames]

    # The model expects an <image> token for each image.
    # Prepend the required number of tokens to the text prompt.
    prompt_with_images = "<image>" * len(images) + text_prompt

    # The processor expects named arguments for text, images, and audio.
    inputs = processor(
        text=prompt_with_images,
        images=images,
        audio=audio_path,
        return_tensors="pt",
    ).to(model.device)

    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        yield "⚠️ The input is too long. Please shorten the context."
        return

    streamer = TextIteratorStreamer(
        processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=False,
    )

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    buffer = ""
    print("\n🟢 ASSISTANT (streaming): ", end="", flush=True)
    for delta in streamer:
        buffer += delta
        print(delta, end="", flush=True)
        if any(p in buffer for p in ".!?"):
            # Split by sentence-ending punctuation to yield complete sentences
            import re
            sentences = re.split(r'(?<=[.!?])\s*', buffer)
            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                if sentence:
                    yield sentence
            buffer = sentences[-1]

    if buffer.strip():
        yield buffer.strip()
    print()

# --- Main Prototyping Function ---
def prototype_run(video_file, audio_file=None):
    print("\n⚡️ Prototype run started.")

    # 1️⃣ Extract frames from video
    frames = extract_frames_from_video(video_file, fps=TARGET_FPS, max_frames=MAX_FRAMES_IN_PROMPT)
    if not frames:
        print("❌ No frames extracted. Exiting.")
        return

    # 2️⃣ Passive Mode (just describing the scene)
    speak("Describing your surroundings now.")
    for sentence in generate_response(text_prompt=SYSTEM_PROMPT, frames=frames):
        speak(sentence)

    # 3️⃣ Active Mode (user's audio question)
    if audio_file:
        active_prompt_text = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_WITH_AUDIO}"
        speak("Answering your question now.")
        for sentence in generate_response(
            text_prompt=active_prompt_text,
            frames=frames,
            audio_path=audio_file
        ):
            speak(sentence)

    print("\n✅ Prototype run complete.")

