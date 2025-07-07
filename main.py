import os
import pathlib
import tempfile
import time
import torch
import cv2
import speech_recognition as sr
import pyttsx3
from threading import Thread, Lock
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation.streamers import TextIteratorStreamer

# --- Core Model Initialization ---
print("Initializing model...")
model_id = "google/gemma-3n-E4B-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2" # Optional: for faster inference if supported
)
print("Model initialized.")

# --- System Configuration ---
TARGET_FPS = int(os.getenv("TARGET_FPS", "3"))
MAX_FRAMES_IN_PROMPT = int(os.getenv("MAX_FRAMES", "10")) # Frames to send to the model
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "8000"))
CAPTURE_DURATION_SECONDS = 5 # How many seconds of video to analyze at a time
SYSTEM_PROMPT = (
    "You are an expert assistant for visually impaired users. "
    "Your role is to describe the user's surroundings based on a sequence of video frames. "
    "Be concise and clear. Focus on immediate obstacles, people, and important environmental features like doors, stairs, or crosswalks. "
    "Describe the scene as if you are right there with the user. For example, say 'A person is walking towards you on your left' instead of 'The image shows a person.'"
)

# --- TTS and STT Initialization ---
tts_engine = pyttsx3.init()
recognizer = sr.Recognizer()
microphone = sr.Microphone()
is_speaking = Lock()

def speak(text):
    """Converts text to speech, ensuring sequential output."""
    with is_speaking:
        print(f"ASSISTANT: {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()

# --- Frame and Input Processing ---
def capture_frames_from_camera(duration_seconds, fps):
    """Captures frames from the default camera for a given duration."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return []

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
        
        # Small sleep to prevent busy-waiting
        time.sleep(0.01)

    cap.release()
    print(f"Captured {len(frame_paths)} frames.")
    return frame_paths

def format_prompt(text_prompt, frames):
    """Formats the prompt for the Gemma model."""
    # The model expects the prompt to start with text, followed by images.
    prompt_parts = [text_prompt]
    for frame_path in frames:
        prompt_parts.append(pathlib.Path(frame_path))
    return prompt_parts

# --- AI Response Generation ---
@torch.inference_mode()
def generate_response(prompt):
    """Generates a response from the model in a streaming fashion."""
    inputs = processor(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    # Check token limit
    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        return "The current context is too long. Please start a new conversation."

    streamer = TextIteratorStreamer(
        processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=False,
    )

    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    # Yield generated text chunks
    buffer = ""
    print("ASSISTANT (streaming): ", end="", flush=True)
    for delta in streamer:
        buffer += delta
        print(delta, end="", flush=True)
        # Yield complete sentences to the TTS engine
        if any(p in buffer for p in ".!?"):
            sentences = buffer.split('.')
            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                if sentence:
                    yield f"{sentence}."
            buffer = sentences[-1]
    
    # Yield any remaining text
    if buffer.strip():
        yield buffer.strip()
    print() # Newline after streaming is complete

# --- Main Application Logic ---
def main_loop():
    """The main operational loop for the assistant."""
    speak("VidStream Insight is now active. I will describe your surroundings. Say 'Hey Assistant' to ask a question.")
    
    while True:
        try:
            # --- Passive Mode: Continuous Description ---
            print("\n--- Entering Passive Mode ---")
            frames = capture_frames_from_camera(CAPTURE_DURATION_SECONDS, TARGET_FPS)
            
            if not frames:
                speak("Could not capture video from the camera. Please check the connection.")
                time.sleep(5)
                continue

            # Generate description for the captured frames
            passive_prompt = format_prompt(
                text_prompt=SYSTEM_PROMPT,
                frames=frames[-MAX_FRAMES_IN_PROMPT:] # Use the most recent frames
            )
            
            # Stream response and speak it sentence by sentence
            for sentence in generate_response(passive_prompt):
                speak(sentence)

            # --- Active Mode: Listen for Command ---
            print("\n--- Listening for Wake Word ('Hey Assistant') ---")
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    # Use listen in background for non-blocking wake word detection
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                    text = recognizer.recognize_google(audio)
                    print(f"Heard: {text}")
                    if "assistant" in text.lower():
                        speak("Yes? How can I help?")
                        # Listen for the actual command
                        audio_command = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                        command = recognizer.recognize_google(audio_command)
                        print(f"USER COMMAND: {command}")
                        
                        # Generate response based on command and current frames
                        active_prompt = format_prompt(
                            text_prompt=f"{SYSTEM_PROMPT}\n\nUser question: {command}",
                            frames=frames[-MAX_FRAMES_IN_PROMPT:]
                        )
                        for sentence in generate_response(active_prompt):
                            speak(sentence)
                except sr.UnknownValueError:
                    # This is expected if no speech is detected
                    print("No wake word detected.")
                except sr.RequestError as e:
                    print(f"Could not request results from speech recognition service; {e}")
                except sr.WaitTimeoutError:
                    print("Listening timed out.")

        except KeyboardInterrupt:
            print("\nShutting down.")
            speak("Goodbye.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            speak("An error occurred. Restarting the loop.")
            time.sleep(2)

if __name__ == "__main__":
    main_loop()
