import time
import pathlib
import shutil

# Import modular components
import config
from model_handler import load_model
from audio_handler import AudioHandler
from video_handler import capture_frames
from llm_handler import format_prompt, generate_response

def cleanup(temp_dir):
    """Safely removes the temporary directory."""
    if temp_dir and pathlib.Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory: {temp_dir}")

def run_passive_mode(model, processor, audio_handler):
    """Captures frames, generates a description, and speaks it."""
    temp_dir, frames = capture_frames(config.CAPTURE_DURATION_SECONDS, config.TARGET_FPS)
    
    if not frames:
        audio_handler.speak("Could not capture video from the camera. Please check the connection.")
        time.sleep(5)
        return None, [] # Return None for temp_dir

    passive_prompt = format_prompt(
        text_prompt=config.SYSTEM_PROMPT,
        frames=frames[-config.MAX_FRAMES_IN_PROMPT:]
    )
    
    for sentence in generate_response(model, processor, passive_prompt):
        audio_handler.speak(sentence)
        
    return temp_dir, frames

def run_active_mode(model, processor, audio_handler, temp_dir, frames):
    """Listens for a command and generates a response based on audio and video."""
    audio_handler.speak("Yes? How can I help?")
    audio_data = audio_handler.listen_for_command()
    
    if audio_data:
        audio_path = pathlib.Path(temp_dir) / "command.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_data)
            
        active_prompt = format_prompt(
            text_prompt=f"{config.SYSTEM_PROMPT}\n\n{config.USER_PROMPT_WITH_AUDIO}",
            frames=frames[-config.MAX_FRAMES_IN_PROMPT:],
            audio_path=audio_path
        )
        for sentence in generate_response(model, processor, active_prompt):
            audio_handler.speak(sentence)

def main():
    """The main operational loop for the assistant."""
    model, processor = load_model()
    audio_handler = AudioHandler()
    
    audio_handler.speak("VidStream Insight is now active. I will describe your surroundings.")
    
    temp_dir_path = None
    try:
        while True:
            # --- Passive Mode ---
            print("\n--- Entering Passive Mode ---")
            temp_dir_path, frames = run_passive_mode(model, processor, audio_handler)
            
            if not temp_dir_path:
                continue

            # --- Listen for Wake Word ---
            if audio_handler.listen_for_wake_word(config.WAKE_WORD):
                run_active_mode(model, processor, audio_handler, temp_dir_path, frames)

            # --- Cleanup for the current loop ---
            cleanup(temp_dir_path)
            temp_dir_path = None

    except KeyboardInterrupt:
        print("\nShutting down.")
        audio_handler.speak("Goodbye.")
    except Exception as e:
        print(f"An unexpected error occurred in main loop: {e}")
        audio_handler.speak("An error occurred. Restarting the loop.")
        time.sleep(2)
    finally:
        cleanup(temp_dir_path)

if __name__ == "__main__":
    main()