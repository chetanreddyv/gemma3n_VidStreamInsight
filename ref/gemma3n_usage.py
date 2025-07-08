import os
import pathlib
import tempfile
import torch
import av
from threading import Thread
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation.streamers import TextIteratorStreamer

# Core model initialization
model_id = "google/gemma-3n-E4B-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

# System configuration
VIDEO_FILE_TYPES = (".mp4", ".mov", ".webm")
IMAGE_FILE_TYPES = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_FILE_TYPES = (".mp3", ".wav")
TARGET_FPS = int(os.getenv("TARGET_FPS", "3"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "30"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "10_000"))

def extract_frames_from_video(
    video_path: str,
    target_fps: float,
    max_frames: int | None = None,
) -> list[str]:
    """Extract frames from video at specified FPS rate"""
    temp_dir = tempfile.mkdtemp(prefix="frames_")
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    
    time_base = video_stream.time_base
    duration = float(video_stream.duration * time_base)
    interval = 1.0 / target_fps
    
    total_frames = int(duration * target_fps)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
        
    target_times = [i * interval for i in range(total_frames)]
    target_index = 0
    frame_paths = []
    
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
            
        timestamp = float(frame.pts * time_base)
        
        if target_index < len(target_times) and abs(timestamp - target_times[target_index]) < (interval / 2):
            frame_path = pathlib.Path(temp_dir) / f"frame_{target_index:04d}.jpg"
            frame.to_image().save(frame_path)
            frame_paths.append(str(frame_path))
            target_index += 1
            
            if max_frames is not None and target_index >= max_frames:
                break
                
    container.close()
    return frame_paths

def process_user_input(input_data):
    """Process user input (text, video, images, audio)"""
    # For video input, extract frames
    if input_data.get("type") == "video":
        frames = extract_frames_from_video(
            input_data["video_path"],
            target_fps=TARGET_FPS,
            max_frames=MAX_FRAMES,
        )
        
        return {
            "type": "multimodal",
            "text": input_data.get("text", ""),
            "frames": frames
        }
    
    # For other media types
    return input_data

def format_conversation_history(history):
    """Format conversation history for the model"""
    messages = []
    for item in history:
        if item["role"] == "assistant":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": item["content"]}]})
        else:
            # Process user messages according to content type
            if item.get("type") == "multimodal":
                content = [{"type": "text", "text": item["text"]}]
                for frame in item.get("frames", []):
                    content.append({"type": "image", "image": frame})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": item["content"]}]})
    
    return messages

@torch.inference_mode()
def generate_response(user_input, history, system_prompt="", max_new_tokens=512):
    """Generate AI response from model"""
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    
    # Add conversation history
    messages.extend(format_conversation_history(history))
    
    # Process and add current user input
    processed_input = process_user_input(user_input)
    
    # Format the user input into appropriate message structure
    user_content = []
    if processed_input.get("type") == "multimodal":
        user_content.append({"type": "text", "text": processed_input["text"]})
        for frame in processed_input.get("frames", []):
            user_content.append({"type": "image", "image": frame})
    else:
        user_content.append({"type": "text", "text": processed_input["content"]})
    
    messages.append({"role": "user", "content": user_content})
    
    # Prepare inputs for the model
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # Check token limit
    n_tokens = inputs["input_ids"].shape[1]
    if n_tokens > MAX_INPUT_TOKENS:
        return "Input too long. Maximum token limit exceeded."
    
    inputs = inputs.to(device=model.device, dtype=torch.bfloat16)
    
    # Set up streamer for real-time output
    streamer = TextIteratorStreamer(processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    
    # Generate in a separate thread
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    # Collect and return output
    output = ""
    for delta in streamer:
        output += delta
        # In a real application, each delta could be sent to TTS immediately
        
    return output

def main_application():
    """Main application loop"""
    # Initialize system
    history = []
    system_prompt = "You are a helpful assistant for visually impaired users. Describe surroundings clearly and concisely."
    
    # In a real implementation, this would be connected to a camera feed
    # and audio input/output system, rather than using a chat interface
    
    while True:
        # In passive mode: continuously process video frames
        video_input = capture_video_frame()  # This would be implemented to connect to a camera
        
        # Process the video input
        user_input = {
            "type": "video",
            "video_path": video_input,
            "text": ""  # No explicit query in passive mode
        }
        
        # Generate description
        response = generate_response(user_input, history, system_prompt)
        
        # In a real implementation: Send to TTS engine
        # text_to_speech(response)
        
        # Check for voice commands (active mode)
        audio_command = listen_for_commands()  # This would be implemented
        if audio_command:
            # Process the specific user query
            user_input = {
                "type": "multimodal",
                "text": audio_command,
                "video_path": video_input
            }
            
            # Generate targeted response
            response = generate_response(user_input, history, system_prompt)
            
            # In a real implementation: Send to TTS engine
            # text_to_speech(response)
            
        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        
        # Limit history length to manage token count
        if len(history) > 10:  # Arbitrary limit
            history = history[-10:]