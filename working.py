import os
import pathlib
import tempfile
import torch
import av
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from pydub import AudioSegment
import gradio as gr

MODEL_PATH = "google/gemma-3n-E2B-it"
# Import the Colab userdata module to access secrets
from google.colab import userdata

# Load the Hugging Face token from Colab secrets
HF_TOKEN = userdata.get('HF_TOKEN')
# Set the HF_TOKEN environment variable
os.environ['HF_TOKEN'] = HF_TOKEN

processor = AutoProcessor.from_pretrained(MODEL_PATH, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
).eval().to("cuda")

# System configuration
TARGET_FPS = 3
MAX_FRAMES = 30
MIN_AUDIO_DURATION_S = 1.0  # Add minimum audio duration in seconds

def extract_frames_from_video(video_path, target_fps=TARGET_FPS, max_frames=MAX_FRAMES):
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
            frame_img = frame.to_image()
            frame_img.save(frame_path)
            frame_paths.append((str(frame_path), target_index * interval))
            target_index += 1
            
            if target_index >= max_frames:
                break
                
    container.close()
    return frame_paths, duration

def extract_audio_segment(audio_path, start_time, duration):
    """Extract a segment of audio from the given file."""
    audio = AudioSegment.from_file(audio_path)
    segment = audio[start_time*1000:(start_time + duration)*1000]
    
    # Save the segment to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    segment.export(temp_file.name, format="wav")
    return temp_file.name

def process_inputs(image, audio):
    """Process a single frame and audio segment"""
    # The system prompt instructs the model on how to behave.
    system_prompt = (
        "You are an expert AI assistant for a visually impaired person, acting as their eyes. "
        "Based on the image and audio from their surroundings, provide clear, concise, and real-time "
        "descriptions for safe navigation. Focus on the path ahead, obstacles (e.g., curbs, stairs, people), "
        "potential hazards (e.g., moving vehicles), and key landmarks (e.g., doors, crosswalks). "
        "Give direct, simple instructions like 'Walk straight ahead,' 'Caution: step down for the curb,' or "
        "'Stop, a car is approaching from your left.' Your tone should be calm and reassuring."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image", "image": image},
                {"type": "audio", "audio": audio},
            ]
        }
    ]

    input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_len = input_ids["input_ids"].shape[-1]

    input_ids = input_ids.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=256,
            disable_compile=True
        )
    text = processor.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return text[0]

def process_video(video_path, audio_path):
    """Process video by extracting frames and analyzing each with audio"""
    try:
        # Extract frames using the av library
        frame_paths, duration = extract_frames_from_video(video_path)
        
        if not frame_paths:
            yield "Error: No frames extracted from video."
            return
        
        all_results = []
        
        # Process each frame with corresponding audio segment
        for frame_path, timestamp in frame_paths:
            # Calculate segment duration, ensuring it's not too short
            inter_frame_duration = duration / len(frame_paths)
            segment_duration = max(inter_frame_duration, MIN_AUDIO_DURATION_S)
            
            # Extract audio for this segment
            audio_segment = extract_audio_segment(audio_path, timestamp, segment_duration)
            
            # Load image as PIL Image
            image = Image.open(frame_path)
            
            # Process the frame with audio
            result = process_inputs(image, audio_segment)
            
            # Format result with timestamp
            time_str = f"[{timestamp:.1f}s - {timestamp + segment_duration:.1f}s]"
            current_result = f"{time_str}: {result}"
            all_results.append(current_result)
            
            # Clean up
            os.unlink(audio_segment)

            yield "\n\n".join(all_results)
            
        # Clean up frame files
        for path, _ in frame_paths:
            if os.path.exists(path):
                os.unlink(path)
                
    except Exception as e:
        yield f"Error processing video: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Audio(label="Upload Audio", type="filepath")
    ],
    outputs=gr.Textbox(label="Analysis Results"),
    title="Video Stream Analysis with Audio",
    description="Upload a video file and its audio. The system processes the video frames and analyzes them with the Gemma 3n model."
)

if __name__ == "__main__":
    iface.launch()