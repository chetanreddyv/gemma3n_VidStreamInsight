import cv2
import numpy as np
import tempfile
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import gradio as gr
from pydub import AudioSegment
import time

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


def process_inputs(image, audio):
    messages = [
        {
        "role": "user",
        "content": [
            {"type": "image", "image": image,},
            {"type": "audio", "audio": audio,},
        ]
    },]

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


def extract_audio_segment(audio_path, start_time, duration):
    """Extract a segment of audio from the given file."""
    audio = AudioSegment.from_file(audio_path)
    segment = audio[start_time*1000:(start_time + duration)*1000]
    
    # Save the segment to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    segment.export(temp_file.name, format="wav")
    return temp_file.name


def process_video_stream(video_path, audio_path, status_box):
    """Process video in 2-second batches at 2 FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video file."
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Settings for processing
    target_fps = 2  # Process at 2 FPS
    batch_duration = 2.0  # seconds per batch
    frame_interval = int(fps / target_fps)
    frames_per_batch = int(batch_duration * target_fps)
    
    all_results = []
    current_time = 0.0
    
    while current_time < duration:
        batch_frames = []
        
        # Seek to the current position
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        
        # Collect frames for this batch
        for i in range(frames_per_batch):
            # Skip frames to achieve 2 FPS
            for _ in range(frame_interval):
                ret, frame = cap.read()
                if not ret:
                    break
            
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame_rgb)
        
        # If we have frames in this batch
        if batch_frames:
            # Extract audio for this segment
            audio_segment = extract_audio_segment(audio_path, current_time, batch_duration)
            
            # Use middle frame as representative frame
            middle_frame = batch_frames[len(batch_frames)//2] if len(batch_frames) > 0 else batch_frames[0]
            
            # Process the batch
            result = process_inputs(middle_frame, audio_segment)
            all_results.append(f"[{current_time:.1f}s - {current_time + batch_duration:.1f}s]: {result}")
            
            # Update status
            status_box.value = "\n\n".join(all_results)
            
            # Clean up
            os.unlink(audio_segment)
            
        current_time += batch_duration
    
    cap.release()
    return "\n\n".join(all_results)


def video_processor(video_file, audio_file):
    """Wrapper for gradio to handle the processing and update the UI."""
    status_box = gr.Textbox(label="Processing Status", interactive=False)
    return process_video_stream(video_file, audio_file, status_box), status_box


# Gradio interface
iface = gr.Interface(
    fn=video_processor,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Audio(label="Upload Audio", type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Final Results"),
        gr.Textbox(label="Processing Status")
    ],
    title="Video Stream Analysis with Audio",
    description="Upload a video file and its audio. The system processes the video in 2-second batches at 2 FPS, analyzing each batch with the Gemma 3n model."
)

if __name__ == "__main__":
    iface.launch()