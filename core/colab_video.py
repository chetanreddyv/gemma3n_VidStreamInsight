import os
import pathlib
import tempfile
from collections.abc import Iterator
from threading import Thread

import av
import gradio as gr
import torch
from gradio.utils import get_upload_folder
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation.streamers import TextIteratorStreamer

model_id = "google/gemma-3n-E4B-it"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)


VIDEO_FILE_TYPES = (".mp4", ".mov", ".webm")
IMAGE_FILE_TYPES = (".jpg", ".jpeg", ".png", ".webp")


GRADIO_TEMP_DIR = get_upload_folder()

TARGET_FPS = int(os.getenv("TARGET_FPS", "2"))
BATCH_DURATION_SECS = int(os.getenv("BATCH_DURATION_SECS", "4"))
MAX_FRAMES_PER_BATCH = TARGET_FPS * BATCH_DURATION_SECS
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "10_000"))


def get_file_type(path: str) -> str:
    if path.endswith(VIDEO_FILE_TYPES):
        return "video"
    elif path.endswith(IMAGE_FILE_TYPES):
        return "image"
    return "unsupported"


def count_files_in_new_message(paths: list[str]) -> tuple[int, int]:
    video_count = 0
    non_video_count = 0
    for path in paths:
        if path.endswith(VIDEO_FILE_TYPES):
            video_count += 1
        else:
            non_video_count += 1
    return video_count, non_video_count


def validate_media_constraints(message: dict) -> bool:
    video_count, non_video_count = count_files_in_new_message(message["files"])
    if video_count > 1:
        gr.Warning("Only one video is supported.")
        return False
    if video_count == 1 and non_video_count > 0:
        gr.Warning("Mixing images and videos is not allowed.")
        return False
    return True


def extract_frames_in_batches(
    video_path: str,
    target_fps: float,
    batch_duration_secs: int,
    parent_dir: str | None = None,
) -> Iterator[list[str]]:
    """Processes a video and yields batches of frame paths."""
    max_frames_per_batch = int(target_fps * batch_duration_secs)
    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        time_base = video_stream.time_base

        if video_stream.duration is None or time_base is None:
            raise ValueError("Video stream is missing duration or time_base")

        total_duration = float(video_stream.duration * time_base)
        batch_start_time = 0.0

        while batch_start_time < total_duration:
            batch_end_time = batch_start_time + batch_duration_secs
            temp_dir = tempfile.mkdtemp(prefix="frames_", dir=parent_dir)
            frame_paths = []

            # Seek to the start of the current batch
            container.seek(int(batch_start_time / time_base), backward=True, stream=video_stream)

            for frame in container.decode(video=0):
                timestamp = float(frame.pts * time_base)
                if timestamp >= batch_end_time:
                    break
                if timestamp >= batch_start_time:
                    frame_path = pathlib.Path(temp_dir) / f"frame_{len(frame_paths):04d}.jpg"
                    frame.to_image().save(frame_path)
                    frame_paths.append(frame_path.as_posix())
                    if len(frame_paths) >= max_frames_per_batch:
                        break
            
            if frame_paths:
                yield frame_paths

            # Clean up the temporary directory for this batch
            for path in frame_paths:
                os.remove(path)
            os.rmdir(temp_dir)

            batch_start_time = batch_end_time

    finally:
        if 'container' in locals() and container.is_open:
            container.close()


def extract_frames_to_tempdir(
    video_path: str,
    target_fps: float,
    max_frames: int | None = None,
    parent_dir: str | None = None,
    prefix: str = "frames_",
) -> str:
    temp_dir = tempfile.mkdtemp(prefix=prefix, dir=parent_dir)

    container = av.open(video_path)
    video_stream = container.streams.video[0]

    if video_stream.duration is None or video_stream.time_base is None:
        raise ValueError("video_stream is missing duration or time_base")

    time_base = video_stream.time_base
    duration = float(video_stream.duration * time_base)
    interval = 1.0 / target_fps

    total_frames = int(duration * target_fps)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    target_times = [i * interval for i in range(total_frames)]
    target_index = 0

    for frame in container.decode(video=0):
        if frame.pts is None:
            continue

        timestamp = float(frame.pts * time_base)

        if target_index < len(target_times) and abs(timestamp - target_times[target_index]) < (interval / 2):
            frame_path = pathlib.Path(temp_dir) / f"frame_{target_index:04d}.jpg"
            frame.to_image().save(frame_path)
            target_index += 1

            if max_frames is not None and target_index >= max_frames:
                break

    container.close()
    return temp_dir


def process_new_user_message(message: dict) -> list[dict]:
    if not message["files"]:
        return [{"type": "text", "text": message["text"]}]

    file_types = [get_file_type(path) for path in message["files"]]

    if len(file_types) == 1 and file_types[0] == "video":
        gr.Info(f"Video will be processed at {TARGET_FPS} FPS, max {MAX_FRAMES} frames in this Space.")

        temp_dir = extract_frames_to_tempdir(
            message["files"][0],
            target_fps=TARGET_FPS,
            max_frames=MAX_FRAMES,
            parent_dir=GRADIO_TEMP_DIR,
        )
        paths = sorted(pathlib.Path(temp_dir).glob("*.jpg"))
        return [
            {"type": "text", "text": message["text"]},
            *[{"type": "image", "image": path.as_posix()} for path in paths],
        ]

    return [
        {"type": "text", "text": message["text"]},
        *[{"type": file_type, file_type: path} for path, file_type in zip(message["files"], file_types, strict=True)],
    ]


def process_history(history: list[dict]) -> list[dict]:
    messages = []
    current_user_content: list[dict] = []
    for item in history:
        if item["role"] == "assistant":
            if current_user_content:
                messages.append({"role": "user", "content": current_user_content})
                current_user_content = []
            messages.append({"role": "assistant", "content": [{"type": "text", "text": item["content"]}]})
        else:
            content = item["content"]
            if isinstance(content, str):
                current_user_content.append({"type": "text", "text": content})
            else:
                filepath = content[0]
                file_type = get_file_type(filepath)
                current_user_content.append({"type": file_type, file_type: filepath})
    return messages


@torch.inference_mode()
def generate(message: dict, history: list[dict], system_prompt: str = "", max_new_tokens: int = 512) -> Iterator[str]:
    if not validate_media_constraints(message):
        yield ""
        return

    # Check if a video is part of the message
    is_video_message = any(get_file_type(path) == "video" for path in message["files"])

    if not is_video_message:
        # Standard processing for text and images
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.extend(process_history(history))
        messages.append({"role": "user", "content": process_new_user_message(message)})
        
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(device=model.device, dtype=torch.bfloat16)
        
        streamer = TextIteratorStreamer(processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        output = ""
        for delta in streamer:
            output += delta
            yield output
        return

    # --- Batch processing logic for video ---
    video_path = message["files"][0]
    user_text = message["text"]
    full_output = ""
    batch_num = 1

    gr.Info(f"Starting video batch processing: {BATCH_DURATION_SECS}s chunks at {TARGET_FPS} FPS.")

    frame_batches = extract_frames_in_batches(
        video_path,
        target_fps=TARGET_FPS,
        batch_duration_secs=BATCH_DURATION_SECS,
        parent_dir=GRADIO_TEMP_DIR,
    )

    for frame_paths in frame_batches:
        batch_start_time = (batch_num - 1) * BATCH_DURATION_SECS
        batch_end_time = batch_num * BATCH_DURATION_SECS
        
        # Construct message for the current batch
        batch_prompt = f"{user_text}\n\nThis is for the video segment from {batch_start_time}s to {batch_end_time}s."
        user_content = [{"type": "text", "text": batch_prompt}]
        user_content.extend([{"type": "image", "image": path} for path in frame_paths])
        
        messages = [{"role": "user", "content": user_content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(device=model.device, dtype=torch.bfloat16)

        streamer = TextIteratorStreamer(processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        batch_header = f"--- Analysis for Batch {batch_num} ({batch_start_time}s - {batch_end_time}s) ---\n"
        full_output += batch_header
        yield full_output
        
        batch_output = ""
        for delta in streamer:
            batch_output += delta
            yield full_output + batch_output
        
        full_output += batch_output + "\n\n"
        batch_num += 1

    if batch_num == 1:
        yield "Could not extract any frames from the video."


examples = [
    [
        {
            "text": "What is the capital of France?",
            "files": [],
        }
    ],
    [
        {
            "text": "Describe this image in detail.",
            "files": ["assets/cat.jpeg"],
        }
    ],
    [
        {
            "text": "Transcribe the following speech segment in English.",
            "files": ["assets/speech.wav"],
        }
    ],
    [
        {
            "text": "Transcribe the following speech segment in English.",
            "files": ["assets/speech2.wav"],
        }
    ],
    [
        {
            "text": "Describe this video",
            "files": ["assets/holding_phone.mp4"],
        }
    ],
]

demo = gr.ChatInterface(
    fn=generate,
    type="messages",
    textbox=gr.MultimodalTextbox(
        file_types=list(IMAGE_FILE_TYPES + VIDEO_FILE_TYPES),
        file_count="multiple",
        autofocus=True,
    ),
    multimodal=True,
    additional_inputs=[
        gr.Textbox(label="System Prompt", value="You are a helpful assistant."),
        gr.Slider(label="Max New Tokens", minimum=100, maximum=2000, step=10, value=700),
    ],
    stop_btn=False,
    title="Gemma 3n E4B it",
    examples=examples,
    run_examples_on_click=False,
    cache_examples=False,
    css_paths="style.css",
    delete_cache=(1800, 1800),
)

if __name__ == "__main__":
    demo.launch()