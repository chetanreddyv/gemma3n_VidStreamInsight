import os
import pathlib
import tempfile
import torch
import av
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from pydub import AudioSegment
import gradio as gr
import threading
import time
from queue import Queue  

MODEL_PATH = "google/gemma-3n-E2B-it"
# Import the Colab userdata module to access secrets
from google.colab import userdata
import os
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

# Real-time streaming configuration
CHUNK_DURATION_S = 2.0  # Process 2-second chunks
PROCESSING_DELAY_S = 0.1  # Small delay between chunks for real-time feel
MAX_CONCURRENT_CHUNKS = 3  # Limit concurrent processing

class VideoChunkProcessor:
    def __init__(self, video_path, audio_path):
        self.video_path = video_path
        self.audio_path = audio_path
        self.container = av.open(video_path)
        self.video_stream = self.container.streams.video[0]
        self.time_base = self.video_stream.time_base
        self.duration = float(self.video_stream.duration * self.time_base)
        
        # Threading and Queues for parallel processing
        self.processing_queue = Queue(maxsize=MAX_CONCURRENT_CHUNKS)
        self.result_queue = Queue()
        self.stop_event = threading.Event()
        self.producer_thread = None
        self.consumer_thread = None

    def _producer(self):
        """Extracts chunks and puts them in a queue for processing."""
        current_time = 0.0
        chunk_id = 0
        while current_time < self.duration and not self.stop_event.is_set():
            frame_path = self.extract_chunk_frame(current_time)
            audio_path_chunk = self.extract_chunk_audio(current_time)
            
            if frame_path:
                self.processing_queue.put((chunk_id, current_time, frame_path, audio_path_chunk))
                chunk_id += 1
            
            current_time += CHUNK_DURATION_S
        self.processing_queue.put(None) # Sentinel to stop consumer

    def _consumer(self):
        """Processes chunks from the queue."""
        while not self.stop_event.is_set():
            chunk_data = self.processing_queue.get()
            if chunk_data is None: # End of stream
                break
            
            chunk_id, start_time, frame_path, audio_path_chunk = chunk_data
            result = self.process_chunk(chunk_id, start_time, frame_path, audio_path_chunk)
            self.result_queue.put(result)
        self.result_queue.put(None) # Sentinel to stop result loop

    def start_processing(self):
        """Starts the producer and consumer threads."""
        self.producer_thread = threading.Thread(target=self._producer)
        self.consumer_thread = threading.Thread(target=self._consumer)
        self.producer_thread.start()
        self.consumer_thread.start()

    def stop_processing(self):
        """Stops all threads and cleans up."""
        self.stop_event.set()
        # Clear queues to unblock threads
        while not self.processing_queue.empty():
            self.processing_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()
        if self.producer_thread:
            self.producer_thread.join(timeout=2)
        if self.consumer_thread:
            self.consumer_thread.join(timeout=2)
        self.container.close()
        
    def extract_chunk_frame(self, start_time):
        """Extract single representative frame from chunk"""
        try:
            # Get middle frame of the chunk
            target_time = start_time + (CHUNK_DURATION_S / 2)
            
            if target_time >= self.duration:
                return None
                
            # Create temporary file for frame
            temp_dir = tempfile.mkdtemp(prefix="chunk_")
            frame_path = pathlib.Path(temp_dir) / f"frame_{start_time:.1f}.jpg"
            
            # Seek and extract frame
            self.container.seek(int(target_time / self.time_base))
            for frame in self.container.decode(video=0):
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * self.time_base)
                if abs(timestamp - target_time) < CHUNK_DURATION_S / 2:
                    frame_img = frame.to_image()
                    frame_img.save(frame_path)
                    return str(frame_path)
            return None
        except Exception as e:
            print(f"Error extracting frame: {e}")  # Added error logging
            return None
    
    def extract_chunk_audio(self, start_time):
        """Extract audio chunk"""
        try:
            audio = AudioSegment.from_file(self.audio_path)
            start_ms = max(0, int(start_time * 1000))
            end_ms = min(len(audio), int((start_time + CHUNK_DURATION_S) * 1000))
            segment = audio[start_ms:end_ms]
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            segment.export(temp_file.name, format="wav")
            return temp_file.name
        except Exception as e:
            print(f"Error extracting audio: {e}")  # Added error logging
            # Return silent audio on error
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            empty_audio = AudioSegment.silent(duration=int(CHUNK_DURATION_S * 1000))
            empty_audio.export(temp_file.name, format="wav")
            return temp_file.name
    
    def process_chunk(self, chunk_id, start_time, frame_path, audio_path):
        """Process single chunk independently"""
        try:
            # Load media
            image = Image.open(frame_path)
            
            # Generate navigation instruction
            instruction = self.generate_instruction(image, audio_path)
            
            # Format result with timestamp
            end_time = min(start_time + CHUNK_DURATION_S, self.duration)
            time_str = f"[{start_time:.1f}s-{end_time:.1f}s]"
            result = f"{time_str}: {instruction}"
            
            # Clean up temporary files
            if os.path.exists(frame_path):
                os.unlink(frame_path)
                # Remove temp directory if empty
                temp_dir = os.path.dirname(frame_path)
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            return (chunk_id, start_time, result)
            
        except Exception as e:
            print(f"Error processing chunk: {e}")  # Added error logging
            return (chunk_id, start_time, f"[{start_time:.1f}s]: Error processing chunk")
    
    def generate_instruction(self, image, audio_path):
        """Generate navigation instruction for chunk"""
        system_prompt = (
            "You are a real-time navigation assistant for a blind person. "
            "Give ONE immediate, concise instruction (5-10 words max) based on the scene. "
            "Focus on movement, obstacles, and safety. Be direct."
        )
        
        content = [
            {"type": "text", "text": system_prompt},
            {"type": "image", "image": image},
        ]
        if audio_path:
            content.append({"type": "audio", "audio": audio_path})

        messages = [{"role": "user", "content": content}]

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
                max_new_tokens=15,
                min_new_tokens=2,
                do_sample=True,
                temperature=0.6,
                repetition_penalty=1.1,
                disable_compile=True
            )
        
        text = processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        result = text[0].strip()
        
        return result if result else "Continue straight"

def process_video_stream(video_path, audio_path):
    """Process video as live stream with real-time chunks using parallel threads."""
    processor_obj = None
    try:
        processor_obj = VideoChunkProcessor(video_path, audio_path)
        processor_obj.start_processing()
        
        recent_instructions = []
        total_chunks = int(processor_obj.duration / CHUNK_DURATION_S)
        processed_chunks = 0

        while True:
            result = processor_obj.result_queue.get()
            if result is None: # End of results
                break

            processed_chunks += 1
            _, timestamp, instruction = result
            recent_instructions.append(instruction)
            
            if len(recent_instructions) > 5:
                recent_instructions.pop(0)
            
            progress = (processed_chunks / total_chunks) * 100 if total_chunks > 0 else 100
            current_time_display = (processed_chunks * CHUNK_DURATION_S)
            
            display_text = "\n\n".join(recent_instructions)
            display_text += f"\n\n🔴 LIVE | Progress: {progress:.1f}% | Time: {current_time_display:.1f}s/{processor_obj.duration:.1f}s"
            
            yield display_text
            time.sleep(PROCESSING_DELAY_S) # Prevent overwhelming the UI

        final_text = "\n\n".join(recent_instructions)
        final_text += "\n\n✅ **Live Stream Complete**"
        yield final_text
        
    except Exception as e:
        yield f"Error processing video: {str(e)}"
    finally:
        if processor_obj:
            processor_obj.stop_processing()

# Gradio interface for live streaming
iface = gr.Interface(
    fn=process_video_stream,
    inputs=[
        gr.Video(label="Upload Video for Live Processing"),
        gr.Audio(label="Upload Corresponding Audio", type="filepath")
    ],
    outputs=gr.Textbox(
        label="🔴 LIVE Navigation Instructions", 
        info="Real-time streaming navigation guidance",
        max_lines=10
    ),
    title="🔴 LIVE AI Navigation Stream",
    description="""
    **Real-time live streaming navigation assistant**
    
    🔴 **LIVE PROCESSING**: Each 2-second chunk processed independently
    ⚡ **Instant Results**: Immediate instructions as chunks complete
    🔄 **Continuous Stream**: Simulates real-time camera feed
    📱 **Rolling Display**: Shows recent 5 instructions
    
    **Perfect for blind navigation assistance - processes like a live camera stream!**
    """,
    examples=None,
    cache_examples=False,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()