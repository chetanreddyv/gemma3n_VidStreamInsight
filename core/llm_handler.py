import torch
import pathlib
from threading import Thread
from transformers.generation.streamers import TextIteratorStreamer
import config

def format_prompt(text_prompt, frames, audio_path=None):
    """Formats the prompt for the Gemma model."""
    prompt_parts = [text_prompt]
    if audio_path:
        prompt_parts.append(pathlib.Path(audio_path))
    for frame_path in frames:
        prompt_parts.append(pathlib.Path(frame_path))
    return prompt_parts

@torch.inference_mode()
def generate_response(model, processor, prompt):
    """Generates a response from the model in a streaming fashion."""
    inputs = processor(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    if inputs["input_ids"].shape[1] > config.MAX_INPUT_TOKENS:
        yield "The current context is too long. Please start a new conversation."
        return

    streamer = TextIteratorStreamer(
        processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=False,
    )

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    buffer = ""
    print("ASSISTANT (streaming): ", end="", flush=True)
    for delta in streamer:
        buffer += delta
        print(delta, end="", flush=True)
        if any(p in buffer for p in ".!?"):
            # Split by the last punctuation mark to get complete sentences
            last_punc = max(buffer.rfind(p) for p in ".!?")
            sentence = buffer[:last_punc+1].strip()
            if sentence:
                yield sentence
            buffer = buffer[last_punc+1:]
    
    if buffer.strip():
        yield buffer.strip()
    print() # Newline after streaming