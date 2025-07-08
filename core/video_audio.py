import spaces
import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# Load model and processor
MODEL_PATH = "google/gemma-3n-E4B-it"
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
).eval().to("cuda")

@spaces.GPU
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

# Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Image(label="Upload Image", type="pil"),
        gr.Audio(label="Ask Question about the Image", type="filepath")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Visual (Audio) Question Answering",
    description="Upload an image as context and ask a quesiton about the image. The model will generate a text response.",
    examples=[["cat.jpg", "cats.wav"]]
)

if __name__ == "__main__":
    iface.launch()