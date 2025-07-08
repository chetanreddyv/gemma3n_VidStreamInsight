import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import config

def load_model():
    """
    Loads the Gemma model and processor from Hugging Face.
    Handles authentication for gated models.
    """
    print("Initializing model...")
    
    # The user's provided code for Colab secrets is not robust for local execution.
    # The recommended way is to use `huggingface-cli login` or set HF_TOKEN env var.
    # transformers will automatically use the token.
    
    try:
        processor = AutoProcessor.from_pretrained(config.MODEL_ID, token=config.HF_TOKEN)
        model = AutoModelForImageTextToText.from_pretrained(
            config.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=config.HF_TOKEN
            # attn_implementation="flash_attention_2" # Optional: for faster inference
        )
        print("Model initialized successfully.")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you are logged in to Hugging Face (use 'huggingface-cli login') or have set the HF_TOKEN in config.py.")
        exit()