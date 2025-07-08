import os

# --- Model and API Configuration ---
MODEL_ID = "google/gemma-3n-E4B-it"
# Set your Hugging Face token here if not using environment variables or Colab secrets
HF_TOKEN = os.getenv("HF_TOKEN") 

# --- System Performance ---
TARGET_FPS = int(os.getenv("TARGET_FPS", "3"))
CAPTURE_DURATION_SECONDS = 5  # How many seconds of video to analyze at a time

# --- Model Prompt Configuration ---
MAX_FRAMES_IN_PROMPT = int(os.getenv("MAX_FRAMES", "10")) # Max frames to send to the model
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "8000")) # Max input tokens for the model
MAX_NEW_TOKENS = 512 # Max tokens to generate in response

# --- Prompts ---
SYSTEM_PROMPT = (
    "You are an expert assistant for visually impaired users. "
    "Your role is to describe the user's surroundings based on a sequence of video frames. "
    "Be concise and clear. Focus on immediate obstacles, people, and important environmental features like doors, stairs, or crosswalks. "
    "Describe the scene as if you are right there with the user. For example, say 'A person is walking towards you on your left' instead of 'The image shows a person.'"
)
USER_PROMPT_WITH_AUDIO = (
    "The user has provided a spoken question. Analyze the video frames in the context of this audio question and provide a direct answer."
)

# --- Audio ---
WAKE_WORD = "assistant"