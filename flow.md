# Flow

## 1. Overview

VidStreamInsight is an AI-powered assistant designed to aid visually impaired individuals by providing real-time descriptions of their surroundings. Using a continuous video feed, the system acts as a virtual guide, offering both passive narration and on-demand assistance for specific tasks.

## 2. System Workflow

The process can be broken down into four main stages:

### a. Video Input
- The system continuously captures video from a user-operated camera (e.g., a smartphone or a wearable device).
- This live video stream is the primary input for the AI model.

### b. Frame Processing
- The incoming video stream is broken down into a sequence of individual image frames.
- This is done at a consistent rate (frames per second) to ensure a smooth and up-to-date understanding of the environment.

### c. AI Analysis & Perception
- The extracted frames are fed into the Gemma 3n AI model.
- The model analyzes the sequence of frames to interpret the scene, identifying key elements such as:
    - Obstacles (e.g., curbs, poles, signs)
    - People and their direction of movement
    - Vehicles (cars, bicycles)
    - Environmental features (e.g., crosswalks, doors, stairs)
- The AI builds a continuous, real-time "perception" of the user's surroundings.

### d. Text-to-Speech (TTS) Conversion & Audio Output

The system provides guidance through spoken audio, which involves two stages:

1.  **Text Generation:** The Gemma 3n model generates descriptive text based on its visual analysis.

2.  **Speech Synthesis:** This text is then fed into a Text-to-Speech (TTS) engine, which converts it into natural-sounding audio for the user.

The system provides guidance in two primary modes:

1.  **Continuous Guidance (Passive Mode):**
    - The AI proactively generates text describing the environment, which is converted to speech.
    - It provides a running commentary, much like a guide dog, warning of upcoming obstacles or changes in the environment.
    - *Example Output: "You are approaching a curb in 5 feet. A person is walking towards you on your left."*

2.  **On-Demand Assistance (Active Mode):**
    - The user can ask specific questions or request help with a task.
    - *Example User Command: "Help me cross this road."*
    - The AI then focuses its analysis on the specific request, providing step-by-step instructions.
    - *Example Output for Crossing a Road: "The crosswalk light is red. Please wait. The light is now green. I do not detect any oncoming vehicles. It is safe to cross."*


Pseudocode:

// Initialization
PROCEDURE Initialize():
    Load Gemma3n model and processor
    Configure system parameters (target FPS, max frames, etc.)
    Set up user interface components

// Main Input Processing
PROCEDURE ProcessUserInput(user_message):
    IF message contains video:
        Extract frames from video at target FPS (up to max frames)
        Convert video to sequence of image frames
    ELSE IF message contains images/audio:
        Process media files directly
        
    Return structured message with text and media

// History Management
PROCEDURE FormatConversationHistory(history):
    Organize past messages into proper format for model
    Return structured conversation history

// AI Generation
PROCEDURE GenerateResponse(user_input, conversation_history):
    Format inputs for Gemma3n model
    Check if input exceeds token limits
    
    // Start streaming generation
    Begin asynchronous text generation
    
    WHILE model is generating:
        Collect text chunks as they're produced
        Stream chunks back to user interface
        
    Return complete generated response

// Main Application Loop
PROCEDURE RunApplication():
    Initialize()
    
    WHILE application is running:
        Wait for user input
        processed_input = ProcessUserInput(user_input)
        conversation = FormatConversationHistory(history)
        response = GenerateResponse(processed_input, conversation)
        Display response to user