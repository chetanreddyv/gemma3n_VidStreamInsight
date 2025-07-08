import speech_recognition as sr
import pyttsx3
from threading import Lock

class AudioHandler:
    """Handles all audio input (STT) and output (TTS)."""
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_speaking = Lock()
        
        # Adjust for ambient noise once at the beginning
        with self.microphone as source:
            print("Calibrating microphone for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Calibration complete.")

    def speak(self, text):
        """Converts text to speech, ensuring sequential output."""
        with self.is_speaking:
            print(f"ASSISTANT: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def listen_for_wake_word(self, wake_word, timeout=5):
        """Listens for a specific wake word."""
        with self.microphone as source:
            print(f"\n--- Listening for Wake Word ('{wake_word}') ---")
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
                text = self.recognizer.recognize_google(audio)
                print(f"Heard: {text}")
                return wake_word in text.lower()
            except sr.UnknownValueError:
                print("No speech detected.")
            except sr.WaitTimeoutError:
                print("Listening for wake word timed out.")
            except sr.RequestError as e:
                print(f"STT Request Error: {e}")
        return False

    def listen_for_command(self, timeout=5, phrase_time_limit=5):
        """Listens for a user command and returns the audio data."""
        with self.microphone as source:
            try:
                print("Listening for your command...")
                audio_command = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("USER COMMAND (audio captured)")
                return audio_command.get_wav_data()
            except sr.WaitTimeoutError:
                self.speak("I didn't hear a command. Please try again.")
            except sr.RequestError as e:
                print(f"STT Request Error: {e}")
                self.speak("Sorry, I'm having trouble with speech recognition.")
        return None