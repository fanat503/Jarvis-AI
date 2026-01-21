# ai_engine.py
import cv2
import threading
import speech_recognition as sr
import pyttsx3
import requests
import webbrowser
import subprocess
import numpy as np
import config

# --- TTS ENGINE ---
engine = pyttsx3.init()
engine.setProperty('rate', 200)

def speak(text):
    """Outputs audio response via TTS."""
    try:
        print(f"🤖 Jarvis: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Engine Error: {e}")

# --- ASYNC VOICE LISTENER ---
class VoiceThread:
    """Background thread for continuous speech recognition."""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.last_command = None
        self.is_running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
    def _worker(self):
        print("🎤 Voice Service: Active (Background)")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.is_running:
                try:
                    # Non-blocking listen with timeout
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    try:
                        text = self.recognizer.recognize_google(audio, language="ru-RU").lower()
                        print(f"🗣️ Input: {text}")
                        self.last_command = text
                    except sr.UnknownValueError:
                        pass # Ignore unintelligible noise
                except Exception:
                    pass

# --- VIDEO CAPTURE WORKER ---
class ThreadedCamera:
    """Decoupled video capture to prevent I/O blocking."""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status = False
        self.frame = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            
    def get_frame(self):
        return self.status, self.frame
    
    def release(self):
        self.capture.release()

# --- EXTERNAL INTEGRATIONS ---
def get_weather():
    """Fetches weather data via HTTP Request."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={config.CITY}&appid={config.API_KEY}&units=metric&lang=ru"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return f"Current conditions in {config.CITY}: {desc}, {round(temp)}°C."
        return "Weather API unavailable."
    except Exception:
        return "Network connection failed."

def run_system_command(cmd):
    """Handles OS-level commands and browser automation."""
    if "youtube" in cmd or "ютуб" in cmd:
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")
        
    elif "google" in cmd or "гугл" in cmd:
        speak("Opening Google")
        webbrowser.open("https://www.google.com")
        
    elif "search" in cmd or "найди" in cmd:
        query = cmd.replace("search", "").replace("найди", "").strip()
        if query:
            speak(f"Searching: {query}")
            webbrowser.open(f"https://www.google.com/search?q={query}")
        else:
            speak("Search query is empty.")
            
    elif "calculator" in cmd or "калькулятор" in cmd:
        speak("Launching Calculator")
        subprocess.Popen('calc.exe')
        
    elif "notepad" in cmd or "блокнот" in cmd:
        speak("Opening Notepad")
        subprocess.Popen('notepad.exe')

# --- IMAGE PROCESSING ---
def overlay_image(background, overlay, x, y, size=None):
    """Composites RGBA overlay onto RGB background."""
    try:
        if size:
            overlay = cv2.resize(overlay, size)
            
        h, w, _ = overlay.shape
        # Boundary checks
        if x < 0 or y < 0 or x+w > background.shape[1] or y+h > background.shape[0]:
            return background
            
        # Alpha blending
        fg = overlay[..., :3]
        alpha = overlay[..., 3] / 255.0
        
        for c in range(3):
            background[y:y+h, x:x+w, c] = (alpha * fg[...,c] + 
                                         (1.0-alpha) * background[y:y+h, x:x+w, c])
        return background
    except Exception:
        return background

def load_nose_asset():
    """Loads and preprocesses the nose asset."""
    img = cv2.imread(config.NOSE_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Fallback: Generate red circle if file missing
        img = np.zeros((50,50,4), dtype=np.uint8)
        cv2.circle(img, (25,25), 25, (0,0,255,255), -1)
        return img
        
    # Remove white background if present
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    white_px = np.where((img[...,0]>200) & (img[...,1]>200) & (img[...,2]>200))
    img[white_px[0], white_px[1], 3] = 0
    return img