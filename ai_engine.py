import cv2
import threading
import speech_recognition as sr
import pyttsx3
import requests
import webbrowser
import subprocess
import time
import numpy as np
import config
import pyautogui

# --- AUDIO ENGINE ---
engine = pyttsx3.init()
engine.setProperty('rate', 200)

def speak(text):
    """Outputs text via TTS system."""
    try:
        print(f"🤖 Jarvis: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Audio Error: {e}")

class VoiceThread:
    """Async voice recognition worker."""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.last_command = None
        self.is_running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        
    def loop(self):
        print("🎤 Voice Service: Online")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.is_running:
                try:
                    # Listen with timeout to prevent blocking
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    try:
                        text = self.recognizer.recognize_google(audio, language="ru-RU").lower()
                        print(f"🗣️ Input: {text}")
                        self.last_command = text
                    except sr.UnknownValueError:
                        pass
                except Exception:
                    # Sleep briefly if microphone fails to avoid CPU spike
                    time.sleep(0.5)

# --- VIDEO ENGINE ---
class ThreadedCamera:
    """Non-blocking video capture class."""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            
    def get_frame(self): return self.status, self.frame
    
    def release(self): self.capture.release()

# --- UTILITIES ---
def get_weather():
    """Fetches weather data via API."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={config.CITY}&appid={config.API_KEY}&units=metric&lang=ru"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return f"Conditions in {config.CITY}: {desc}, {round(temp)}°C."
        return "Weather service unavailable."
    except: return "Network error."

def run_system_command(cmd):
    """Executes OS level commands."""
    if any(k in cmd for k in ["youtube", "ютуб"]):
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")
        
    elif any(k in cmd for k in ["google", "гугл"]):
        speak("Opening Google")
        webbrowser.open("https://www.google.com")
        
    elif any(k in cmd for k in ["search", "найди"]):
        query = cmd.replace("search", "").replace("найди", "").strip()
        if query:
            speak(f"Searching: {query}")
            webbrowser.open(f"https://www.google.com/search?q={query}")
        else: speak("Query empty.")
            
    elif any(k in cmd for k in ["calculator", "калькулятор"]):
        speak("Launching Calc")
        subprocess.Popen('calc.exe')
        
    elif any(k in cmd for k in ["notepad", "блокнот"]):
        speak("Opening Notepad")
        subprocess.Popen('notepad.exe')

# --- GRAPHICS ---
def draw_hud(img, mode_text):
    """Renders the Heads-Up Display (HUD)."""
    # Simple black box for performance
    cv2.rectangle(img, (0, 0), (200, 100), (0, 0, 0), -1)
    
    # Text Elements
    cv2.putText(img, 'JARVIS SYSTEM', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(img, f"Mode: {mode_text}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

def remove_white_bg(image, threshold=200):
    if image.shape[2] == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    white_px = np.where((image[...,0]>threshold) & (image[...,1]>threshold) & (image[...,2]>threshold))
    image[white_px[0], white_px[1], 3] = 0
    return image

def overlay_image(background, overlay, x, y, size=None):
    try:
        if size: overlay = cv2.resize(overlay, size)
        h, w, _ = overlay.shape
        if x < 0 or y < 0 or x+w > background.shape[1] or y+h > background.shape[0]:
            return background
            
        fg = overlay[..., :3]
        alpha = overlay[..., 3] / 255.0
        
        for c in range(3):
            background[y:y+h, x:x+w, c] = (alpha * fg[...,c] + (1.0-alpha) * background[y:y+h, x:x+w, c])
        return background
    except: return background

def load_nose_asset():
    img = cv2.imread(config.NOSE_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.zeros((50,50,4), dtype=np.uint8)
        cv2.circle(img, (25,25), 25, (0,0,255,255), -1)
    else:
        img = remove_white_bg(img)
    return img


def move_mouse_with_nose(nose_x, nose_y, screen_w, screen_h, cam_w, cam_h):
    """
    Moves mouse cursor based on nose position using linear mapping.
    """
    try:
        # 1. Map Coordinates (Proportion)
        # Отражаем X (cam_w - nose_x), потому что мышку двигать зеркально неудобно
        # Если ты используешь flip в камере, то (cam_w - nose_x) не нужно.
        # Давай использовать чистую математику:
        
        target_x = int((nose_x / cam_w) * screen_w)
        target_y = int((nose_y / cam_h) * screen_h)
        
        # 2. Safety Bounds (Чтобы мышка не улетела в бесконечность)
        target_x = max(0, min(screen_w, target_x))
        target_y = max(0, min(screen_h, target_y))
        
        # 3. Move (Instantly)
        # _pause=False отключает встроенную задержку pyautogui (для скорости)
        pyautogui.moveTo(target_x, target_y, _pause=False)
        
    except Exception:
        pass    