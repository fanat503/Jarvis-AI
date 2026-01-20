import cv2
import mediapipe as mp
import threading
import math
import time
import numpy as np
import speech_recognition as sr
import pyttsx3
import datetime
import requests
import webbrowser
import subprocess
import os

# --- CONFIGURATION ---
API_KEY = "YOUR_OPENWEATHER_API_KEY_HERE"  # <--- INSERT YOUR KEY
CITY = "Minsk"
CAMERA_URL = "http://192.168.100.7:8080/video" # <--- CHECK YOUR IP

# --- ASSETS ---
NOSE_PATH = "nose.png"

# --- GLOBAL VARIABLES ---
show_nose = True
last_time = 0.0
cooldown = 2.0
count = 0

# --- VOICE ENGINE ---
engine = pyttsx3.init()
engine.setProperty('rate', 200)

def speak(text):
    try:
        print(f"🤖 Jarvis: {text}")
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# --- FEATURES ---
def get_weather():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric&lang=ru"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return f"Weather in {CITY}: {desc}, temperature {round(temp)} degrees."
        else:
            return "Cannot retrieve weather data. Check API key."
    except:
        return "Connection error."

def run_system_command(cmd):
    if "youtube" in cmd or "ютуб" in cmd:
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")
    
    elif "google" in cmd or "гугл" in cmd:
        speak("Opening Google")
        webbrowser.open("https://www.google.com")

    elif "search" in cmd or "найди" in cmd:
        query = cmd.replace("search", "").replace("найди", "").strip()
        if query:
            speak(f"Searching for {query}")
            webbrowser.open(f"https://www.google.com/search?q={query}")
        else:
            speak("What should I search for?")

    elif "calculator" in cmd or "калькулятор" in cmd:
        speak("Launching Calculator")
        subprocess.Popen('calc.exe')
    
    elif "notepad" in cmd or "блокнот" in cmd:
        speak("Opening Notepad")
        subprocess.Popen('notepad.exe')

# --- THREADS ---
class VoiceThread:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.last_command = None
        self.is_running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.daemon = True
        self.thread.start()
        
    def loop(self):
        print("🎤 Voice System Online")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.is_running:
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    try:
                        text = self.recognizer.recognize_google(audio, language="ru-RU").lower()
                        print(f"🗣️ Heard: {text}")
                        self.last_command = text
                    except: pass
                except: pass

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.capture.isOpened(): (self.status, self.frame) = self.capture.read()
            
    def get_frame(self): return self.status, self.frame

# --- IMAGE UTILS ---
def remove_white_bg(image, threshold=200):
    if image.shape[2] == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    white_pixels = np.where((image[:,:,0]>threshold) & (image[:,:,1]>threshold) & (image[:,:,2]>threshold))
    image[white_pixels[0], white_pixels[1], 3] = 0
    return image

def overlay_image(background, overlay, x, y, overlay_size=None):
    try:
        if overlay_size: overlay = cv2.resize(overlay, overlay_size)
        h, w, c = overlay.shape
        if x < 0 or y < 0: return background
        overlay_img = overlay[..., :3]
        mask = overlay[..., 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (mask * overlay_img[:,:,c] + (1.0-mask) * background[y:y+h, x:x+w, c])
        return background
    except: return background

# --- INITIALIZATION ---
nose_img = cv2.imread(NOSE_PATH, cv2.IMREAD_UNCHANGED)
if nose_img is None:
    nose_img = np.zeros((50,50,4), dtype=np.uint8)
    cv2.circle(nose_img, (25,25), 25, (0,0,255,255), -1)
else:
    nose_img = remove_white_bg(nose_img)

mp_face = mp.solutions.face_detection.FaceDetection(0.75)
mp_hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = ThreadedCamera(CAMERA_URL)
voice = VoiceThread()
speak("Systems active")

# --- MAIN LOOP ---
while True:
    success, img = cap.get_frame()
    if not success or img is None: continue

    img = cv2.resize(img, (800, 600))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Voice Logic
    if voice.last_command:
        cmd = voice.last_command
        voice.last_command = None
        print(f"⚡ CMD: {cmd}")
        
        if "photo" in cmd or "фото" in cmd:
            count += 1
            cv2.imwrite(f'VOICE_FOTO-{count}.jpg', img)
            speak("Photo saved")
            
        elif "nose" in cmd or "нос" in cmd:
            show_nose = not show_nose
            speak("Filter toggled")
            
        elif "exit" in cmd or "выход" in cmd:
            speak("Shutting down")
            break
            
        elif "time" in cmd or "время" in cmd:
            now = datetime.datetime.now()
            speak(f"It is {now.hour}:{now.minute}")
            
        elif "weather" in cmd or "погода" in cmd:
            speak("Checking weather...")
            report = get_weather()
            speak(report)
            
        elif any(x in cmd for x in ["open", "run", "search", "открой", "запусти", "найди", "ютуб", "гугл"]):
            run_system_command(cmd)

    # 2. Face Detection
    results = mp_face.process(img_rgb)
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            nose_w = int(bbox.width * w * 0.5)
            kp = detection.location_data.relative_keypoints
            nx, ny = int(kp[2].x * w), int(kp[2].y * h)
            if show_nose:
                overlay_image(img, nose_img, nx - nose_w//2, ny - nose_w//2, (nose_w, nose_w))

    # 3. Hand Detection
    res_hands = mp_hands.process(img_rgb)
    if res_hands.multi_hand_landmarks:
        for lm in res_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, lm, mp.solutions.hands.HAND_CONNECTIONS)
            h, w, c = img.shape
            x1, y1 = int(lm.landmark[4].x*w), int(lm.landmark[4].y*h)
            x2, y2 = int(lm.landmark[20].x*w), int(lm.landmark[20].y*h)
            length = math.hypot(x2-x1, y2-y1)
            
            if length < 30 and (time.time() - last_time) > cooldown:
                count += 1
                cv2.imwrite(f'GESTURE-{count}.jpg', img)
                speak("Gesture photo")
                last_time = time.time()

    cv2.imshow('Jarvis AI', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

voice.is_running = False
cap.capture.release()
cv2.destroyAllWindows()