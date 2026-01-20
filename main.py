import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_detection
from mediapipe.python.solutions import hands
from mediapipe.python.solutions import drawing_utils
import threading
import math
import time
import numpy as np
import speech_recognition as sr
import pyttsx3
import datetime
import requests  # <--- Теперь просто и чисто

# --- 0. CONFIGURATION (НАСТРОЙКИ) ---
# ВСТАВЬ СЮДА СВОЙ КЛЮЧ!
API_KEY = "3360b5fa4d6bc687ac496ee3abbb4bde" 
CITY = "Minsk" 
# ПРОВЕРЬ IP КАМЕРЫ! (В приложении на телефоне)
CAMERA_URL = "http://192.168.100.7:8080/video" 

path = "nose.png"
show_nose = True
last_time = 0.0
cooldown = 2.0
count = 0

# --- 1. VOICE ENGINE ---
engine = pyttsx3.init()
engine.setProperty('rate', 200)

def speak(text):
    try:
        print(f"🤖 Jarvis: {text}")
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# --- 2. WEATHER FUNCTION ---
def get_weather():
    try:
        # Формируем ссылку
        url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric&lang=ru"
        
        # Отправляем запрос (ждем максимум 5 секунд)
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # Если сервер ответил ОК (200)
        if response.status_code == 200:
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return f"Сейчас в городе {CITY} {desc}. Температура {round(temp)} градусов."
        else:
            # Если ошибка (неверный город или ключ)
            print(f"⚠️ Ошибка сервера погоды: {data}")
            return "Не могу получить данные о погоде."
            
    except Exception as e:
        print(f"❌ Ошибка интернета: {e}")
        return "Нет доступа к интернету."

# --- 3. VOICE THREAD ---
class VoiceThread:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.last_command = None
        self.is_running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.daemon = True
        self.thread.start()
        
    def loop(self):
        print("🎤 Ears Online...")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.is_running:
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    try:
                        text = self.recognizer.recognize_google(audio, language="ru-RU").lower()
                        print(f"🗣️ Heard: {text}")
                        self.last_command = text
                    except:
                        pass
                except:
                    pass

# --- 4. CAMERA THREAD ---
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

# --- 5. IMAGE UTILS ---
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

# --- SETUP ---
nose_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if nose_img is None:
    nose_img = np.zeros((50,50,4), dtype=np.uint8)
    cv2.circle(nose_img, (25,25), 25, (0,0,255,255), -1)
else:
    nose_img = remove_white_bg(nose_img)

faceDetection = face_detection.FaceDetection(0.75)
hand_model = hands.Hands(model_complexity=0, min_detection_confidence=0.5)
mp_draw = drawing_utils

# --- START ---
cap = ThreadedCamera(CAMERA_URL)
voice = VoiceThread()
speak("Системы активны")

# --- MAIN LOOP ---
while True:
    success, img = cap.get_frame()
    if not success or img is None: continue # Если камера не работает - ждем

    img = cv2.resize(img, (800, 600))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. COMMANDS
    if voice.last_command:
        cmd = voice.last_command
        voice.last_command = None
        print(f"⚡ CMD: {cmd}")
        
        if "фото" in cmd:
            count += 1
            cv2.imwrite(f'VOICE_FOTO-{count}.jpg', img)
            speak("Готово")
        elif "нос" in cmd or "фильтр" in cmd:
            show_nose = not show_nose
            speak("Ок")
        elif "выход" in cmd:
            speak("Пока")
            break
        elif "время" in cmd:
            now = datetime.datetime.now()
            speak(f"Сейчас {now.hour} часов {now.minute} минут")
        elif "погода" in cmd:
            speak("Минуту...")
            report = get_weather()
            speak(report)

    # 2. FACE
    results = faceDetection.process(img_rgb)
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            nose_w = int(bbox.width * w * 0.5)
            kp = detection.location_data.relative_keypoints
            nx, ny = int(kp[2].x * w), int(kp[2].y * h)
            if show_nose:
                overlay_image(img, nose_img, nx - nose_w//2, ny - nose_w//2, (nose_w, nose_w))

    # 3. HANDS
    res_hands = hands.process(img_rgb)
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
                speak("Фото")
                last_time = time.time()

    cv2.imshow('Jarvis', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

voice.is_running = False
cap.capture.release()
cv2.destroyAllWindows()