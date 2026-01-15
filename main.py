import cv2
import mediapipe as mp
import threading
import math
import time
import numpy as np
import speech_recognition as sr
import pyttsx3
import datetime

# --- 1. SETTINGS & SETUP ---
last_time = 0.0
cooldown = 2.0
count = 0
show_nose = True
path = "nose.png"

# --- 2. VOICE ENGINE (Audio Output) ---
engine = pyttsx3.init()
engine.setProperty('rate', 200)

def speak(text):
    # We use a lock or run inside main thread to avoid crashes
    try:
        print(f"🤖 Jarvis: {text}")
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# --- 3. VOICE THREAD (The Ears) 👂 ---
# This runs in parallel with the camera!
class VoiceThread:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.last_command = None # Here we store the command
        self.is_running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.daemon = True # Kills thread when app closes
        self.thread.start()
        
    def loop(self):
        print("🎤 Voice System Online (Background)")
        with sr.Microphone() as source:
            # Calibrate noise level once
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.is_running:
                try:
                    # Listen without blocking the main video loop effectively
                    # phantom listen to keep loop alive
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    
                    try:
                        text = self.recognizer.recognize_google(audio, language="ru-RU").lower()
                        print(f"🗣️ Heard: {text}")
                        self.last_command = text # Send command to Main Thread
                    except sr.UnknownValueError:
                        pass # Silence is golden
                    except sr.RequestError:
                        print("❌ Internet Error")
                        
                except Exception as e:
                    pass

# --- 4. CAMERA THREAD (The Eyes) 👀 ---
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

# --- 5. IMAGE PROCESSING ---
def remove_white_bg(image, threshold=200):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    white_pixels = np.where(
        (image[:, :, 0] > threshold) & 
        (image[:, :, 1] > threshold) & 
        (image[:, :, 2] > threshold)
    )
    image[white_pixels[0], white_pixels[1], 3] = 0
    return image

def overlay_image(background, overlay, x, y, overlay_size=None):
    try:
        if overlay_size is not None:
            overlay = cv2.resize(overlay, overlay_size)
        h, w, c = overlay.shape
        if x + w > background.shape[1]: w = background.shape[1] - x
        if y + h > background.shape[0]: h = background.shape[0] - y
        if x < 0 or y < 0: return background
        overlay_img = overlay[:h, :w, :3]
        alpha_mask = overlay[:h, :w, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(0, 3):
            background[y:y+h, x:x+w, c] = (alpha_mask * overlay_img[:, :, c] + 
                                          alpha_inv * background[y:y+h, x:x+w, c])
        return background
    except:
        return background

# Load resources
nose_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if nose_img is None:
    nose_img = np.zeros((50, 50, 4), dtype=np.uint8)
    cv2.circle(nose_img, (25, 25), 25, (0, 0, 255, 255), cv2.FILLED)
else:
    nose_img = remove_white_bg(nose_img)

# Initialize AI Models
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# START SYSTEMS
cap = ThreadedCamera("http://192.168.100.7:8080/video") # Check IP!
voice = VoiceThread() # <--- Starting the Ears

speak("Джарвис на связи. Системы в норме.")

# --- MAIN LOOP ---
while True:
    success, img = cap.get_frame()
    if not success or img is None: continue

    img = cv2.resize(img, (800, 600))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # --- LOGIC: Check Voice Commands ---
    # Мы проверяем переменную last_command в каждом кадре
    if voice.last_command:
        cmd = voice.last_command
        voice.last_command = None # "Стираем" команду, чтобы не выполнять вечно
        
        print(f"⚡ EXECUTE: {cmd}")
        
        if "фото" in cmd:
            speak("Делаю снимок")
            count += 1
            cv2.imwrite(f'VOICE_FOTO-{count}.jpg', img)
            
        elif "нос" in cmd or "фильтр" in cmd:
            show_nose = not show_nose
            if show_nose: speak("Фильтр активен")
            else: speak("Фильтр выключен")
            
        elif "выход" in cmd or "пока" in cmd:
            speak("До свидания")
            break
        elif 'время' in cmd:
            now = datetime.datetime.now()
            current_time = f'Сейчас {now.hour} часов и {now.minute} минут'
            time_speak = speak(current_time, )    

    # --- VISION: Face ---
    results_face = faceDetection.process(img_rgb)
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            w = int(bboxC.width * iw)
            keypoints = detection.location_data.relative_keypoints
            nose_tip = keypoints[2]
            nose_x = int(nose_tip.x * iw) - int(w*0.5)//2
            nose_y = int(nose_tip.y * ih) - int(w*0.5)//2 - int(ih * 0.05)

            if show_nose:
                overlay_image(img, nose_img, nose_x, nose_y, (int(w*0.5), int(w*0.5)))

    # --- VISION: Hands ---
    results_hands = hands.process(img_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, c = img.shape
            x1 = int(hand_landmarks.landmark[4].x * w)
            y1 = int(hand_landmarks.landmark[4].y * h)
            x2 = int(hand_landmarks.landmark[20].x * w)
            y2 = int(hand_landmarks.landmark[20].y * h)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            length = math.hypot(x2 - x1, y2 - y1)
            if length < 30:
                current_time = time.time()
                if current_time - last_time > cooldown:
                    count += 1
                    cv2.imwrite(f'MY_FOTO-{count}.jpg', img)
                    speak("Жест принят")
                    white_flash = np.ones((600, 800, 3), dtype='uint8') * 255
                    cv2.imshow('Selfie Camera', white_flash)
                    cv2.waitKey(100)
                    last_time = current_time

    cv2.imshow('Selfie Camera', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

voice.is_running = False # Stop the ears
cap.capture.release() 
cv2.destroyAllWindows()