import cv2
import mediapipe as mp
import time
import math
import datetime
import os
import pyautogui 
import config
import ai_engine as ai

def main():
    # --- SYSTEM INIT ---
    print(">>> Initializing Core Modules...")
    
    nose_img = ai.load_nose_asset()

    # AI Models (FaceMesh for Mouse + Nose)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Hand Tracking
    mp_hands = mp.solutions.hands.Hands(
        model_complexity=0, 
        min_detection_confidence=0.5, 
        max_num_hands=1
    )
    mp_draw = mp.solutions.drawing_utils

    # Threads
    cap = ai.ThreadedCamera(config.CAMERA_URL)
    voice = ai.VoiceThread()
    
    ai.speak("Jarvis Core Online")

    # Runtime Vars
    show_nose = config.DEFAULT_NOSE_STATE
    last_gesture_time = 0.0
    photo_count = 0
    parrot_mode = False
    mouse_mode = False  # <--- Теперь переменная внутри main()
    
    # Screen Size for Mouse Control
    screen_w, screen_h = pyautogui.size()

    # Storage Path
    storage_path = r"D:\Jarvis_Photos"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # --- MAIN LOOP ---
    while True:
        success, img = cap.get_frame()
        if not success or img is None:
            continue

        # Flip image for Mouse Control (Mirror Effect)
        img = cv2.flip(img, 1) # <--- ВАЖНО для мышки!
        
        # Resize logic (FaceMesh needs original size better, but 800x600 is ok)
        img = cv2.resize(img, (800, 600))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. VOICE PROCESSING
        if voice.last_command:
            cmd = voice.last_command
            voice.last_command = None 
            print(f"⚡ CMD: {cmd}")
            
            # Mode Toggle
            if "parrot" in cmd or "попугай" in cmd:
                parrot_mode = not parrot_mode
                status = "active" if parrot_mode else "standby"
                ai.speak(f"Mimic mode {status}")
                continue 

            # Mimic Logic
            if parrot_mode:
                ai.speak(cmd)
                continue

            # Execution Logic
            if "photo" in cmd or "фото" in cmd:
                photo_count += 1
                filepath = os.path.join(storage_path, f'VOICE_SNAP_{photo_count}.jpg')
                cv2.imwrite(filepath, img)
                ai.speak("Image saved")
                
            elif "nose" in cmd or "нос" in cmd:
                show_nose = not show_nose
                ai.speak("Filter toggled")
                
            # MOUSE COMMAND
            elif "mouse" in cmd or "мышь" in cmd:
                mouse_mode = not mouse_mode
                status = "active" if mouse_mode else "disabled"
                ai.speak(f"Mouse control {status}")

            elif "exit" in cmd or "выход" in cmd:
                ai.speak("Systems offline")
                break
                
            elif "weather" in cmd or "погода" in cmd:
                ai.speak("Checking data...")
                ai.speak(ai.get_weather())
                
            elif "time" in cmd or "время" in cmd:
                now = datetime.datetime.now()
                ai.speak(f"It is {now.hour}:{now.minute}")
                
            # System Automation
            elif any(k in cmd for k in ["open", "run", "search", "открой", "запусти", "найди", "youtube", "google", "ютуб", "гугл"]):
                ai.run_system_command(cmd)

        # 2. FACE MESH (Mouse + Nose)
        # Обрати внимание: этот блок теперь внутри while True (с отступом)
        results_mesh = mp_face_mesh.process(img_rgb)
        
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                h, w, _ = img.shape
                
                # Nose Tip (Index 4)
                nose_pt = face_landmarks.landmark[4]
                nx, ny = int(nose_pt.x * w), int(nose_pt.y * h)
                
                # --- MOUSE CONTROL ---
                if mouse_mode:
                    # Рисуем прицел
                    cv2.circle(img, (nx, ny), 8, (0, 255, 0), -1)
                    # Двигаем (вызываем функцию из ai_engine)
                    # Важно: Передаем w, h (размеры камеры)
                    ai.move_mouse_with_nose(nx, ny, screen_w, screen_h, w, h)
                
                # --- NOSE FILTER ---
                if show_nose:
                    # FaceMesh не дает bbox, считаем ширину по скулам
                    # 234 (Left Ear), 454 (Right Ear)
                    left_ear = face_landmarks.landmark[234]
                    right_ear = face_landmarks.landmark[454]
                    
                    face_width = int(abs(left_ear.x - right_ear.x) * w)
                    nose_size = int(face_width * 0.4) 
                    
                    img = ai.overlay_image(img, nose_img, nx - nose_size//2, ny - nose_size//2, (nose_size, nose_size))

        # 3. HAND GESTURES
        results_hands = mp_hands.process(img_rgb)
        if results_hands.multi_hand_landmarks:
            for lm in results_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, lm, mp.solutions.hands.HAND_CONNECTIONS)
                h, w, _ = img.shape
                
                # Landmarks: 4 (Thumb), 20 (Pinky)
                x1, y1 = int(lm.landmark[4].x * w), int(lm.landmark[4].y * h)
                x2, y2 = int(lm.landmark[20].x * w), int(lm.landmark[20].y * h)
                
                dist = math.hypot(x2-x1, y2-y1)
                
                if dist < 30 and (time.time() - last_gesture_time) > config.GESTURE_COOLDOWN:
                    photo_count += 1
                    filepath = os.path.join(storage_path, f'GESTURE_SNAP_{photo_count}.jpg')
                    cv2.imwrite(filepath, img)
                    ai.speak("Gesture accepted")
                    last_gesture_time = time.time()

        # 4. UI RENDER
        # Показываем статус (MIMIC / MOUSE / COMMAND)
        if parrot_mode:
            status = "MIMIC"
        elif mouse_mode:
            status = "MOUSE CTRL"
        else:
            status = "COMMAND"
            
        try:
            img = ai.draw_hud(img, status) 
        except: pass

        cv2.imshow('Jarvis AI', img)
        
        # Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            show_nose = not show_nose
        elif key == ord('m'): # Hotkey for mouse
            mouse_mode = not mouse_mode

    # --- SHUTDOWN ---
    voice.is_running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()