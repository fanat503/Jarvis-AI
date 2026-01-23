import cv2
import mediapipe as mp
import time
import math
import datetime
import os
import config
import ai_engine as ai

def main():
    # --- SYSTEM INIT ---
    print(">>> Initializing Core Modules...")
    
    nose_img = ai.load_nose_asset()

    # AI Models
    mp_face = mp.solutions.face_detection.FaceDetection(0.75)
    mp_hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, max_num_hands=1)
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
    
    # Storage Path
    storage_path = r"D:\Jarvis_Photos"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # --- MAIN LOOP ---
    while True:
        success, img = cap.get_frame()
        if not success or img is None:
            continue

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
                ai.speak("Image saved to drive")
                
            elif "nose" in cmd or "нос" in cmd:
                show_nose = not show_nose
                ai.speak("AR Filter toggled")
                
            elif "exit" in cmd or "выход" in cmd:
                ai.speak("Systems offline")
                break
                
            elif "weather" in cmd or "погода" in cmd:
                ai.speak("Accessing meteorological data...")
                ai.speak(ai.get_weather())
                
            elif "time" in cmd or "время" in cmd:
                now = datetime.datetime.now()
                ai.speak(f"Current time: {now.hour}:{now.minute}")
                
            # System Automation (Expanded triggers)
            elif any(k in cmd for k in ["open", "run", "search", "открой", "запусти", "найди", "youtube", "google", "ютуб", "гугл"]):
                ai.run_system_command(cmd)

        # 2. FACE TRACKING
        results_face = mp_face.process(img_rgb)
        if results_face.detections:
            for detection in results_face.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                nose_w = int(bbox.width * w * 0.5)
                kp = detection.location_data.relative_keypoints
                nx, ny = int(kp[2].x * w), int(kp[2].y * h)
                
                if show_nose:
                    img = ai.overlay_image(img, nose_img, nx - nose_w//2, ny - nose_w//2, (nose_w, nose_w))

        # 3. HAND GESTURES
        results_hands = mp_hands.process(img_rgb)
        if results_hands.multi_hand_landmarks:
            for lm in results_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, lm, mp.solutions.hands.HAND_CONNECTIONS)
                h, w, _ = img.shape
                
                # Thumb Tip (ID 4)
                x1, y1 = int(lm.landmark[4].x * w), int(lm.landmark[4].y * h)
                # Pinky Tip (ID 20)
                x2, y2 = int(lm.landmark[20].x * w), int(lm.landmark[20].y * h)
                
                dist = math.hypot(x2-x1, y2-y1)
                
                if dist < 30 and (time.time() - last_gesture_time) > config.GESTURE_COOLDOWN:
                    photo_count += 1
                    filepath = os.path.join(storage_path, f'GESTURE_SNAP_{photo_count}.jpg')
                    cv2.imwrite(filepath, img)
                    ai.speak("Gesture accepted")
                    last_gesture_time = time.time()

        # 4. UI RENDER
        status_text = "MIMIC" if parrot_mode else "COMMAND"
        try:
            img = ai.draw_hud(img, status_text) 
        except: pass

        cv2.imshow('Jarvis Interface', img)
        
        # Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            show_nose = not show_nose

    # --- SHUTDOWN ---
    voice.is_running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()