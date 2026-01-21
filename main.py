import cv2
import mediapipe as mp
import time
import math
import datetime
import config
import ai_engine as ai

def main():
    # --- INITIALIZATION ---
    print(">>> Initializing AI Modules...")
    
    # 1. Load Assets
    nose_img = ai.load_nose_asset()

    # 2. Setup Computer Vision (MediaPipe)
    mp_face = mp.solutions.face_detection.FaceDetection(0.75)
    mp_hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # 3. Start I/O Threads
    cap = ai.ThreadedCamera(config.CAMERA_URL)
    voice = ai.VoiceThread()
    
    ai.speak("Jarvis Core Online")

    # 4. Runtime Variables
    show_nose = config.DEFAULT_NOSE_STATE
    last_gesture_time = 0.0
    photo_count = 0

    # --- MAIN LOOP ---
    while True:
        # 1. Frame Acquisition
        success, img = cap.get_frame()
        if not success or img is None:
            continue

        # Resize for performance
        img = cv2.resize(img, (800, 600))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Voice Command Handling
        if voice.last_command:
            cmd = voice.last_command
            voice.last_command = None # Ack command
            
            # --- Command Router ---
            if "photo" in cmd or "фото" in cmd:
                photo_count += 1
                filename = f'VOICE_FOTO-{photo_count}.jpg'
                cv2.imwrite(filename, img)
                ai.speak("Snapshot captured")
                
            elif "nose" in cmd or "нос" in cmd:
                show_nose = not show_nose
                ai.speak("Filter toggled")
                
            elif "exit" in cmd or "выход" in cmd:
                ai.speak("Terminating systems")
                break
                
            elif "weather" in cmd or "погода" in cmd:
                ai.speak("Retrieving data...")
                ai.speak(ai.get_weather())
                
            elif "time" in cmd or "время" in cmd:
                now = datetime.datetime.now()
                ai.speak(f"Time is {now.hour}:{now.minute}")
                
            # System Automation Trigger
            elif any(k in cmd for k in ["open", "run", "search", "открой", "запусти", "найди"]):
                ai.run_system_command(cmd)

        # 3. Face Tracking & AR
        results_face = mp_face.process(img_rgb)
        if results_face.detections:
            for detection in results_face.detections:
                # Bounding box calc
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                
                # Keypoints
                kp = detection.location_data.relative_keypoints
                nose_tip = kp[2]
                
                # Render Overlay
                if show_nose:
                    size = int(bbox.width * w * 0.5)
                    nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)
                    # Adjust centering
                    img = ai.overlay_image(img, nose_img, nx - size//2, ny - size//2, (size, size))

        # 4. Hand Gesture Recognition
        results_hands = mp_hands.process(img_rgb)
        if results_hands.multi_hand_landmarks:
            for lm in results_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, lm, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Pinch Detection (Thumb tip to Index tip)
                h, w, _ = img.shape
                x1, y1 = int(lm.landmark[4].x * w), int(lm.landmark[4].y * h)
                x2, y2 = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h) # Using index finger tip (id 8)
                
                dist = math.hypot(x2-x1, y2-y1)
                
                if dist < 30 and (time.time() - last_gesture_time) > config.GESTURE_COOLDOWN:
                    photo_count += 1
                    cv2.imwrite(f'GESTURE-{photo_count}.jpg', img)
                    ai.speak("Gesture accepted")
                    last_gesture_time = time.time()

        # 5. Display & Input
        cv2.imshow('Jarvis AI Interface', img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            show_nose = not show_nose

    # --- CLEANUP ---
    print(">>> Shutting down...")
    voice.is_running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()