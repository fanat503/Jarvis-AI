import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Настройки
cam_w, cam_h = 640, 480
# Размеры твоего экрана (узнай их! Обычно 1920x1080)
screen_w, screen_h = pyautogui.size() 

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

mp_face_mesh = mp.solutions.face_mesh
# Используем MESH, потому что он точнее (468 точек), чем просто Detection
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1) # Зеркалим, чтобы было удобно управлять
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_img)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Точка носа (ID 4 - кончик носа в FaceMesh)
        nose = landmarks[4] 
        
        # Переводим координаты из (0.0 - 1.0) в пиксели камеры
        x = int(nose.x * cam_w)
        y = int(nose.y * cam_h)
        
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        
        # --- ТВОЯ ЗАДАЧА (МАТЕМАТИКА) ---
        # Тебе нужно перевести координаты X, Y (внутри камеры 640x480)
        # в координаты mouse_x, mouse_y (внутри экрана 1920x1080).
        # Простая пропорция: mouse_x = (x / cam_w) * screen_w
        
        # ЗАПОЛНИ ЭТИ СТРОЧКИ:
        mouse_x = (x/cam_w) * screen_w
        mouse_y = (y/cam_h) * screen_h

        print(f"Nose: {x},{y} -> Mouse: {mouse_x},{mouse_y}") 
        
        # Двигаем мышку (pyautogui.moveTo)
        # ВАЖНО: Двигай только если координаты реальные, иначе мышка улетит.
        pyautogui.moveTo(int(mouse_x, mouse_y)) # <--- Раскомментируй, когда напишешь формулу
        
    cv2.imshow('Nose Mouse', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()