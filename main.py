import cv2
import mediapipe as mp
import threading
import math
import time
import numpy as np

# Control flags
show_nose = True
show_blur = True 

path = r"nose.png"

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

# Load resources
nose_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if nose_img is None:
    nose_img = np.zeros((50, 50, 4), dtype=np.uint8)
    cv2.circle(nose_img, (25, 25), 25, (0, 0, 255, 255), cv2.FILLED)
else:
    nose_img = remove_white_bg(nose_img)

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

# App settings
last_time = 0.0
cooldown = 2.0
count = 0

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

# Make sure the IP is correct
cap = ThreadedCamera("http://192.168.100.7:8080/video")

print("Controls:")
print("'n' - Toggle Nose")
print("'b' - Toggle Blur")
print("'q' - Quit")

while True:
    success, img = cap.get_frame()
    if not success or img is None: continue

    img = cv2.resize(img, (800, 600))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Face Detection
    results_face = faceDetection.process(img_rgb)
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            # BLUR LOGIC
            if show_blur:
                try:
                    # Safety check for boundaries
                    y1, y2 = max(0, y), min(ih, y+h)
                    x1, x2 = max(0, x), min(iw, x+w)
                    
                    if x2 > x1 and y2 > y1:
                        face_roi = img[y1:y2, x1:x2]
                        # Apply heavy Gaussian blur
                        face_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
                        img[y1:y2, x1:x2] = face_roi
                except:
                    pass

            # Nose calculations
            keypoints = detection.location_data.relative_keypoints
            nose_tip = keypoints[2]
            nose_center_x = int(nose_tip.x * iw)
            nose_center_y = int(nose_tip.y * ih)

            nose_width = int(w * 0.5)
            nose_height = int(w * 0.5)
            nose_x = nose_center_x - nose_width // 2
            nose_y = nose_center_y - nose_height // 2 - int(ih * 0.05)

            if show_nose:
                overlay_image(img, nose_img, nose_x, nose_y, (nose_width, nose_height))

    # 2. Hand Detection
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
                    last_time = current_time
                    count += 1
                    file_name = f'MY_FOTO-{count}.jpg'
                    cv2.imwrite(file_name, img)
                    print(f"SAVED: {file_name}")
                    white_flash = np.ones((600, 800, 3), dtype='uint8') * 255
                    cv2.imshow('Selfie Camera', white_flash)
                    cv2.waitKey(100)
                else:
                    cv2.circle(img, (x2, y2), 15, (0, 255, 255), cv2.FILLED)

    cv2.imshow('Selfie Camera', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    
    if key == ord('n'): 
        show_nose = not show_nose
        print(f"Nose: {show_nose}")

    if key == ord('b'):
        show_blur = not show_blur
        print(f"Blur: {show_blur}")

cap.capture.release() 
cv2.destroyAllWindows()