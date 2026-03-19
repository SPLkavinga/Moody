import cv2
import mediapipe as mp
import pyautogui
import time
import threading
from queue import Queue
import numpy as np

# Threaded Camera Capture 
class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.queue = Queue(maxsize=1)
        self.running = True
        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()  # drop old frame
                except:
                    pass
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def release(self):
        self.running = False
        self.cap.release()


# Gesture Mouse Control

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
mid_screen_y = screen_height // 2

# smoothing
smooth_factor = 0.5
prev_mouse_x, prev_mouse_y = 0, 0

# states
is_active = False
last_toggle_time = 0

# left click/drag state
left_gesture_start = None
dragging = False
CLICK_HOLD_TIME = 0.5

# right click state
right_click_start = None
right_click_active = False
last_right_click_time = 0
RIGHT_CLICK_HOLD = 0.3
RIGHT_CLICK_COOLDOWN = 1.0

# vertical scroll
SCROLL_INTERVAL = 0.15
V_DEADZONE = max(30, int(screen_height * 0.05))
SCROLL_STEP = 50
last_scroll_time = 0.0


# Utility Functions

def landmarks_to_array(lm_list):
    """Convert hand landmarks to numpy array (x,y,z)"""
    return np.array([[lm.x, lm.y, lm.z] for lm in lm_list])

def finger_extended_np(lms, tip_idx, pip_idx):
    """Return True if finger is extended"""
    return lms[tip_idx, 1] < lms[pip_idx, 1]

def thumb_really_extended_np(lms):
    """Strict thumb check (extended, apart from index)"""
    tip = lms[mp_hands.HandLandmark.THUMB_TIP.value]
    ip = lms[mp_hands.HandLandmark.THUMB_IP.value]
    index_tip = lms[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
    return tip[0] < ip[0] and abs(tip[0] - index_tip[0]) > 0.08

def five_fingers_extended(lms):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(finger_extended_np(lms, t, p) for t, p in zip(tips, pips)) and thumb_really_extended_np(lms)


# Main Loop

stream = CameraStream()

with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=1) as hands:  # track one hand only

    last_frame_time = 0
    display_interval = 0.03  # reduce OpenCV display update rate 

    while True:
        frame = stream.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        now = time.time()

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lms = landmarks_to_array(hand_landmarks.landmark)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Activation,Deactivation using 5 fingers extended
            if five_fingers_extended(lms) and (now - last_toggle_time > 1.5):
                is_active = not is_active
                print("Virtual Mouse " + ("Activated" if is_active else "Deactivated"))
                last_toggle_time = now

            if is_active:
                # Finger states
                index_ext = finger_extended_np(lms, 8, 6)
                middle_ext = finger_extended_np(lms, 12, 10)
                ring_ext = finger_extended_np(lms, 16, 14)
                pinky_ext = finger_extended_np(lms, 20, 18)
                thumb_ext = thumb_really_extended_np(lms)

                # Cursor movement 
                index_tip = lms[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                target_x = int(index_tip[0] * screen_width)
                target_y = int(index_tip[1] * screen_height)

                if index_ext:
                    mouse_x = int(prev_mouse_x + (target_x - prev_mouse_x) * smooth_factor)
                    mouse_y = int(prev_mouse_y + (target_y - prev_mouse_y) * smooth_factor)
                    if abs(mouse_x - prev_mouse_x) > 2 or abs(mouse_y - prev_mouse_y) > 2:
                        pyautogui.moveTo(mouse_x, mouse_y)
                        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                # Left click drag 
                left_click_gesture = thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext
                if left_click_gesture:
                    if left_gesture_start is None:
                        left_gesture_start = now
                    elif not dragging and (now - left_gesture_start > CLICK_HOLD_TIME):
                        pyautogui.mouseDown()
                        dragging = True
                        print("Drag Start")
                else:
                    if left_gesture_start is not None:
                        hold_time = now - left_gesture_start
                        left_gesture_start = None
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False
                            print("Drag End")
                        elif hold_time <= CLICK_HOLD_TIME:
                            pyautogui.click()
                            print("Left Click")

                # Right click (rock sign) 
                rock_sign = index_ext and pinky_ext and not middle_ext and not ring_ext and not thumb_ext
                if rock_sign:
                    if right_click_start is None:
                        right_click_start = now
                    elif (not right_click_active
                          and (now - right_click_start > RIGHT_CLICK_HOLD)
                          and (now - last_right_click_time > RIGHT_CLICK_COOLDOWN)):
                        pyautogui.click(button="right")
                        right_click_active = True
                        last_right_click_time = now
                        print("Right Click (Rock Sign)")
                else:
                    right_click_start = None
                    right_click_active = False

                # Vertical Scroll
                scroll_gesture = index_ext and middle_ext and ring_ext and not pinky_ext
                if scroll_gesture and (now - last_scroll_time >= SCROLL_INTERVAL):
                    avg_y = np.mean([lms[8,1], lms[12,1], lms[16,1]])
                    finger_y_screen = int(avg_y * screen_height)
                    dy = finger_y_screen - mid_screen_y
                    if abs(dy) > V_DEADZONE:
                        if dy > 0:
                            pyautogui.scroll(-50)
                            print("Scroll Down")
                        else:
                            pyautogui.scroll(50)
                            print("Scroll Up")
                    last_scroll_time = now

        else:
            prev_mouse_x, prev_mouse_y = pyautogui.position()

        # Display status and gesture info on frame
        status_text = "ACTIVE" if is_active else "INACTIVE"
        status_color = (0, 255, 0) if is_active else (0, 0, 255)
        cv2.putText(frame, f"Status: {status_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display gesture and action info when active
        if is_active and results.multi_hand_landmarks:
            y_offset = 60
            
            # Current gesture detection
            if left_click_gesture:
                if dragging:
                    cv2.putText(frame, "Action: DRAGGING", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                else:
                    cv2.putText(frame, "Gesture: Thumb + Index (Click/Drag)", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
            elif rock_sign:
                cv2.putText(frame, "Gesture: Rock Sign (Right Click)", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
            elif scroll_gesture:
                cv2.putText(frame, "Gesture: 3 Fingers (Scroll)", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
            elif index_ext:
                cv2.putText(frame, "Gesture: Index Extended (Move Cursor)", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
        
        # Instructions
        cv2.putText(frame, "Show 5 fingers to toggle ON/OFF", (10, screen_height - 20 if 'screen_height' in dir() else 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Reduced OpenCV display rate
        if now - last_frame_time > display_interval:
            cv2.imshow("Virtual Mouse", frame)
            last_frame_time = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

stream.release()
cv2.destroyAllWindows()
