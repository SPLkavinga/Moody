import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import webbrowser
import subprocess
import os
import platform
import warnings
import pyautogui
from queue import Queue
import json
import csv
import hashlib
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path
import sys
warnings.filterwarnings('ignore')

# Import theme configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from launcher import theme_config

# Import advanced analytics
from advanced_analytics import AdvancedAnalytics, ReportGenerator, REPORTLAB_AVAILABLE, PANDAS_AVAILABLE

# Import voice assistant
from voice_assistant import MoodyVoiceAssistant

# Model / features 
import joblib
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
from collections import deque
from live_emotion_inference import FEATURE_ORDER, compute_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model2")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# Hand Gesture Mouse Controller

class HandGestureController:
    def __init__(self):
        self.running = False
        self.is_active = False
        self.thread = None
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        self.mid_screen_y = self.screen_height // 2
        
        # Smoothing
        self.smooth_factor = 0.5
        self.prev_mouse_x, self.prev_mouse_y = 0, 0
        
        # States
        self.last_toggle_time = 0
        
        # Left click/drag
        self.left_gesture_start = None
        self.dragging = False
        self.CLICK_HOLD_TIME = 0.5
        
        # Right click
        self.right_click_start = None
        self.right_click_active = False
        self.last_right_click_time = 0
        self.RIGHT_CLICK_HOLD = 0.3
        self.RIGHT_CLICK_COOLDOWN = 1.0
        
        # Vertical scroll
        self.SCROLL_INTERVAL = 0.15
        self.V_DEADZONE = max(30, int(self.screen_height * 0.05))
        self.last_scroll_time = 0.0
        
        # MediaPipe hands
        self.mp_hands = mp_hands
        self.hands = None
        
    def landmarks_to_array(self, lm_list):
        return np.array([[lm.x, lm.y, lm.z] for lm in lm_list])
    
    def finger_extended_np(self, lms, tip_idx, pip_idx):
        return lms[tip_idx, 1] < lms[pip_idx, 1]
    
    def thumb_really_extended_np(self, lms):
        tip = lms[self.mp_hands.HandLandmark.THUMB_TIP.value]
        ip = lms[self.mp_hands.HandLandmark.THUMB_IP.value]
        index_tip = lms[self.mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
        return tip[0] < ip[0] and abs(tip[0] - index_tip[0]) > 0.08
    
    def five_fingers_extended(self, lms):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        return all(self.finger_extended_np(lms, t, p) for t, p in zip(tips, pips)) and self.thumb_really_extended_np(lms)
    
    def start(self, cap):
        if not self.running:
            self.running = True
            self.cap = cap
            self.thread = threading.Thread(target=self._run_gesture_control, daemon=True)
            self.thread.start()
            print("Hand Gesture Controller started")
    
    def stop(self):
        self.running = False
        self.is_active = False
        if self.hands:
            self.hands.close()
            self.hands = None
        print("Hand Gesture Controller stopped")
    
    def _run_gesture_control(self):
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=1
        )
        
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            now = time.time()
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                lms = self.landmarks_to_array(hand_landmarks.landmark)
                
                # Toggle activation with 5 fingers
                if self.five_fingers_extended(lms) and (now - self.last_toggle_time > 1.5):
                    self.is_active = not self.is_active
                    print("Virtual Mouse " + ("Activated" if self.is_active else "Deactivated"))
                    self.last_toggle_time = now
                
                if self.is_active:
                    # Finger states
                    index_ext = self.finger_extended_np(lms, 8, 6)
                    middle_ext = self.finger_extended_np(lms, 12, 10)
                    ring_ext = self.finger_extended_np(lms, 16, 14)
                    pinky_ext = self.finger_extended_np(lms, 20, 18)
                    thumb_ext = self.thumb_really_extended_np(lms)
                    
                    # Cursor movement
                    index_tip = lms[self.mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                    target_x = int(index_tip[0] * self.screen_width)
                    target_y = int(index_tip[1] * self.screen_height)
                    
                    if index_ext:
                        mouse_x = int(self.prev_mouse_x + (target_x - self.prev_mouse_x) * self.smooth_factor)
                        mouse_y = int(self.prev_mouse_y + (target_y - self.prev_mouse_y) * self.smooth_factor)
                        if abs(mouse_x - self.prev_mouse_x) > 2 or abs(mouse_y - self.prev_mouse_y) > 2:
                            pyautogui.moveTo(mouse_x, mouse_y)
                            self.prev_mouse_x, self.prev_mouse_y = mouse_x, mouse_y
                    
                    # Left click / drag
                    left_click_gesture = thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext
                    if left_click_gesture:
                        if self.left_gesture_start is None:
                            self.left_gesture_start = now
                        elif not self.dragging and (now - self.left_gesture_start > self.CLICK_HOLD_TIME):
                            pyautogui.mouseDown()
                            self.dragging = True
                            print("Drag Start")
                    else:
                        if self.left_gesture_start is not None:
                            hold_time = now - self.left_gesture_start
                            self.left_gesture_start = None
                            if self.dragging:
                                pyautogui.mouseUp()
                                self.dragging = False
                                print("Drag End")
                            elif hold_time <= self.CLICK_HOLD_TIME:
                                pyautogui.click()
                                print("Left Click")
                    
                    # Right click (rock sign)
                    rock_sign = index_ext and pinky_ext and not middle_ext and not ring_ext and not thumb_ext
                    if rock_sign:
                        if self.right_click_start is None:
                            self.right_click_start = now
                        elif (not self.right_click_active
                              and (now - self.right_click_start > self.RIGHT_CLICK_HOLD)
                              and (now - self.last_right_click_time > self.RIGHT_CLICK_COOLDOWN)):
                            pyautogui.click(button="right")
                            self.right_click_active = True
                            self.last_right_click_time = now
                            print("Right Click (Rock Sign)")
                    else:
                        self.right_click_start = None
                        self.right_click_active = False
                    
                    # Vertical Scroll
                    scroll_gesture = index_ext and middle_ext and ring_ext and not pinky_ext
                    if scroll_gesture and (now - self.last_scroll_time >= self.SCROLL_INTERVAL):
                        avg_y = np.mean([lms[8,1], lms[12,1], lms[16,1]])
                        finger_y_screen = int(avg_y * self.screen_height)
                        dy = finger_y_screen - self.mid_screen_y
                        if abs(dy) > self.V_DEADZONE:
                            if dy > 0:
                                pyautogui.scroll(-50)
                                print("Scroll Down")
                            else:
                                pyautogui.scroll(50)
                                print("Scroll Up")
                        self.last_scroll_time = now
            else:
                self.prev_mouse_x, self.prev_mouse_y = pyautogui.position()
            
            time.sleep(0.03)


# Main Emotion Recognition App

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root

        # Theme configuration
        self.current_theme = theme_config.get_current_theme()
        self.colors = theme_config.get_theme_colors()

        # App state
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.detection_active = False
        self._proba_window = deque(maxlen=10)

        # Canonical 7 labels used by UI/actions
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Hand gesture controller
        self.gesture_controller = HandGestureController()

        # Popup background mode state
        self.popup_window = None
        self.popup_actions_frame = None
        self.popup_gesture_btn = None
        self.popup_emotion_label = None

        # notification icon window
        self.notification_window = None
        self.notification_button = None

        # popup drag + remember position
        self._popup_drag_offset_x = 0
        self._popup_drag_offset_y = 0
        self._popup_last_x = None
        self._popup_last_y = None

        # Notification icon drag
        self._notif_drag_offset_x = 0
        self._notif_drag_offset_y = 0

        # Multi user profiles
        self.profiles_dir = os.path.join(os.path.dirname(__file__), "user_data")
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.current_user = None
        self.user_settings = {}
        self.emotion_log = []
        
        # Analytics tracking
        self.session_start_time = None
        self.emotion_durations = {emotion: 0.0 for emotion in self.emotion_labels}
        self.last_emotion_time = None
        self.emotion_streak_start = None
        self.emotion_streak_emotion = None
        self.daily_happy_spikes = 0
        self.calm_streak_start = None
        self.last_analytics_check = None
        
        # Advanced analytics engine
        self.analytics = AdvancedAnalytics()
        
        # Advanced analytics
        self.emotion_transitions = defaultdict(lambda: defaultdict(int))
        self.hourly_emotions = defaultdict(lambda: defaultdict(int))
        self.activity_log = []
        self.stress_indicators = []
        self.productivity_score = 0.0
        self.wellbeing_score = 0.0

        # Emotion actions with specific songs & games per emotion
        self.emotion_actions = {
            'happy': [
                ('🎵 Play "Happy" - Pharrell Williams', self.song_happy_pharrell),
                ('🎵 Play "Uptown Funk" - Bruno Mars', self.song_uptown_funk),
                ('🎵 Play "Good as Hell" - Lizzo', self.song_good_as_hell),
                ('🎮 Play Pac-Man Online', self.game_pacman),
                ('🎮 Play Friday Night Funkin\' (Music Battle)', self.game_friday_night_funkin),
                ('🎮 Open Games Platform', self.open_games),
                ('📺 Browse YouTube', self.open_youtube),
                ('📱 Open Social Media', self.open_social_media),
                ('📸 Launch Camera App', self.open_camera_app),
                ('🎨 Open Paint/Creative Tools', self.open_creative_apps),
                ('💬 Start Video Call', self.open_video_call),
                ('🎉 Play Party Music', self.play_party_music),
                ('📝 Create Happy Journal Entry', self.open_happy_journal),
            ],
            'sad': [
                ('🎵 Play "Fix You" - Coldplay', self.song_fix_you),
                ('🎵 Play "Lean on Me" - Bill Withers', self.song_lean_on_me),
                ('🎵 Play "Here Comes the Sun" - Beatles', self.song_here_comes_the_sun),
                ('🎮 Play Little Alchemy (Creative Discovery)', self.game_little_alchemy),
                ('🎮 Play Bubble Shooter (Soothing)', self.game_bubble_shooter),
                ('🎵 Play Comforting Music', self.play_comforting_music),
                ('🎬 Watch Comedy/Feel-Good Shows', self.watch_comedy),
                ('☕ View Self-Care Guide', self.show_selfcare_tips),
                ('📖 Read Motivational Content', self.show_motivational_quotes),
                ('🧘 Open Meditation App', self.open_meditation),
                ('💬 Connect with Friends', self.open_messaging),
                ('📝 Write in Journal', self.open_journal),
                ('🌈 Watch Mood-Lifting Videos', self.open_mood_lifting),
                ('🎧 Listen to Healing Sounds', self.play_healing_music),
                ('🆘 View Support Resources', self.show_support_resources),
                ('🌻 Positive Affirmations', self.show_affirmations),
                ('📞 Contact Helpline Info', self.show_emergency_contacts),
            ],
            'angry': [
                ('🎵 Play "Weightless" - Marconi Union', self.song_weightless),
                ('🎵 Play "Breathe Me" - Sia', self.song_breathe_me),
                ('🎵 Play "Let It Be" - Beatles', self.song_let_it_be),
                ('🎮 Play Slice Master (Slash It Out!)', self.game_slice_master),
                ('🎮 Play Punchers (Punch It Out!)', self.game_punchers),
                ('🎵 Play Calming Music', self.play_calming_music),
                ('🧘 Start Breathing Exercise', self.start_breathing_exercise),
                ('🏃 Open Workout/Exercise Videos', self.open_fitness_app),
                ('✍️ Vent in Journal', self.open_journal),
                ('🌿 Listen to Nature Sounds', self.play_nature_sounds),
                ('🎮 Play Stress-Relief Games', self.open_stress_games),
                ('🥊 Virtual Stress Relief', self.open_stress_relief),
                ('📊 Anger Management Tips', self.show_anger_tips),
                ('🎯 Redirect Energy Productively', self.suggest_productive_activity),
                ('💪 Physical Exercise Guide', self.suggest_exercise),
                ('🧊 Cool Down Technique', self.show_cooldown_tips),
                ('📉 Track Your Triggers', self.open_mood_tracker),
            ],
            'fear': [
                ('🎵 Play "Somewhere Over the Rainbow" - IZ', self.song_over_the_rainbow),
                ('🎵 Play "Brave" - Sara Bareilles', self.song_brave),
                ('🎵 Play "Three Little Birds" - Bob Marley', self.song_three_little_birds),
                ('🎮 Play Mahjong Solitaire (Calming Puzzle)', self.game_mahjong_solitaire),
                ('🎮 Play Color Fill (Relaxing)', self.game_color_fill),
                ('🎵 Play Comforting Music', self.play_comforting_music),
                ('🧘 Guided Meditation', self.open_meditation),
                ('📞 Emergency Contacts', self.show_emergency_contacts),
                ('💪 Empowerment Content', self.show_motivational_quotes),
                ('🌟 Positive Affirmations', self.show_affirmations),
                ('🔒 Safety Resources', self.show_safety_resources),
                ('💬 Support Chat', self.open_support_chat),
                ('🎧 Anxiety Relief Audio', self.play_anxiety_relief),
                ('🌬️ Breathing Exercises', self.start_breathing_exercise),
                ('📝 Write Your Worries', self.open_journal),
                ('🛡️ Grounding Techniques', self.show_grounding_techniques),
                ('☮️ Peace and Calm Guide', self.show_peace_guide),
            ],
            'surprise': [
                ('🎵 Play "Don\'t Stop Me Now" - Queen', self.song_dont_stop_me_now),
                ('🎵 Play "Celebration" - Kool & The Gang', self.song_celebration),
                ('🎵 Play "Walking on Sunshine"', self.song_walking_on_sunshine),
                ('🎮 Play Agar.io (Surprise Attack!)', self.game_agario),
                ('🎮 Play Helix Jump (How Far?)', self.game_helix_jump),
                ('📸 Capture This Moment', self.open_camera_app),
                ('📱 Share on Social Media', self.open_social_media),
                ('🎵 Play Energetic Music', self.play_upbeat_music),
                ('📝 Document Your Thoughts', self.open_journal),
                ('🎉 Celebration Ideas', self.show_celebration_ideas),
                ('📹 Record Video Message', self.open_video_recorder),
                ('🎊 Share Your Excitement', self.open_messaging),
                ('⚡ Explore Exciting Content', self.open_exciting_content),
                ('🎯 Channel This Energy', self.suggest_productive_activity),
                ('🌟 Reflect on the Moment', self.show_reflection_prompt),
            ],
            'disgust': [
                ('🎵 Play "What a Wonderful World" - Louis Armstrong', self.song_wonderful_world),
                ('🎵 Play "Ocean Eyes" - Billie Eilish', self.song_ocean_eyes),
                ('🎵 Play "Clair de Lune" - Debussy', self.song_clair_de_lune),
                ('🎮 Play Tile Guru (Zen Match)', self.game_tile_guru),
                ('🎮 Play Jigsaw Puzzle Online', self.game_jigsaw_puzzle),
                ('🌿 Fresh Air Reminder', self.suggest_fresh_air),
                ('🎵 Play Pleasant Music', self.play_relaxing_music),
                ('🧘 Mindfulness Exercise', self.open_meditation),
                ('🚿 Self-Care Routine Guide', self.show_selfcare_tips),
                ('🌸 View Beautiful Nature', self.show_nature_content),
                ('🧼 Cleansing Rituals', self.show_cleansing_tips),
                ('🍵 Comfort Recipes', self.show_comfort_recipes),
                ('🎨 Art Therapy', self.open_art_therapy),
                ('🌊 Cleansing Visualization', self.show_cleansing_visualization),
                ('💚 Reset Your Space', self.show_space_reset_tips),
            ],
            'neutral': [
                ('🎵 Play "Bohemian Rhapsody" - Queen', self.song_bohemian_rhapsody),
                ('🎵 Play "Blinding Lights" - The Weeknd', self.song_blinding_lights),
                ('🎵 Play "Lofi Hip Hop Radio"', self.song_lofi_radio),
                ('🎮 Play 2048 (Brain Teaser)', self.game_2048),
                ('🎮 Play Tetris Online', self.game_tetris),
                ('🎮 Play Chess Online', self.game_chess),
                ('🎵 Discover New Music', self.discover_music),
                ('📚 Learn Something New', self.open_learning_resources),
                ('🎮 Browse Casual Games', self.open_games),
                ('📺 Explore Entertainment', self.open_youtube),
                ('💭 Start Mood Journal', self.open_journal),
                ('🎯 Open Productivity Apps', self.open_productivity),
                ('🌐 Explore Your Interests', self.explore_interests),
                ('📊 Daily Planning Tool', self.open_planner),
                ('🧩 Brain Training Games', self.open_brain_games),
                ('📖 Read Articles/News', self.open_reading),
                ('🎨 Creative Projects', self.open_creative_apps),
                ('🌟 Set New Goals', self.open_goal_setting),
            ],
        }

        # UI,model,camera
        self.setup_ui()
        self.setup_model()
        self.setup_camera()
        self.setup_responsive_layout()
        
        # Analytics tab reference
        self.analytics_tab = None

        # Voice assistant
        self.voice_assistant = None
        self.voice_tab = None
        self.voice_log_text = None
        self.voice_status_label = None
        self.voice_bg_var = None

        # Start maximized
        try:
            if platform.system() == "Windows":
                self.root.state('zoomed')
            else:
                self.root.attributes('-zoomed', True)
        except Exception:
            pass
        
        # Auto login from launcher 
        launch_user = os.environ.get("MOODY_USER", None)
        if launch_user:
            self.root.after(500, lambda: self._load_user_profile(launch_user))
        else:
            # if launched directly without launcher go back to dashboard
            self.root.after(500, self._require_launcher_login)

    def setup_ui(self):
        self.root.title("Moody")
        self.root.configure(bg=self.colors['bg_primary'])
        self.root.minsize(1000, 700)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background=self.colors['bg_primary'])
        style.configure('Dark.TLabel', background=self.colors['bg_primary'], foreground=self.colors['text_primary'], font=('Segoe UI', 10))
        style.configure('Title.TLabel', background=self.colors['bg_primary'], foreground=self.colors['text_primary'], font=('Segoe UI', 18, 'bold'))
        style.configure('Emotion.TLabel', background=self.colors['bg_primary'], foreground=self.colors['accent_primary'], font=('Segoe UI', 16, 'bold'))
        style.configure('Dark.TButton', background=self.colors['bg_tertiary'], foreground=self.colors['text_primary'], font=('Segoe UI', 9), padding=8)
        style.map('Dark.TButton', background=[('active', self.colors['bg_secondary']), ('pressed', self.colors['bg_tertiary'])])
        style.configure('Gesture.TButton', background=self.colors['accent_secondary'], foreground=self.colors['text_primary'], font=('Segoe UI', 9), padding=8)
        style.map('Gesture.TButton', background=[('active', self.colors['accent_secondary']), ('pressed', self.colors['accent_secondary'])])

        # Main container
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill='both', expand=True, padx=15, pady=15)
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Store reference to main container
        self.main_container = main_container
        
        # Create notebook for multiple views
        self.main_notebook = ttk.Notebook(main_container)
        self.main_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Main app frame (existing UI)
        self.main_app_frame = ttk.Frame(self.main_notebook, style='Dark.TFrame')
        self.main_notebook.add(self.main_app_frame, text="🎭 Emotion Recognition")
        
        # Grid configuration for main app frame
        self.main_app_frame.grid_rowconfigure(1, weight=1)
        self.main_app_frame.grid_columnconfigure(0, weight=2)
        self.main_app_frame.grid_columnconfigure(1, weight=1)
        self.main_app_frame.grid_columnconfigure(2, weight=2)

        # Title and user info
        title_frame = ttk.Frame(self.main_app_frame, style='Dark.TFrame')
        title_frame.grid(row=0, column=0, columnspan=3, pady=(0, 15), sticky='ew')
        title_frame.grid_columnconfigure(1, weight=1)
        
        title_label = ttk.Label(
            title_frame,
            text="Moody",
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, sticky='w')
        
        # Profile and analytics buttons
        user_controls_frame = ttk.Frame(title_frame, style='Dark.TFrame')
        user_controls_frame.grid(row=0, column=2, sticky='e')
        
        self.profile_btn = ttk.Button(
            user_controls_frame,
            text="Switch User",
            style='Dark.TButton',
            command=self.show_profile_selector
        )
        self.profile_btn.pack(side='left', padx=(0, 5))
        
        self.analytics_btn = ttk.Button(
            user_controls_frame,
            text="Analytics",
            style='Dark.TButton',
            command=self.show_analytics_panel
        )
        self.analytics_btn.pack(side='left', padx=(0, 5))
        
        self.report_btn = ttk.Button(
            user_controls_frame,
            text="Download Report",
            style='Gesture.TButton',
            command=self.generate_report_dialog
        )
        self.report_btn.pack(side='left', padx=(0, 5))
        
        self.logout_btn = ttk.Button(
            user_controls_frame,
            text="Logout",
            style='Dark.TButton',
            command=self.logout_user
        )
        self.logout_btn.pack(side='left', padx=(0, 5))
        
        self.dashboard_btn = ttk.Button(
            user_controls_frame,
            text="Back to Dashboard",
            style='Gesture.TButton',
            command=self.back_to_dashboard
        )
        self.dashboard_btn.pack(side='left')
        
        self.current_user_label = ttk.Label(
            title_frame,
            text="Not logged in",
            style='Dark.TLabel',
            font=('Segoe UI', 9, 'italic'),
            foreground='#ff6666'
        )
        self.current_user_label.grid(row=1, column=0, columnspan=3, sticky='w', pady=(5, 0))

        # LEFT COLUMN  Camera
        left_frame = ttk.Frame(self.main_app_frame, style='Dark.TFrame')
        left_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 10))
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        camera_label = ttk.Label(
            left_frame,
            text="Live Camera Feed",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        )
        camera_label.grid(row=0, column=0, pady=(0, 10), sticky='w')

        self.camera_container = ttk.Frame(left_frame, style='Dark.TFrame')
        self.camera_container.grid(row=1, column=0, sticky='nsew')
        self.camera_container.grid_rowconfigure(0, weight=1)
        self.camera_container.grid_columnconfigure(0, weight=1)

        self.video_label = ttk.Label(self.camera_container, style='Dark.TLabel', anchor='center')
        self.video_label.grid(row=0, column=0, sticky='nsew')

        # Control frame with gesture control and speech buttons
        control_frame = ttk.Frame(left_frame, style='Dark.TFrame')
        control_frame.grid(row=2, column=0, pady=(10, 0), sticky='ew')
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)
        control_frame.grid_columnconfigure(3, weight=1)
        control_frame.grid_columnconfigure(4, weight=1)

        self.start_btn = ttk.Button(
            control_frame,
            text="Start Detection",
            style='Dark.TButton',
            command=self.start_detection
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 5), sticky='ew')

        self.stop_btn = ttk.Button(
            control_frame,
            text="Stop Detection",
            style='Dark.TButton',
            command=self.stop_detection,
            state='disabled'
        )
        self.stop_btn.grid(row=0, column=1, padx=5, sticky='ew')

        self.gesture_btn = ttk.Button(
            control_frame,
            text="Enable Gestures",
            style='Gesture.TButton',
            command=self.toggle_gesture_control,
            state='disabled'
        )
        self.gesture_btn.grid(row=0, column=2, padx=(5, 5), sticky='ew')

        # Background running button
        self.background_btn = ttk.Button(
            control_frame,
            text="Background Run",
            style='Dark.TButton',
            command=self.enable_background_mode,
            state='disabled'
        )
        self.background_btn.grid(row=0, column=3, padx=5, sticky='ew')

        # Enable Speech button
        self.speech_btn = ttk.Button(
            control_frame,
            text="Enable Speech",
            style='Gesture.TButton',
            command=self.show_voice_assistant_tab
        )
        self.speech_btn.grid(row=0, column=4, padx=(5, 0), sticky='ew')

        # Gesture status label
        self.gesture_status_label = ttk.Label(
            left_frame,
            text="Gesture Control: OFF", 
            style='Dark.TLabel',
            font=('Segoe UI', 10, 'italic')
        )
        self.gesture_status_label.grid(row=3, column=0, pady=(5, 0), sticky='w')

        # MIDDLE COLUMN Emotion Display
        middle_frame = ttk.Frame(self.main_app_frame, style='Dark.TFrame')
        middle_frame.grid(row=1, column=1, sticky='nsew', padx=10)
        middle_frame.grid_rowconfigure(2, weight=1)
        middle_frame.grid_columnconfigure(0, weight=1)

        emotion_title = ttk.Label(
            middle_frame,
            text="Current Emotion",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        )
        emotion_title.grid(row=0, column=0, pady=(0, 15), sticky='ew')

        emotion_display_frame = ttk.Frame(middle_frame, style='Dark.TFrame')
        emotion_display_frame.grid(row=1, column=0, pady=(0, 20), sticky='ew')
        self.emotion_icon_label = ttk.Label(
            emotion_display_frame,
            text="😐",
            font=('Segoe UI', 64),
            style='Dark.TLabel',
            anchor='center'
        )
        self.emotion_icon_label.pack()
        self.emotion_text_label = ttk.Label(
            emotion_display_frame,
            text="Neutral",
            style='Emotion.TLabel'
        )
        self.emotion_text_label.pack()
        self.confidence_label = ttk.Label(
            emotion_display_frame,
            text="Confidence: 0%",
            style='Dark.TLabel',
            font=('Segoe UI', 11)
        )
        self.confidence_label.pack(pady=(5, 0))

        history_label = ttk.Label(
            middle_frame,
            text="Recent Emotions",
            style='Dark.TLabel',
            font=('Segoe UI', 11, 'bold')
        )
        history_label.grid(row=2, column=0, pady=(20, 10), sticky='new')
        self.history_frame = ttk.Frame(middle_frame, style='Dark.TFrame')
        self.history_frame.grid(row=3, column=0, sticky='nsew')

        # RIGHT COLUMN Actions
        right_frame = ttk.Frame(self.main_app_frame, style='Dark.TFrame')
        right_frame.grid(row=1, column=2, sticky='nsew', padx=(10, 0))
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        actions_label = ttk.Label(
            right_frame,
            text="Suggested Actions",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        )
        actions_label.grid(row=0, column=0, pady=(0, 10), sticky='w')

        self.actions_canvas = tk.Canvas(right_frame, bg='#2a2a2a', highlightthickness=0)
        actions_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.actions_canvas.yview)
        self.actions_scrollable_frame = ttk.Frame(self.actions_canvas, style='Dark.TFrame')

        self.actions_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.actions_canvas.configure(scrollregion=self.actions_canvas.bbox("all"))
        )
        self.actions_canvas.bind('<Configure>', self._on_canvas_configure)

        self.actions_canvas.create_window((0, 0), window=self.actions_scrollable_frame, anchor="nw")
        self.actions_canvas.configure(yscrollcommand=actions_scrollbar.set)
        self.actions_canvas.grid(row=1, column=0, sticky='nsew')
        actions_scrollbar.grid(row=1, column=1, sticky='ns')

        self.update_action_suggestions()

    def _on_canvas_configure(self, event):
        items = self.actions_canvas.find_withtag("all")
        if items:
            self.actions_canvas.itemconfig(items[0], width=event.width)

    def setup_responsive_layout(self):
        self.root.bind('<Configure>', self._on_window_configure)
        self._last_width = self.root.winfo_width()
        self._last_height = self.root.winfo_height()

    def _on_window_configure(self, event):
        if event.widget == self.root:
            current_width = event.width
            current_height = event.height
            if abs(current_width - self._last_width) > 10 or abs(current_height - self._last_height) > 10:
                self._last_width = current_width
                self._last_height = current_height
                self.root.update_idletasks()

    def setup_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(LABELS_PATH)
            self.model_loaded = True
            print("Loaded:", MODEL_PATH, LABELS_PATH)
            print("Feature order ({}): {}".format(len(FEATURE_ORDER), FEATURE_ORDER))
        except Exception as e:
            self.model_loaded = False
            messagebox.showerror("Model Error", f"Failed to load model/labels: {e}")

    def setup_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("Camera and FaceMesh initialized")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.cap = None
            self.face_mesh = None

    def _canonical_label(self, label: str) -> str:
        s = (label or "").strip().lower()
        mapping = {
            "anger": "angry", "angry": "angry",
            "disgust": "disgust", "disgusted": "disgust",
            "fear": "fear", "fearful": "fear",
            "happy": "happy", "happiness": "happy",
            "neutral": "neutral",
            "sad": "sad", "sadness": "sad",
            "surprise": "surprise", "surprised": "surprise",
        }
        return mapping.get(s, "neutral")

    def _actions_for(self, canonical_label: str):
        return self.emotion_actions.get(canonical_label) or self.emotion_actions["neutral"]

    def predict_emotion_from_frame(self, frame_bgr):
        if not self.model_loaded or self.face_mesh is None:
            return "neutral", 0.0

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return "neutral", 0.0

        lm = res.multi_face_landmarks[0]
        landmarks = []
        for p in lm.landmark:
            x_px = int(round(p.x * w))
            y_px = int(round(p.y * h))
            z_px = p.z * w
            landmarks.append((x_px, y_px, z_px))

        feat_dict = compute_features(landmarks, w, h)
        x = np.array([feat_dict.get(name, 0.0) for name in FEATURE_ORDER], dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(FEATURE_ORDER):
            x = x[:, :len(FEATURE_ORDER)] if x.shape[1] > len(FEATURE_ORDER) else \
                np.pad(x, ((0, 0), (0, len(FEATURE_ORDER) - x.shape[1])), mode='constant', constant_values=0.0)

        try:
            proba = self.model.predict_proba(x)[0] if hasattr(self.model, "predict_proba") \
                else self._one_hot(self.model.predict(x), self.label_encoder.classes_.shape[0])
        except Exception:
            scores = getattr(self.model, "decision_function", lambda X: self.model.predict_proba(X))(x)
            e = np.exp(scores - np.max(scores))
            proba = (e / e.sum()).ravel()

        self._proba_window.append(proba)
        smoothed = np.mean(np.stack(self._proba_window, axis=0), axis=0)
        idx = int(np.argmax(smoothed))
        confidence = float(smoothed[idx])
        raw_label = self.label_encoder.inverse_transform([idx])[0]
        label = self._canonical_label(raw_label)
        return label, confidence

    @staticmethod
    def _one_hot(y_pred, n_classes):
        arr = np.zeros((1, n_classes), dtype=np.float32)
        arr[0, int(y_pred[0])] = 1.0
        return arr

    def detect_emotions(self):
        while self.detection_active:
            if self.cap is None:
                break

            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            emotion, confidence = self.predict_emotion_from_frame(frame)
            cv2.putText(
                frame,
                f'{emotion}: {confidence:.2f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            self.root.after(0, self.update_emotion_display, emotion, confidence)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            container_width = max(self.camera_container.winfo_width(), 400)
            container_height = max(self.camera_container.winfo_height(), 300)

            aspect = frame_rgb.shape[1] / frame_rgb.shape[0]
            if container_width / container_height > aspect:
                new_height = container_height
                new_width = int(container_height * aspect)
            else:
                new_width = container_width
                new_height = int(container_width / aspect)

            frame_pil = Image.fromarray(frame_rgb).resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.root.after(0, self.update_video_display, frame_tk)

            time.sleep(0.03)

    def update_video_display(self, frame_tk):
        self.video_label.configure(image=frame_tk)
        self.video_label.image = frame_tk

    def update_emotion_display(self, emotion, confidence):
        emotion = self._canonical_label(emotion)
        changed = (emotion != self.current_emotion)
        
        # Track emotion duration
        current_time = time.time()
        if self.last_emotion_time is not None and self.current_emotion:
            duration = current_time - self.last_emotion_time
            self.emotion_durations[self.current_emotion] += duration
        self.last_emotion_time = current_time
        
        # Track emotion changes and streaks
        if changed:
            self._track_emotion_change(emotion, confidence)
        
        self.current_emotion = emotion
        self.emotion_confidence = confidence

        self.emotion_icon_label.configure(text=self.get_emotion_icon(emotion))
        self.emotion_text_label.configure(text=emotion.capitalize())
        self.confidence_label.configure(text=f"Confidence: {confidence:.1%}")

        if changed:
            self.update_action_suggestions()
            self.update_background_popup_actions()  # keep popup in sync

        # Log emotion
        self._log_emotion(emotion, confidence)
        
        # Check for achievements
        self._check_achievements()
        
        self.add_to_history(emotion, confidence)

    def add_to_history(self, emotion, confidence):
        for w in self.history_frame.winfo_children():
            w.destroy()
        history_text = f"{self.get_emotion_icon(emotion)} {emotion.capitalize()} - {confidence:.1%}"
        ttk.Label(
            self.history_frame,
            text=history_text,
            style='Dark.TLabel',
            font=('Segoe UI', 10)
        ).pack(anchor='w', pady=3)

    def update_action_suggestions(self):
        for w in self.actions_scrollable_frame.winfo_children():
            w.destroy()
        actions = self._actions_for(self.current_emotion)
        for text, func in actions:
            ttk.Button(
                self.actions_scrollable_frame,
                text=text,
                style='Dark.TButton',
                command=func
            ).pack(fill='x', pady=4, padx=6)
        self.actions_scrollable_frame.update_idletasks()
        self.actions_canvas.configure(scrollregion=self.actions_canvas.bbox("all"))
        self.actions_canvas.yview_moveto(0.0)

    def get_emotion_icon(self, emotion):
        icons = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢',
            'surprise': '😮'
        }
        return icons.get(emotion, '😐')

    def start_detection(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded")
            return
        if self.cap is None:
            messagebox.showerror("Error", "Camera not available")
            return
        
        # Require user (should already be set via launcher)
        if self.current_user is None:
            messagebox.showwarning(
                "Login Required",
                "Please login through the Moody Launcher."
            )
            return
        
        self.detection_active = True
        self.session_start_time = datetime.now()
        self.last_emotion_time = time.time()
        self.last_analytics_check = time.time()
        self.calm_streak_start = time.time()
        self.daily_happy_spikes = 0
        
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.gesture_btn.configure(state='normal')
        self.background_btn.configure(state='normal')
        if self.popup_gesture_btn is not None and self.popup_gesture_btn.winfo_exists():
            self.popup_gesture_btn.state(["!disabled"])
        threading.Thread(target=self.detect_emotions, daemon=True).start()

    def stop_detection(self):
        self.detection_active = False
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.background_btn.configure(state='disabled')
        
        # Save emotion log
        if self.current_user:
            self._save_emotion_log()

        # Keep gesture control running if active only stop emotion detection
        # Gesture button stays enabled so user can toggle gestures independently
        self.gesture_btn.configure(state='normal')

    def toggle_gesture_control(self):
        # Only require camera to be available
        if self.cap is None:
            messagebox.showwarning("Gesture Control", "Camera is not available. Cannot enable hand gestures.")
            return

        if not self.gesture_controller.running:
            self.gesture_controller.start(self.cap)
            self.gesture_btn.configure(text="🖐️ Disable Gestures")
            self.gesture_status_label.configure(
                text="Gesture Control: ON (Show 5 fingers to activate mouse)"
            )
            if self.popup_gesture_btn is not None and self.popup_gesture_btn.winfo_exists():
                self.popup_gesture_btn.configure(text="🖐️ Disable Gestures")
                self.popup_gesture_btn.state(["!disabled"])

            messagebox.showinfo(
                "Gesture Control", 
                "Hand Gesture Control Enabled!\n\n"
                "Show 5 fingers: Toggle mouse control ON/OFF\n"
                "Index finger: Move cursor\n"
                "Thumb and Index: Click (hold for drag)\n"
                "Index and Pinky (rock sign): Right click\n"
                "Index, Middle and Ring: Scroll (move hand up/down)"
            )
        else:
            self.gesture_controller.stop()
            self.gesture_btn.configure(text="Enable Gestures")
            self.gesture_status_label.configure(text="Gesture Control: OFF")
            if self.popup_gesture_btn is not None and self.popup_gesture_btn.winfo_exists():
                self.popup_gesture_btn.configure(text="Enable Gestures")

    # BACKGROUND MODE and POPUP
    def enable_background_mode(self):
        if not self.detection_active:
            messagebox.showwarning("Background Run", "Start detection before enabling background mode.")
            return

        # Minimize main window
        try:
            if platform.system() == "Windows":
                self.root.state('iconic')
            else:
                self.root.iconify()
        except Exception:
            self.root.iconify()

        # Show ONLY the small icon on left side
        self.show_notification_icon()
        # Popup will be shown when user clicks the icon

    def show_background_popup(self):
        # If already exists, bring it back
        if self.popup_window is not None and self.popup_window.winfo_exists():
            self.popup_window.deiconify()
            self.popup_window.lift()
            return

        self.popup_window = tk.Toplevel(self.root)
        self.popup_window.title("Emotion Suggestions")
        self.popup_window.overrideredirect(True)  # borderless
        self.popup_window.attributes("-topmost", True)
        self.popup_window.configure(bg="#1a1a1a")

        # Size & position (top-right by default or last drag position)
        width, height = 320, 420
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()

        if self._popup_last_x is None or self._popup_last_y is None:
            x = ws - width - 10
            y = 10
        else:
            x = self._popup_last_x
            y = self._popup_last_y

        self.popup_window.geometry(f"{width}x{height}+{x}+{y}")
        self.popup_window.protocol("WM_DELETE_WINDOW", self.on_popup_close)

        # Make popup draggable
        self.popup_window.bind("<ButtonPress-1>", self._start_popup_drag)
        self.popup_window.bind("<B1-Motion>", self._do_popup_drag)

        popup_frame = ttk.Frame(self.popup_window, style='Dark.TFrame')
        popup_frame.pack(fill="both", expand=True, padx=8, pady=8)

        title = ttk.Label(
            popup_frame,
            text="Emotion Suggestions",
            style='Dark.TLabel',
            font=('Segoe UI', 10, 'bold')
        )
        title.pack(anchor="w", pady=(0, 4))

        # Emotion label inside popup
        self.popup_emotion_label = ttk.Label(
            popup_frame,
            text=f"{self.get_emotion_icon(self.current_emotion)} {self.current_emotion.capitalize()}",
            style='Emotion.TLabel'
        )
        self.popup_emotion_label.pack(anchor="w", pady=(0, 6))

        # Actions frame
        self.popup_actions_frame = ttk.Frame(popup_frame, style='Dark.TFrame')
        self.popup_actions_frame.pack(fill="both", expand=True)

        # Gesture toggle button (same behavior as main)
        self.popup_gesture_btn = ttk.Button(
            popup_frame,
            text="Enable Gestures",
            style='Gesture.TButton',
            command=self.toggle_gesture_control
        )
        if not self.detection_active:
            self.popup_gesture_btn.state(["disabled"])
        self.popup_gesture_btn.pack(fill="x", pady=(8, 4))

        # Restore main window button
        restore_btn = ttk.Button(
            popup_frame,
            text="Restore App",
            style='Dark.TButton',
            command=self.restore_from_background
        )
        restore_btn.pack(fill="x")

        self.update_background_popup_actions()

    def on_popup_close(self):
        if self.popup_window is not None and self.popup_window.winfo_exists():
            try:
                geo = self.popup_window.geometry() 
                parts = geo.split('+')
                if len(parts) == 3:
                    x = int(parts[1])
                    y = int(parts[2])
                    self._popup_last_x = x
                    self._popup_last_y = y
            except Exception:
                pass

            # Just hide (withdraw) icon toggles it
            self.popup_window.withdraw()

    def restore_from_background(self):
        # Bring back main window and close popup icon
        try:
            self.root.deiconify()
            if platform.system() == "Windows":
                self.root.state('normal')
        except Exception:
            pass

        # Close popup (if exists)
        if self.popup_window is not None and self.popup_window.winfo_exists():
            self.on_popup_close()

        # Hide notification icon
        self.hide_notification_icon()

    def update_background_popup_actions(self):
        # Only update if popup exists and visible object is there
        if self.popup_window is None or not self.popup_window.winfo_exists() or self.popup_actions_frame is None:
            return

        # Update emotion label
        if self.popup_emotion_label is not None:
            self.popup_emotion_label.configure(
                text=f"{self.get_emotion_icon(self.current_emotion)} {self.current_emotion.capitalize()}"
            )

        # Rebuild actions (show top few to keep it compact)
        for w in self.popup_actions_frame.winfo_children():
            w.destroy()
        actions = self._actions_for(self.current_emotion)
        for text, func in actions[:6]:  # limit to first 6 for the popup
            ttk.Button(
                self.popup_actions_frame,
                text=text,
                style='Dark.TButton',
                command=func
            ).pack(fill="x", pady=2)

    # DRAG HANDLERS FOR POPUP 
    def _start_popup_drag(self, event):
        self._popup_drag_offset_x = event.x
        self._popup_drag_offset_y = event.y

    def _do_popup_drag(self, event):
        x = event.x_root - self._popup_drag_offset_x
        y = event.y_root - self._popup_drag_offset_y
        self.popup_window.geometry(f"+{x}+{y}")
        self._popup_last_x = x
        self._popup_last_y = y

    # DRAG HANDLERS FOR NOTIFICATION ICON 
    def _start_notif_drag(self, event):
        self._notif_drag_offset_x = event.x
        self._notif_drag_offset_y = event.y

    def _do_notif_drag(self, event):
        x = event.x_root - self._notif_drag_offset_x
        y = event.y_root - self._notif_drag_offset_y
        self.notification_window.geometry(f"+{x}+{y}")

    def _end_notif_drag(self, event):
        # could save position if needed
        pass

    # NOTIFICATION ICON WINDOW 
    def show_notification_icon(self):
        # If exists, just bring to front
        if self.notification_window is not None and self.notification_window.winfo_exists():
            self.notification_window.deiconify()
            self.notification_window.lift()
            return

        self.notification_window = tk.Toplevel(self.root)
        self.notification_window.overrideredirect(True)
        self.notification_window.attributes("-topmost", True)
        self.notification_window.configure(bg="#1a1a1a")

        # Small square at left side of screen
        size = 60
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = 10
        y = hs // 2 - size // 2
        self.notification_window.geometry(f"{size}x{size}+{x}+{y}")

        # Make notification icon draggable
        self.notification_window.bind("<ButtonPress-1>", self._start_notif_drag)
        self.notification_window.bind("<B1-Motion>", self._do_notif_drag)
        self.notification_window.bind("<ButtonRelease-1>", self._end_notif_drag)

        # Create canvas for custom icon
        canvas = tk.Canvas(
            self.notification_window,
            width=size,
            height=size,
            bg="#1a1a1a",
            highlightthickness=0,
            bd=0
        )
        canvas.pack(fill="both", expand=True)

        # Draw colorful circular button with gradient effect
        center = size // 2
        radius = 24
        
        # Outer glow
        canvas.create_oval(
            center - radius - 2, center - radius - 2,
            center + radius + 2, center + radius + 2,
            fill="#6a4c93", outline=""
        )
        
        # Main circle with gradient colors
        canvas.create_oval(
            center - radius, center - radius,
            center + radius, center + radius,
            fill="#8b5cf6", outline="#a78bfa", width=2
        )
        
        # Emoji in center
        canvas.create_text(
            center, center,
            text="🎭",
            font=("Segoe UI", 22, "bold"),
            fill="white"
        )
        
        # Red notification badge (top right)
        badge_x = size - 12
        badge_y = 12
        badge_radius = 10
        
        # Badge circle
        canvas.create_oval(
            badge_x - badge_radius, badge_y - badge_radius,
            badge_x + badge_radius, badge_y + badge_radius,
            fill="#ef4444", outline="#dc2626", width=1
        )
        
        # Badge number
        canvas.create_text(
            badge_x, badge_y,
            text="1",
            font=("Segoe UI", 10, "bold"),
            fill="white"
        )
        
        # Bind click to canvas
        canvas.bind("<Button-1>", lambda e: self.toggle_popup_from_notification())
        
        # Store canvas reference
        self.notification_canvas = canvas

    def hide_notification_icon(self):
        if self.notification_window is not None and self.notification_window.winfo_exists():
            self.notification_window.destroy()
        self.notification_window = None
        self.notification_canvas = None

    def toggle_popup_from_notification(self):
        # If popup doesn't exist or was destroyed create it
        if self.popup_window is None or not self.popup_window.winfo_exists():
            self.show_background_popup()
            return

        # If popup is hidden show it if not hide it
        try:
            if not self.popup_window.winfo_viewable():
                self.popup_window.deiconify()
                self.popup_window.lift()
            else:
                self.on_popup_close()  # withdraw remember position
        except Exception:
            self.show_background_popup()

    # PROFILE MANAGEMENT 
    def _require_launcher_login(self):
        """If no MOODY_USER env var, redirect to launcher for login."""
        messagebox.showinfo(
            "Login Required",
            "Please login through the Moody Launcher first."
        )
        self.back_to_dashboard(force_close=True)

    def force_profile_selection(self):
        """Kept for compatibility – redirects to dashboard."""
        if self.current_user is None:
            self._require_launcher_login()
    
    def show_profile_selector(self):
        """Login is now handled by the launcher. Redirect to dashboard."""
        response = messagebox.askyesno(
            "Switch User",
            "To switch users you will be redirected to the Moody Launcher.\n\nContinue?"
        )
        if response:
            self.back_to_dashboard()
    
    def _load_profiles(self):
        """Load list of existing profiles with passwords"""
        profiles_file = os.path.join(self.profiles_dir, "profiles.json")
        if os.path.exists(profiles_file):
            with open(profiles_file, 'r') as f:
                data = json.load(f)
                # Return dictionary format {username password hash}
                return data.get('profiles', {})
        return {}
    
    def _save_profiles_list(self, profiles):
        """Save profiles dictionary with password hashes"""
        profiles_file = os.path.join(self.profiles_dir, "profiles.json")
        with open(profiles_file, 'w') as f:
            json.dump({'profiles': profiles}, f, indent=2)
    
    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_login(self, username, password):
        """Verify username and password"""
        profiles = self._load_profiles()
        
        # Guest has no password
        if username == "Guest":
            return True
        
        if username not in profiles:
            return False
        
        password_hash = self._hash_password(password)
        return profiles[username] == password_hash
    
    def _username_exists(self, username):
        """Check if username already exists"""
        profiles = self._load_profiles()
        return username in profiles
    
    def _create_account(self, username, password):
        """Create new user account with password"""
        try:
            profiles = self._load_profiles()
            password_hash = self._hash_password(password)
            profiles[username] = password_hash
            self._save_profiles_list(profiles)
            return True
        except Exception as e:
            print(f"Error creating account: {e}")
            return False
    
    def _load_user_profile(self, username):
        """Load user profile and settings"""
        self.current_user = username
        self.current_user_label.configure(
            text=f"User: {username}",
            foreground='#00ff88'
        )
        
        # Load user settings
        settings_file = os.path.join(self.profiles_dir, f"{username}_settings.json")
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                self.user_settings = json.load(f)
        else:
            self.user_settings = {
                'favorite_actions': [],
                'preferences': {}
            }
            self._save_user_settings()
        
        # Load emotion log
        self._load_emotion_log()
    
    def _save_user_settings(self):
        """Save current user settings"""
        if self.current_user:
            settings_file = os.path.join(self.profiles_dir, f"{self.current_user}_settings.json")
            with open(settings_file, 'w') as f:
                json.dump(self.user_settings, f, indent=2)
    
    def logout_user(self):
        """Logout current user and return to launcher dashboard"""
        if self.detection_active:
            response = messagebox.askyesno(
                "Logout",
                "Detection is active. Stop detection and logout?"
            )
            if response:
                self.stop_detection()
            else:
                return
        
        # Save current data
        if self.current_user:
            self._save_emotion_log()
            self._save_user_settings()
        
        # Return to dashboard for re-login
        self.back_to_dashboard()
    
    def back_to_dashboard(self, force_close=False):
        """Return to the main dashboard launcher"""
        if not force_close and self.detection_active:
            response = messagebox.askyesno(
                "Back to Dashboard",
                "Detection is active. Stop detection and return to dashboard?"
            )
            if response:
                self.stop_detection()
            else:
                return
        
        # Save current data if user is logged in
        if self.current_user and not force_close:
            self._save_emotion_log()
            self._save_user_settings()
        
        # Launch dashboard
        try:
            import sys
            base_dir = Path(__file__).resolve().parent.parent
            launcher_script = base_dir / "launcher" / "common_launcher.py"
            venv_python = base_dir / ".venv" / "Scripts" / "python.exe"
            python_exec = str(venv_python if venv_python.exists() else Path(sys.executable))
            
            subprocess.Popen(
                [python_exec, str(launcher_script)],
                cwd=str(launcher_script.parent)
            )
            
            # Close this window
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch dashboard: {e}")
    
    def _load_emotion_log(self):
        """Load emotion log for current user"""
        if self.current_user:
            log_file = os.path.join(self.profiles_dir, f"{self.current_user}_emotions.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    self.emotion_log = json.load(f)
            else:
                self.emotion_log = []
    
    def _log_emotion(self, emotion, confidence):
        """Log emotion with timestamp"""
        if self.current_user and self.detection_active:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion,
                'confidence': confidence,
                'session_id': id(self.session_start_time)
            }
            self.emotion_log.append(log_entry)
            
            # Track using analytics engine
            self.analytics.track_hourly_emotion(emotion)
            self.analytics.add_stress_indicator(emotion, confidence)
            
            # Save every 10 entries to avoid too many writes
            if len(self.emotion_log) % 10 == 0:
                self._save_emotion_log()
    
    def _save_emotion_log(self):
        """Save emotion log to file"""
        if self.current_user:
            log_file = os.path.join(self.profiles_dir, f"{self.current_user}_emotions.json")
            with open(log_file, 'w') as f:
                json.dump(self.emotion_log, f, indent=2)
    
    def _track_emotion_change(self, new_emotion, confidence):
        """Track emotion changes for streaks and spikes"""
        # Track transitions using analytics engine
        if self.current_emotion and self.current_emotion != new_emotion:
            self.analytics.track_emotion_transition(self.current_emotion, new_emotion)
        
        # Track happy spikes
        if new_emotion == 'happy' and confidence > 0.7:
            self.daily_happy_spikes += 1
        
        # Track emotion streaks
        if self.emotion_streak_emotion != new_emotion:
            self.emotion_streak_emotion = new_emotion
            self.emotion_streak_start = time.time()
        
        # Reset calm streak if angry,fear detected
        if new_emotion in ['angry', 'fear'] and confidence > 0.7:
            self.calm_streak_start = time.time()
    
    def _check_achievements(self):
        """Check and display achievement notifications"""
        if not self.detection_active or self.last_analytics_check is None:
            return
        
        current_time = time.time()
        # Check every 5 minutes
        if current_time - self.last_analytics_check < 300:
            return
        
        self.last_analytics_check = current_time
        
        # Check calm streak (2 hours)
        calm_duration = current_time - self.calm_streak_start
        if calm_duration >= 7200:  # 2 hours
            self._show_achievement("🧘 Calm Mastery!", "You stayed calm for 2 hours! Keep it up!")
            self.calm_streak_start = current_time
        
        # Check happy spikes
        if self.daily_happy_spikes >= 3:
            self._show_achievement("Joy Spreader!", f"You had {self.daily_happy_spikes} happy moments today!")
    
    def _show_achievement(self, title, message):
        """Show achievement notification"""
        # Create a toast like notification
        toast = tk.Toplevel(self.root)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.configure(bg="#2a2a2a")
        
        frame = tk.Frame(toast, bg="#2a2a2a", bd=2, relief='raised')
        frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        tk.Label(
            frame,
            text=title,
            bg="#2a2a2a",
            fg="#00ff88",
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=(10, 5), padx=15)
        
        tk.Label(
            frame,
            text=message,
            bg="#2a2a2a",
            fg="#ffffff",
            font=('Segoe UI', 10)
        ).pack(pady=(0, 10), padx=15)
        
        # Position at top right
        toast.update_idletasks()
        width = toast.winfo_width()
        height = toast.winfo_height()
        x = self.root.winfo_screenwidth() - width - 20
        y = 80
        toast.geometry(f"+{x}+{y}")
        
        # Auto close after 4 seconds
        toast.after(4000, toast.destroy)
    
    # ANALYTICS PANEL
    def show_analytics_panel(self):
        """Show analytics panel as a tab within the main application"""
        if not self.current_user:
            messagebox.showwarning("Analytics", "Please login to view analytics!")
            return
        
        # If analytics tab already exists switch to it
        if self.analytics_tab is not None:
            try:
                # Find the tab index
                for i in range(self.main_notebook.index('end')):
                    if self.main_notebook.tab(i, 'text').startswith('📈'):
                        self.main_notebook.select(i)
                        # Refresh the data
                        self._refresh_analytics_tab()
                        return
            except:
                self.analytics_tab = None
        
        # Create new analytics tab
        self.analytics_tab = ttk.Frame(self.main_notebook, style='Dark.TFrame')
        self.main_notebook.add(self.analytics_tab, text=f"📈 Analytics - {self.current_user}")
        
        # Main container with padding
        main_frame = ttk.Frame(self.analytics_tab, style='Dark.TFrame', padding=15)
        main_frame.pack(fill='both', expand=True)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title with close button
        header_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 15))
        header_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(
            header_frame,
            text=f"Mood Analytics - {self.current_user}",
            style='Title.TLabel'
        ).grid(row=0, column=0, sticky='w')
        
        button_frame = ttk.Frame(header_frame, style='Dark.TFrame')
        button_frame.grid(row=0, column=1, sticky='e')
        
        ttk.Button(
            button_frame,
            text="Refresh",
            style='Dark.TButton',
            command=self._refresh_analytics_tab
        ).pack(side='left', padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="Close Tab",
            style='Dark.TButton',
            command=self._close_analytics_tab
        ).pack(side='left')
        
        # Create notebook for sub tabs
        self.analytics_notebook = ttk.Notebook(main_frame)
        self.analytics_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Tab 1 Today's Stats
        self.today_frame = ttk.Frame(self.analytics_notebook, style='Dark.TFrame', padding=15)
        self.analytics_notebook.add(self.today_frame, text="Today")
        
        # Tab 2: Weekly Stats
        self.week_frame = ttk.Frame(self.analytics_notebook, style='Dark.TFrame', padding=15)
        self.analytics_notebook.add(self.week_frame, text="This Week")
        
        # Tab 3: Streaks and Goals
        self.goals_frame = ttk.Frame(self.analytics_notebook, style='Dark.TFrame', padding=15)
        self.analytics_notebook.add(self.goals_frame, text="Achievements")
        
        # Tab 4: Advanced Analytics
        self.advanced_frame = ttk.Frame(self.analytics_notebook, style='Dark.TFrame', padding=15)
        self.analytics_notebook.add(self.advanced_frame, text="Advanced Analytics")
        
        # Tab 5: Patterns and Insights
        self.patterns_frame = ttk.Frame(self.analytics_notebook, style='Dark.TFrame', padding=15)
        self.analytics_notebook.add(self.patterns_frame, text="Patterns & Insights")
        
        # Populate data
        self._populate_today_stats(self.today_frame)
        self._populate_week_stats(self.week_frame)
        self._populate_goals_stats(self.goals_frame)
        self._populate_advanced_analytics(self.advanced_frame)
        self._populate_patterns_insights(self.patterns_frame)
        
        # Switch to analytics tab
        self.main_notebook.select(self.analytics_tab)
    
    def _close_analytics_tab(self):
        """Close the analytics tab"""
        if self.analytics_tab is not None:
            try:
                self.main_notebook.forget(self.analytics_tab)
                self.analytics_tab = None
                # Switch back to main tab
                self.main_notebook.select(0)
            except:
                pass

    
    #  VOICE ASSISTANT TAB
    
    def show_voice_assistant_tab(self):
        """Show voice assistant panel as a new tab in the main notebook."""
        # If voice tab already exists switch to it
        if self.voice_tab is not None:
            try:
                for i in range(self.main_notebook.index('end')):
                    if self.main_notebook.tab(i, 'text').startswith('🎤'):
                        self.main_notebook.select(i)
                        return
            except:
                self.voice_tab = None

        # Create new voice tab
        self.voice_tab = ttk.Frame(self.main_notebook, style='Dark.TFrame')
        self.main_notebook.add(self.voice_tab, text="🎤 Voice Assistant")

        self.voice_tab.grid_rowconfigure(1, weight=1)
        self.voice_tab.grid_columnconfigure(0, weight=1)

        # Header bar
        header = ttk.Frame(self.voice_tab, style='Dark.TFrame')
        header.grid(row=0, column=0, sticky='ew', padx=15, pady=(15, 5))
        header.grid_columnconfigure(1, weight=1)

        ttk.Label(
            header, text="Moody Voice Assistant",
            style='Title.TLabel'
        ).grid(row=0, column=0, sticky='w')

        # Close tab button
        close_btn = ttk.Button(
            header, text="✖ Close Tab",
            style='Dark.TButton',
            command=self._close_voice_tab
        )
        close_btn.grid(row=0, column=2, sticky='e', padx=(10, 0))

        # Main content two columns 
        content = ttk.Frame(self.voice_tab, style='Dark.TFrame')
        content.grid(row=1, column=0, sticky='nsew', padx=15, pady=10)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)   # Log area
        content.grid_columnconfigure(1, weight=0)   # Control panel

        # Chat and Log area 
        log_frame = ttk.Frame(content, style='Dark.TFrame')
        log_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(
            log_frame, text="Activity Log",
            style='Dark.TLabel',
            font=('Segoe UI', 11, 'bold')
        ).grid(row=0, column=0, sticky='w', pady=(0, 5))

        # Scrollable text log
        log_container = ttk.Frame(log_frame, style='Dark.TFrame')
        log_container.grid(row=1, column=0, sticky='nsew')
        log_container.grid_rowconfigure(0, weight=1)
        log_container.grid_columnconfigure(0, weight=1)

        self.voice_log_text = tk.Text(
            log_container,
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            font=('Consolas', 10),
            wrap='word',
            state='disabled',
            relief='flat',
            padx=10, pady=10,
            insertbackground=self.colors['text_primary'],
        )
        self.voice_log_text.grid(row=0, column=0, sticky='nsew')

        log_scrollbar = ttk.Scrollbar(log_container, orient='vertical',
                                       command=self.voice_log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky='ns')
        self.voice_log_text.configure(yscrollcommand=log_scrollbar.set)

        # Configure text tags for coloring
        self.voice_log_text.tag_configure('user', foreground='#60A5FA')
        self.voice_log_text.tag_configure('assistant', foreground='#34D399')
        self.voice_log_text.tag_configure('system', foreground='#FBBF24')
        self.voice_log_text.tag_configure('error', foreground='#F87171')
        self.voice_log_text.tag_configure('timestamp', foreground='#6B7280')

        # RIGHT Control panel
        ctrl_panel = ttk.Frame(content, style='Dark.TFrame')
        ctrl_panel.grid(row=0, column=1, sticky='nsew')
        ctrl_panel.grid_columnconfigure(0, weight=1)

        # Status display
        status_frame = ttk.Frame(ctrl_panel, style='Dark.TFrame')
        status_frame.grid(row=0, column=0, sticky='ew', pady=(0, 15))

        ttk.Label(
            status_frame, text="Status",
            style='Dark.TLabel',
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor='w')

        self.voice_status_label = ttk.Label(
            status_frame,
            text="⏹ Not started",
            style='Dark.TLabel',
            font=('Segoe UI', 10),
            wraplength=250,
        )
        self.voice_status_label.pack(anchor='w', pady=(5, 0))

        # Wake word indicator
        self.voice_wake_indicator = ttk.Label(
            status_frame,
            text="Waiting for wake word",
            style='Dark.TLabel',
            font=('Segoe UI', 9, 'italic'),
        )
        self.voice_wake_indicator.pack(anchor='w', pady=(5, 0))

        # Control buttons
        btn_frame = ttk.Frame(ctrl_panel, style='Dark.TFrame')
        btn_frame.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self.voice_start_btn = ttk.Button(
            btn_frame,
            text="Start Listening",
            style='Gesture.TButton',
            command=self._start_voice_assistant
        )
        self.voice_start_btn.grid(row=0, column=0, padx=(0, 5), sticky='ew')

        self.voice_stop_btn = ttk.Button(
            btn_frame,
            text="Stop",
            style='Dark.TButton',
            command=self._stop_voice_assistant,
            state='disabled'
        )
        self.voice_stop_btn.grid(row=0, column=1, padx=(5, 0), sticky='ew')

        # Background mode checkbox
        self.voice_bg_var = tk.BooleanVar(value=False)
        bg_check = tk.Checkbutton(
            btn_frame,
            text="Background Mode",
            variable=self.voice_bg_var,
            command=self._toggle_voice_background,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['bg_secondary'],
            activebackground=self.colors['bg_primary'],
            activeforeground=self.colors['text_primary'],
            font=('Segoe UI', 9),
        )
        bg_check.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky='w')

        # Wake word info
        info_frame = ttk.Frame(ctrl_panel, style='Dark.TFrame')
        info_frame.grid(row=2, column=0, sticky='ew', pady=(0, 15))

        ttk.Label(
            info_frame, text="🔑 Wake Word",
            style='Dark.TLabel',
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor='w')

        ttk.Label(
            info_frame,
            text='Say  "Hey Moody"  or  "Moody"\nto activate voice commands.',
            style='Dark.TLabel',
            font=('Segoe UI', 9),
            wraplength=250,
            justify='left',
        ).pack(anchor='w', pady=(5, 0))

        # Command reference
        cmd_frame = ttk.Frame(ctrl_panel, style='Dark.TFrame')
        cmd_frame.grid(row=3, column=0, sticky='nsew', pady=(0, 0))
        ctrl_panel.grid_rowconfigure(3, weight=1)

        ttk.Label(
            cmd_frame, text="Quick Command Reference",
            style='Dark.TLabel',
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor='w')

        commands_text = (
            "🖥️ System Apps:\n"
            "  • Open Notepad / Calculator / Explorer\n"
            "  • Task Manager / Settings / CMD / PowerShell\n"
            "  • Word / Excel / PowerPoint / Camera\n\n"
            "🌐 Web & Search:\n"
            "  • Open YouTube / Google / Gmail / GitHub\n"
            "  • Open Reddit / Twitter / Netflix / Spotify\n"
            "  • Search for [topic] / Search YouTube [topic]\n\n"
            "🔊 Volume & Media:\n"
            "  • Volume up / down / mute / max / min\n"
            "  • Set volume to [0-100] / Play / Pause\n\n"
            "🖐️ Gesture Mouse:\n"
            "  • Enable mouse / Enable gesture mouse\n"
            "  • Disable mouse / Disable gesture\n\n"
            "🖱️ Mouse & Click:\n"
            "  • Click / Double click / Right click\n"
            "  • Move mouse [direction]\n\n"
            "⌨️ Keyboard & Typing:\n"
            "  • Copy / Paste / Cut / Undo / Redo\n"
            "  • Type [text] / Save / Save as / Print\n"
            "  • New tab / Close tab / Reopen tab\n\n"
            "🪟 Window Management:\n"
            "  • Minimize / Maximize / Close / Restore\n"
            "  • Snap left / right / Show desktop\n"
            "  • Screenshot / Switch window\n\n"
            "📜 Scroll:\n"
            "  • Scroll up / down / left / right\n"
            "  • Page up / Page down / Top / Bottom\n\n"
            "🔧 System Control:\n"
            "  • Lock screen / Brightness up-down\n"
            "  • Wi-Fi / Bluetooth / Sound settings\n"
            "  • Battery status / Date / Time\n\n"
            "💬 Fun & Utility:\n"
            "  • Tell me a joke / Motivate me\n"
            "  • Set timer [N] seconds\n\n"
            "🛑 Control:\n"
            "  • Stop listening / Go to sleep / Help\n"
            "  \n"
            "💡 Speak naturally! e.g.:\n"
            "  'Hey Moody, can you open YouTube?'\n"
            "  'Hey Moody, please enable mouse'"
        )

        cmd_label = ttk.Label(
            cmd_frame,
            text=commands_text,
            style='Dark.TLabel',
            font=('Segoe UI', 8),
            wraplength=250,
            justify='left',
        )
        cmd_label.pack(anchor='w', pady=(5, 0))

        # Switch to the new tab
        self.main_notebook.select(self.voice_tab)

        # Add welcome message
        self._voice_log("Welcome to Moody Voice Assistant!", "system")
        self._voice_log("Click 'Start Listening' and then say 'Hey Moody' to begin.", "system")

    def _close_voice_tab(self):
        """Close the voice assistant tab and stop it if running."""
        if self.voice_assistant and self.voice_assistant.running:
            self.voice_assistant.stop()
        if self.voice_tab is not None:
            try:
                self.main_notebook.forget(self.voice_tab)
                self.voice_tab = None
                self.voice_log_text = None
                self.voice_status_label = None
                self.voice_wake_indicator = None
                self.main_notebook.select(0)
            except:
                pass

    def _voice_log(self, text, tag="system"):
        """Add a timestamped message to the voice log."""
        if self.voice_log_text is None:
            return
        try:
            def _do():
                self.voice_log_text.configure(state='normal')
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.voice_log_text.insert('end', f"[{timestamp}] ", 'timestamp')
                self.voice_log_text.insert('end', f"{text}\n", tag)
                self.voice_log_text.see('end')
                self.voice_log_text.configure(state='disabled')
            self.root.after(0, _do)
        except Exception:
            pass

    def _voice_status(self, status):
        """Update the voice status label."""
        if self.voice_status_label is None:
            return
        try:
            def _do():
                self.voice_status_label.configure(text=status)
                # Update wake indicator
                if self.voice_wake_indicator:
                    if "Awake" in status or "command" in status.lower():
                        self.voice_wake_indicator.configure(text="🟢 Awake — say a command!")
                    elif "Sleeping" in status or "Waiting" in status:
                        self.voice_wake_indicator.configure(text="💤 Waiting for 'Hey Moody'")
                    elif "stopped" in status.lower():
                        self.voice_wake_indicator.configure(text="⏹ Not active")
            self.root.after(0, _do)
        except Exception:
            pass

    def _voice_on_wake(self):
        """Called when wake word is detected."""
        try:
            def _do():
                if self.voice_wake_indicator:
                    self.voice_wake_indicator.configure(text="Awake — say a command!")
            self.root.after(0, _do)
        except Exception:
            pass

    def _voice_gesture_toggle(self, action):
        """Toggle gesture mouse control from voice command."""
        try:
            if action == "enable":
                if not self.detection_active or self.cap is None:
                    self._voice_log("Start detection first before enabling gesture mouse.", "error")
                    return
                if not self.gesture_controller.running:
                    self.gesture_controller.start(self.cap)
                    self.gesture_btn.configure(text="\U0001f590\ufe0f Disable Gestures")
                    self.gesture_status_label.configure(
                        text="Gesture Control: ON (Show 5 fingers to activate mouse)"
                    )
                    if self.popup_gesture_btn is not None and self.popup_gesture_btn.winfo_exists():
                        self.popup_gesture_btn.configure(text="\U0001f590\ufe0f Disable Gestures")
                        self.popup_gesture_btn.state(["!disabled"])
                    self._voice_log("Gesture mouse control ENABLED.", "system")
                else:
                    self._voice_log("Gesture mouse is already enabled.", "system")
            elif action == "disable":
                if self.gesture_controller.running:
                    self.gesture_controller.stop()
                    self.gesture_btn.configure(text="\U0001f590\ufe0f Enable Gestures")
                    self.gesture_status_label.configure(text="Gesture Control: OFF")
                    if self.popup_gesture_btn is not None and self.popup_gesture_btn.winfo_exists():
                        self.popup_gesture_btn.configure(text="\U0001f590\ufe0f Enable Gestures")
                    self._voice_log("Gesture mouse control DISABLED.", "system")
                else:
                    self._voice_log("Gesture mouse is already disabled.", "system")
        except Exception as e:
            self._voice_log(f"Gesture toggle error: {e}", "error")

    def _start_voice_assistant(self):
        """Start the voice assistant."""
        if self.voice_assistant and self.voice_assistant.running:
            self._voice_log("Voice assistant is already running.", "system")
            return

        self.voice_assistant = MoodyVoiceAssistant(
            on_status_change=self._voice_status,
            on_log=self._voice_log,
            on_wake=self._voice_on_wake,
            gesture_toggle_callback=self._voice_gesture_toggle,
        )

        success = self.voice_assistant.start()
        if success:
            self.voice_start_btn.configure(state='disabled')
            self.voice_stop_btn.configure(state='normal')
            self.speech_btn.configure(text="Speech ON", style='Gesture.TButton')
        else:
            self._voice_log("Failed to start voice assistant. Check microphone.", "error")

    def _stop_voice_assistant(self):
        """Stop the voice assistant."""
        if self.voice_assistant:
            self.voice_assistant.stop()
        self.voice_start_btn.configure(state='normal')
        self.voice_stop_btn.configure(state='disabled')
        self.speech_btn.configure(text="Enable Speech", style='Gesture.TButton')

    def _toggle_voice_background(self):
        """Toggle voice assistant background mode."""
        if self.voice_assistant and self.voice_assistant.running:
            enabled = self.voice_bg_var.get()
            self.voice_assistant.toggle_background(enabled)
            if enabled:
                self._voice_log("Background mode ON — assistant keeps listening even when you switch tabs.", "system")
            else:
                self._voice_log("Background mode OFF.", "system")
        elif self.voice_bg_var and self.voice_bg_var.get():
            self.voice_bg_var.set(False)
            self._voice_log("Start the voice assistant first before enabling background mode.", "system")
    
    def _refresh_analytics_tab(self):
        """Refresh analytics data"""
        if self.analytics_tab is not None and hasattr(self, 'today_frame'):
            # Clear existing widgets
            for widget in self.today_frame.winfo_children():
                widget.destroy()
            for widget in self.week_frame.winfo_children():
                widget.destroy()
            for widget in self.goals_frame.winfo_children():
                widget.destroy()
            
            # Repopulate
            self._populate_today_stats(self.today_frame)
            self._populate_week_stats(self.week_frame)
            self._populate_goals_stats(self.goals_frame)
    
    def _populate_today_stats(self, parent):
        """Populate today's statistics"""
        # Filter today's emotions
        today = datetime.now().date()
        today_emotions = [
            entry for entry in self.emotion_log
            if datetime.fromisoformat(entry['timestamp']).date() == today
        ]
        
        if not today_emotions:
            ttk.Label(
                parent,
                text="No data recorded today yet.",
                style='Dark.TLabel',
                font=('Segoe UI', 11)
            ).pack(pady=20)
            return
        
        # Count emotions
        emotion_counts = Counter([e['emotion'] for e in today_emotions])
        
        ttk.Label(
            parent,
            text="🎯 Emotion Distribution Today",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=(0, 15))
        
        # Display as bars
        for emotion in self.emotion_labels:
            count = emotion_counts.get(emotion, 0)
            percentage = (count / len(today_emotions) * 100) if today_emotions else 0
            
            row_frame = ttk.Frame(parent, style='Dark.TFrame')
            row_frame.pack(fill='x', pady=5)
            
            label_text = f"{self.get_emotion_icon(emotion)} {emotion.capitalize()}"
            ttk.Label(
                row_frame,
                text=label_text,
                style='Dark.TLabel',
                width=12
            ).pack(side='left')
            
            # Progress bar
            progress = ttk.Progressbar(
                row_frame,
                length=300,
                mode='determinate',
                value=percentage
            )
            progress.pack(side='left', padx=10)
            
            ttk.Label(
                row_frame,
                text=f"{percentage:.1f}%",
                style='Dark.TLabel'
            ).pack(side='left')
        
        # Session duration
        if self.session_start_time:
            session_duration = datetime.now() - self.session_start_time
            minutes = int(session_duration.total_seconds() / 60)
            
            ttk.Label(
                parent,
                text=f"\n Current Session: {minutes} minutes",
                style='Dark.TLabel',
                font=('Segoe UI', 11)
            ).pack(pady=10)
        
        # Most common emotion
        if emotion_counts:
            most_common = emotion_counts.most_common(1)[0]
            ttk.Label(
                parent,
                text=f"\n Most Common: {self.get_emotion_icon(most_common[0])} {most_common[0].capitalize()}",
                style='Emotion.TLabel'
            ).pack(pady=5)
    
    def _populate_week_stats(self, parent):
        """Populate this week's statistics"""
        # Filter week's emotions
        week_ago = datetime.now() - timedelta(days=7)
        week_emotions = [
            entry for entry in self.emotion_log
            if datetime.fromisoformat(entry['timestamp']) >= week_ago
        ]
        
        if not week_emotions:
            ttk.Label(
                parent,
                text="No data recorded this week yet.",
                style='Dark.TLabel',
                font=('Segoe UI', 11)
            ).pack(pady=20)
            return
        
        ttk.Label(
            parent,
            text="Weekly Summary",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=(0, 15))
        
        # Total entries
        ttk.Label(
            parent,
            text=f"Total Mood Checks: {len(week_emotions)}",
            style='Dark.TLabel',
            font=('Segoe UI', 11)
        ).pack(pady=5)
        
        # Emotion counts
        emotion_counts = Counter([e['emotion'] for e in week_emotions])
        
        ttk.Label(
            parent,
            text="\nTop Emotions This Week",
            style='Dark.TLabel',
            font=('Segoe UI', 11, 'bold')
        ).pack(pady=10)
        
        for emotion, count in emotion_counts.most_common(5):
            percentage = (count / len(week_emotions) * 100)
            text = f"{self.get_emotion_icon(emotion)} {emotion.capitalize()}: {percentage:.1f}% ({count} times)"
            ttk.Label(
                parent,
                text=text,
                style='Dark.TLabel',
                font=('Segoe UI', 10)
            ).pack(pady=3, anchor='w', padx=20)
        
        # Calculate longest streak
        self._display_longest_streak(parent, week_emotions)
    
    def _display_longest_streak(self, parent, emotions):
        """Calculate and display longest emotion streak"""
        if not emotions:
            return
        
        max_streak = 0
        max_emotion = None
        current_streak = 1
        current_emotion = emotions[0]['emotion']
        
        for i in range(1, len(emotions)):
            if emotions[i]['emotion'] == current_emotion:
                current_streak += 1
            else:
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_emotion = current_emotion
                current_emotion = emotions[i]['emotion']
                current_streak = 1
        
        if current_streak > max_streak:
            max_streak = current_streak
            max_emotion = current_emotion
        
        ttk.Label(
            parent,
            text=f"\n Longest Streak: {self.get_emotion_icon(max_emotion)} {max_emotion.capitalize()} ({max_streak} consecutive)",
            style='Emotion.TLabel'
        ).pack(pady=15)
    
    def _populate_goals_stats(self, parent):
        """Populate achievements and goals"""
        ttk.Label(
            parent,
            text="🏆 Achievements & Goals",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=(0, 20))
        
        # Calm streak
        if self.calm_streak_start:
            calm_duration = time.time() - self.calm_streak_start
            calm_minutes = int(calm_duration / 60)
            calm_hours = calm_minutes / 60
            
            achievement_frame = ttk.Frame(parent, style='Dark.TFrame')
            achievement_frame.pack(fill='x', pady=10, padx=20)
            
            ttk.Label(
                achievement_frame,
                text="🧘 Calm Streak",
                style='Dark.TLabel',
                font=('Segoe UI', 11, 'bold')
            ).pack(anchor='w')
            
            ttk.Label(
                achievement_frame,
                text=f"Current: {calm_hours:.1f} hours ({calm_minutes} minutes)",
                style='Dark.TLabel',
                font=('Segoe UI', 10)
            ).pack(anchor='w', padx=20)
            
            if calm_hours >= 2:
                ttk.Label(
                    achievement_frame,
                    text=" Goal Achieved: 2+ hours calm!",
                    style='Emotion.TLabel'
                ).pack(anchor='w', padx=20, pady=5)
        
        # Happy spikes
        achievement_frame2 = ttk.Frame(parent, style='Dark.TFrame')
        achievement_frame2.pack(fill='x', pady=10, padx=20)
        
        ttk.Label(
            achievement_frame2,
            text=" Happy Moments Today",
            style='Dark.TLabel',
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor='w')
        
        ttk.Label(
            achievement_frame2,
            text=f"Count: {self.daily_happy_spikes}",
            style='Dark.TLabel',
            font=('Segoe UI', 10)
        ).pack(anchor='w', padx=20)
        
        if self.daily_happy_spikes >= 3:
            ttk.Label(
                achievement_frame2,
                text=" Goal Achieved: 3+ happy moments!",
                style='Emotion.TLabel'
            ).pack(anchor='w', padx=20, pady=5)
        
        # Emotion balance
        if self.emotion_log:
            today = datetime.now().date()
            today_emotions = [
                entry for entry in self.emotion_log
                if datetime.fromisoformat(entry['timestamp']).date() == today
            ]
            
            if today_emotions:
                emotion_counts = Counter([e['emotion'] for e in today_emotions])
                positive = emotion_counts.get('happy', 0) + emotion_counts.get('surprise', 0)
                negative = emotion_counts.get('sad', 0) + emotion_counts.get('angry', 0) + emotion_counts.get('fear', 0)
                
                achievement_frame3 = ttk.Frame(parent, style='Dark.TFrame')
                achievement_frame3.pack(fill='x', pady=10, padx=20)
                
                ttk.Label(
                    achievement_frame3,
                    text=" Emotional Balance",
                    style='Dark.TLabel',
                    font=('Segoe UI', 11, 'bold')
                ).pack(anchor='w')
                
                if positive > negative:
                    ttk.Label(
                        achievement_frame3,
                        text=" More positive emotions today! Keep it up!",
                        style='Emotion.TLabel'
                    ).pack(anchor='w', padx=20, pady=5)
                else:
                    ttk.Label(
                        achievement_frame3,
                        text=" Remember to take care of yourself",
                        style='Dark.TLabel',
                        font=('Segoe UI', 10)
                    ).pack(anchor='w', padx=20, pady=5)
    
    def _populate_advanced_analytics(self, parent):
        """Populate advanced analytics tab"""
        # Clear existing widgets
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Scrollable frame
        canvas = tk.Canvas(parent, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Dark.TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        ttk.Label(
            scrollable_frame,
            text=" Advanced Analytics",
            style='Title.TLabel'
        ).pack(pady=(0, 20))
        
        if not self.emotion_log:
            ttk.Label(
                scrollable_frame,
                text="No emotion data yet. Start detection to see analytics!",
                style='Dark.TLabel',
                font=('Segoe UI', 11)
            ).pack(pady=20)
            return
        
        # Wellbeing Score
        wellbeing_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        wellbeing_frame.pack(fill='x', pady=10, padx=20)
        
        ttk.Label(
            wellbeing_frame,
            text=" Wellbeing Score",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor='w')
        
        wellbeing = self.analytics.calculate_wellbeing_score(self.emotion_log)
        score_color = self.analytics.get_score_color(wellbeing)
        
        ttk.Label(
            wellbeing_frame,
            text=f"{wellbeing:.1f}/100",
            style='Emotion.TLabel',
            foreground=score_color
        ).pack(anchor='w', padx=20)
        
        # Progress bar
        self._create_progress_bar(wellbeing_frame, wellbeing, score_color)
        
        ttk.Label(
            wellbeing_frame,
            text=self.analytics.get_wellbeing_interpretation(wellbeing),
            style='Dark.TLabel',
            font=('Segoe UI', 9, 'italic')
        ).pack(anchor='w', padx=20, pady=(5, 0))
        
        # Productivity Score
        productivity_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        productivity_frame.pack(fill='x', pady=10, padx=20)
        
        ttk.Label(
            productivity_frame,
            text=" Productivity Index",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor='w')
        
        productivity = self.analytics.calculate_productivity_score(self.emotion_log)
        prod_color = self.analytics.get_score_color(productivity)
        
        ttk.Label(
            productivity_frame,
            text=f"{productivity:.1f}/100",
            style='Emotion.TLabel',
            foreground=prod_color
        ).pack(anchor='w', padx=20)
        
        self._create_progress_bar(productivity_frame, productivity, prod_color)
        
        if productivity >= 70:
            ttk.Label(
                productivity_frame,
                text="Excellent! You're in a great state for focused work.",
                style='Dark.TLabel',
                font=('Segoe UI', 9, 'italic')
            ).pack(anchor='w', padx=20, pady=(5, 0))
        elif productivity < 40:
            ttk.Label(
                productivity_frame,
                text="Consider taking a short break to reset.",
                style='Dark.TLabel',
                font=('Segoe UI', 9, 'italic')
            ).pack(anchor='w', padx=20, pady=(5, 0))
        
        # Stress Indicators
        stress_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        stress_frame.pack(fill='x', pady=10, padx=20)
        
        ttk.Label(
            stress_frame,
            text=" Stress Indicators (Last 24h)",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor='w')
        
        stress_count = self.analytics.get_recent_stress_count(24)
        stress_level = "Low" if stress_count < 5 else "Moderate" if stress_count < 15 else "High"
        stress_color = "#00ff88" if stress_count < 5 else "#ffaa00" if stress_count < 15 else "#ff4444"
        
        ttk.Label(
            stress_frame,
            text=f"Count: {stress_count} | Status: {stress_level}",
            style='Dark.TLabel',
            font=('Segoe UI', 10),
            foreground=stress_color
        ).pack(anchor='w', padx=20)
        
        if stress_count >= 15:
            ttk.Label(
                stress_frame,
                text=" High stress detected. Try breathing exercises or meditation.",
                style='Dark.TLabel',
                font=('Segoe UI', 9, 'italic'),
                foreground='#ff4444'
            ).pack(anchor='w', padx=20, pady=(5, 0))
        elif stress_count < 5:
            ttk.Label(
                stress_frame,
                text=" Great! You're managing stress well.",
                style='Dark.TLabel',
                font=('Segoe UI', 9, 'italic'),
                foreground='#00ff88'
            ).pack(anchor='w', padx=20, pady=(5, 0))
        
        # Emotion Stability
        stability_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        stability_frame.pack(fill='x', pady=10, padx=20)
        
        ttk.Label(
            stability_frame,
            text=" Emotional Stability",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor='w')
        
        stability = self.analytics.calculate_stability_score(self.emotion_log)
        stability_color = self.analytics.get_score_color(stability)
        
        ttk.Label(
            stability_frame,
            text=f"{stability:.1f}/100",
            style='Emotion.TLabel',
            foreground=stability_color
        ).pack(anchor='w', padx=20)
        
        self._create_progress_bar(stability_frame, stability, stability_color)
        
        if stability > 70:
            ttk.Label(
                stability_frame,
                text="Excellent emotional stability!",
                style='Dark.TLabel',
                font=('Segoe UI', 9, 'italic')
            ).pack(anchor='w', padx=20, pady=(5, 0))
        elif stability < 40:
            ttk.Label(
                stability_frame,
                text="Your emotions have been fluctuating. Consider a calming routine.",
                style='Dark.TLabel',
                font=('Segoe UI', 9, 'italic')
            ).pack(anchor='w', padx=20, pady=(5, 0))
        
        # Session Statistics
        if self.session_start_time:
            session_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
            session_frame.pack(fill='x', pady=10, padx=20)
            
            ttk.Label(
                session_frame,
                text=" Current Session Stats",
                style='Dark.TLabel',
                font=('Segoe UI', 12, 'bold')
            ).pack(anchor='w')
            
            duration = datetime.now() - self.session_start_time
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            ttk.Label(
                session_frame,
                text=f"Duration: {hours}h {minutes}m",
                style='Dark.TLabel'
            ).pack(anchor='w', padx=20)
            
            ttk.Label(
                session_frame,
                text=f"Total Detections: {len(self.emotion_log)}",
                style='Dark.TLabel'
            ).pack(anchor='w', padx=20)
    
    def _populate_patterns_insights(self, parent):
        """Populate patterns and insights tab"""
        # Clear existing widgets
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Scrollable frame
        canvas = tk.Canvas(parent, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Dark.TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        ttk.Label(
            scrollable_frame,
            text=" Patterns & Insights",
            style='Title.TLabel'
        ).pack(pady=(0, 20))
        
        if not self.emotion_log:
            ttk.Label(
                scrollable_frame,
                text="No emotion data yet. Start detection to see patterns!",
                style='Dark.TLabel',
                font=('Segoe UI', 11)
            ).pack(pady=20)
            return
        
        # Emotion Transitions
        if self.analytics.emotion_transitions:
            trans_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
            trans_frame.pack(fill='x', pady=10, padx=20)
            
            ttk.Label(
                trans_frame,
                text=" Most Common Emotion Transitions",
                style='Dark.TLabel',
                font=('Segoe UI', 12, 'bold')
            ).pack(anchor='w')
            
            # Get top 5 transitions
            all_transitions = []
            for from_emotion, to_dict in self.analytics.emotion_transitions.items():
                for to_emotion, count in to_dict.items():
                    all_transitions.append((from_emotion, to_emotion, count))
            
            all_transitions.sort(key=lambda x: x[2], reverse=True)
            
            if all_transitions:
                for from_em, to_em, count in all_transitions[:5]:
                    ttk.Label(
                        trans_frame,
                        text=f"{self.get_emotion_icon(from_em)} {from_em} → {self.get_emotion_icon(to_em)} {to_em}: {count} times",
                        style='Dark.TLabel'
                    ).pack(anchor='w', padx=20, pady=2)
            else:
                ttk.Label(
                    trans_frame,
                    text="Keep using the app to detect transition patterns!",
                    style='Dark.TLabel',
                    font=('Segoe UI', 9, 'italic')
                ).pack(anchor='w', padx=20)
        
        # Peak Emotion Hours
        if self.analytics.hourly_emotions:
            hourly_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
            hourly_frame.pack(fill='x', pady=10, padx=20)
            
            ttk.Label(
                hourly_frame,
                text=" Emotion Patterns by Hour",
                style='Dark.TLabel',
                font=('Segoe UI', 12, 'bold')
            ).pack(anchor='w')
            
            # Find peak emotion for each hour that has data
            for hour in sorted(self.analytics.hourly_emotions.keys()):
                emotions_dict = self.analytics.hourly_emotions[hour]
                if emotions_dict:
                    peak_emotion = max(emotions_dict.items(), key=lambda x: x[1])
                    hour_12 = hour % 12 if hour % 12 != 0 else 12
                    am_pm = 'AM' if hour < 12 else 'PM'
                    
                    ttk.Label(
                        hourly_frame,
                        text=f"{hour_12}:00 {am_pm}: {self.get_emotion_icon(peak_emotion[0])} {peak_emotion[0]} ({peak_emotion[1]} detections)",
                        style='Dark.TLabel'
                    ).pack(anchor='w', padx=20, pady=2)
        
        # Personalized Insights
        insights_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        insights_frame.pack(fill='x', pady=10, padx=20)
        
        ttk.Label(
            insights_frame,
            text="Personalized Insights",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor='w')
        
        wellbeing = self.analytics.calculate_wellbeing_score(self.emotion_log)
        productivity = self.analytics.calculate_productivity_score(self.emotion_log)
        insights = self.analytics.generate_insights(self.emotion_log, wellbeing, productivity)
        
        for insight in insights:
            insight_label = ttk.Label(
                insights_frame,
                text=f"• {insight}",
                style='Dark.TLabel',
                wraplength=600
            )
            insight_label.pack(anchor='w', padx=20, pady=3)
        
        # Recommendations Section
        rec_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        rec_frame.pack(fill='x', pady=10, padx=20)
        
        ttk.Label(
            rec_frame,
            text="Recommendations",
            style='Dark.TLabel',
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor='w')
        
        # Generate recommendations based on current state
        recommendations = []
        
        if wellbeing < 50:
            recommendations.append("Consider scheduling regular breaks throughout your day")
            recommendations.append("Try the suggested self-care actions for your current emotion")
        
        if productivity < 40:
            recommendations.append("Take a 5-10 minute break to reset your focus")
            recommendations.append("Consider changing your environment or activity")
        
        stress_count = self.analytics.get_recent_stress_count(24)
        if stress_count >= 10:
            recommendations.append("Practice breathing exercises or meditation")
            recommendations.append("Identify and address stress triggers")
        
        stability = self.analytics.calculate_stability_score(self.emotion_log)
        if stability < 40:
            recommendations.append("Establish a consistent daily routine")
            recommendations.append("Track your mood patterns to identify triggers")
        
        if not recommendations:
            recommendations.append("Keep up the great work! You're doing well.")
            recommendations.append("Continue monitoring your emotions regularly")
            recommendations.append("Download weekly reports to track long-term progress")
        
        for rec in recommendations:
            ttk.Label(
                rec_frame,
                text=f"✓ {rec}",
                style='Dark.TLabel',
                wraplength=600
            ).pack(anchor='w', padx=20, pady=3)
    
    def _create_progress_bar(self, parent, value, color):
        """Create a visual progress bar"""
        bar_frame = tk.Frame(parent, bg='#3a3a3a', height=20)
        bar_frame.pack(fill='x', padx=20, pady=5)
        
        # Calculate fill width (max 560px with padding)
        fill_width = int((value / 100) * 560)
        fill_bar = tk.Frame(bar_frame, bg=color, height=20, width=fill_width)
        fill_bar.place(x=0, y=0)

    # SPECIFIC SONG METHODS 
    # Happy songs
    def song_happy_pharrell(self):
        webbrowser.open("https://www.youtube.com/watch?v=ZbZSe6N_BXs")
        messagebox.showinfo("🎵 Now Playing", '"Happy" by Pharrell Williams\nBecause you\'re happy!')

    def song_uptown_funk(self):
        webbrowser.open("https://www.youtube.com/watch?v=OPf0YbXqDm0")
        messagebox.showinfo("🎵 Now Playing", '"Uptown Funk" by Bruno Mars\nDon\'t believe me just watch!')

    def song_good_as_hell(self):
        webbrowser.open("https://www.youtube.com/watch?v=SmbmeOgWsqE")
        messagebox.showinfo("🎵 Now Playing", '"Good as Hell" by Lizzo\nFeeling good as hell!')

    # sad songs
    def song_fix_you(self):
        webbrowser.open("https://www.youtube.com/watch?v=k4V3Mo61fJM")
        messagebox.showinfo("🎵 Now Playing", '"Fix You" by Coldplay\nLights will guide you home...')

    def song_lean_on_me(self):
        webbrowser.open("https://www.youtube.com/watch?v=fOZ-MySzAac")
        messagebox.showinfo("🎵 Now Playing", '"Lean on Me" by Bill Withers\nYou\'re not alone.')

    def song_here_comes_the_sun(self):
        webbrowser.open("https://www.youtube.com/watch?v=KQetemT1sWc")
        messagebox.showinfo("🎵 Now Playing", '"Here Comes the Sun" by The Beatles\nIt\'s alright...')

    # Angry songs
    def song_weightless(self):
        webbrowser.open("https://www.youtube.com/watch?v=UfcAVejslrU")
        messagebox.showinfo("🎵 Now Playing", '"Weightless" by Marconi Union\nScientifically proven to reduce anxiety by 65%.')

    def song_breathe_me(self):
        webbrowser.open("https://www.youtube.com/watch?v=SFGvmrJ5rjM")
        messagebox.showinfo("🎵 Now Playing", '"Breathe Me" by Sia\nBreathe... let it go.')

    def song_let_it_be(self):
        webbrowser.open("https://www.youtube.com/watch?v=QDYfEBY9NM4")
        messagebox.showinfo("🎵 Now Playing", '"Let It Be" by The Beatles\nLet it be...')

    # Fear songs
    def song_over_the_rainbow(self):
        webbrowser.open("https://www.youtube.com/watch?v=V1bFr2SWP1I")
        messagebox.showinfo("🎵 Now Playing", '"Somewhere Over the Rainbow" by Israel Kamakawiwo\'ole\nYou are safe.')

    def song_brave(self):
        webbrowser.open("https://www.youtube.com/watch?v=QUQsqBqxoR4")
        messagebox.showinfo("🎵 Now Playing", '"Brave" by Sara Bareilles\nSay what you wanna say!')

    def song_three_little_birds(self):
        webbrowser.open("https://www.youtube.com/watch?v=zaGUr6wBO-A")
        messagebox.showinfo("🎵 Now Playing", '"Three Little Birds" by Bob Marley\nEvery little thing is gonna be alright.')

    # Surprise songs
    def song_dont_stop_me_now(self):
        webbrowser.open("https://www.youtube.com/watch?v=HgzGwKwLmgM")
        messagebox.showinfo("🎵 Now Playing", '"Don\'t Stop Me Now" by Queen\nI\'m having such a good time!')

    def song_celebration(self):
        webbrowser.open("https://www.youtube.com/watch?v=3GwjfUFyY6M")
        messagebox.showinfo("🎵 Now Playing", '"Celebration" by Kool & The Gang\nCelebrate good times!')

    def song_walking_on_sunshine(self):
        webbrowser.open("https://www.youtube.com/watch?v=iPUmE-tne5U")
        messagebox.showinfo("🎵 Now Playing", '"Walking on Sunshine" by Katrina & The Waves\nAnd don\'t it feel good!')

    # Disgust songs
    def song_wonderful_world(self):
        webbrowser.open("https://www.youtube.com/watch?v=A3yCcXgbKrE")
        messagebox.showinfo("🎵 Now Playing", '"What a Wonderful World" by Louis Armstrong\nSee the beauty around you.')

    def song_ocean_eyes(self):
        webbrowser.open("https://www.youtube.com/watch?v=viimfQi_pUw")
        messagebox.showinfo("🎵 Now Playing", '"Ocean Eyes" by Billie Eilish\nGentle and cleansing...')

    def song_clair_de_lune(self):
        webbrowser.open("https://www.youtube.com/watch?v=CvFH_6DNRCY")
        messagebox.showinfo("🎵 Now Playing", '"Clair de Lune" by Debussy\nPure, elegant calm.')

    # Neutral songs
    def song_bohemian_rhapsody(self):
        webbrowser.open("https://www.youtube.com/watch?v=fJ9rUzIMcZQ")
        messagebox.showinfo("🎵 Now Playing", '"Bohemian Rhapsody" by Queen\nA masterpiece to enjoy.')

    def song_blinding_lights(self):
        webbrowser.open("https://www.youtube.com/watch?v=4NRXx6U8ABQ")
        messagebox.showinfo("🎵 Now Playing", '"Blinding Lights" by The Weeknd\nGreat vibes!')

    def song_lofi_radio(self):
        webbrowser.open("https://www.youtube.com/watch?v=jfKfPfyJRdk")
        messagebox.showinfo("🎵 Now Playing", '"Lofi Hip Hop Radio"\nBeats to relax/study to.')

    # SPECIFIC GAME METHODS
    # Happy games 
    def game_pacman(self):
        webbrowser.open("https://www.google.com/logos/2010/pacman10-i.html")
        messagebox.showinfo("🎮 Game Time", "Play Pac-Man right in your browser!\nGoogle's classic Pac-Man game.")

    def game_friday_night_funkin(self):
        webbrowser.open("https://www.crazygames.com/game/friday-night-funkin")
        messagebox.showinfo("🎮 Game Time", "Friday Night Funkin'!\nBeat your opponents in epic music battles!")

    # Sad games
    def game_little_alchemy(self):
        webbrowser.open("https://www.crazygames.com/game/little-alchemy")
        messagebox.showinfo("🎮 Game Time", "Little Alchemy!\nMix elements and discover new things. Gentle and creative!")

    def game_bubble_shooter(self):
        webbrowser.open("https://www.crazygames.com/game/bubble-shooter-classic")
        messagebox.showinfo("🎮 Game Time", "Bubble Shooter!\nPop bubbles — simple, satisfying, and soothing.")

    # Angry games 
    def game_slice_master(self):
        webbrowser.open("https://www.crazygames.com/game/slice-master")
        messagebox.showinfo("🎮 Game Time", "Slice Master!\nSlash and slice everything in sight to release tension!")

    def game_punchers(self):
        webbrowser.open("https://www.crazygames.com/game/punchers")
        messagebox.showinfo("🎮 Game Time", "Punchers!\nPhysics-based boxing — punch it out safely!")

    # Fear games
    def game_mahjong_solitaire(self):
        webbrowser.open("https://www.crazygames.com/game/mahjongg-solitaire")
        messagebox.showinfo("🎮 Game Time", "Mahjong Solitaire!\nA calm, meditative tile-matching puzzle.")

    def game_color_fill(self):
        webbrowser.open("https://www.crazygames.com/game/color-fill-3d")
        messagebox.showinfo("🎮 Game Time", "Color Fill 3D!\nFill colors peacefully — simple and calming.")

    # Surprise games
    def game_agario(self):
        webbrowser.open("https://agar.io")
        messagebox.showinfo("🎮 Game Time", "Agar.io!\nGrow, eat, and surprise others!")

    def game_helix_jump(self):
        webbrowser.open("https://www.crazygames.com/game/helix-jump")
        messagebox.showinfo("🎮 Game Time", "Helix Jump!\nBounce your way down — exciting and addictive!")

    # Disgust games
    def game_tile_guru(self):
        webbrowser.open("https://www.crazygames.com/game/tile-guru")
        messagebox.showinfo("🎮 Game Time", "Tile Guru!\nZen-inspired tile matching — peaceful and relaxing.")

    def game_jigsaw_puzzle(self):
        webbrowser.open("https://www.jigsawplanet.com")
        messagebox.showinfo("🎮 Game Time", "Solve beautiful jigsaw puzzles!\nFocus on something lovely.")

    # Neutral games
    def game_2048(self):
        webbrowser.open("https://play2048.co")
        messagebox.showinfo("🎮 Game Time", "Play 2048!\nClassic brain teaser — how high can you score?")

    def game_tetris(self):
        webbrowser.open("https://tetris.com/play-tetris")
        messagebox.showinfo("🎮 Game Time", "Play official Tetris!\nThe timeless classic.")

    def game_chess(self):
        webbrowser.open("https://www.chess.com/play/online")
        messagebox.showinfo("🎮 Game Time", "Play Chess online!\nChallenge your mind.")

    # ALL ACTION METHODS 
    def play_upbeat_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=upbeat+happy+music+playlist")
        messagebox.showinfo("🎵 Music", "Opening upbeat music to match your mood!")
    
    def play_party_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=party+dance+music")
        messagebox.showinfo("🎉 Party Time", "Let's get the party started!")
    
    def open_creative_apps(self):
        if platform.system() == "Windows":
            try:
                subprocess.Popen("mspaint.exe")
                messagebox.showinfo("🎨 Creative", "Opening Paint for you!")
            except:
                webbrowser.open("https://www.photopea.com")
                messagebox.showinfo("🎨 Creative", "Opening online creative tools!")
        else:
            webbrowser.open("https://www.photopea.com")
            messagebox.showinfo("🎨 Creative", "Opening creative tools for you!")
    
    def open_video_call(self):
        webbrowser.open("https://meet.google.com")
        messagebox.showinfo("💬 Connect", "Share your happiness with friends!")
    
    def open_happy_journal(self):
        self.open_journal()
        messagebox.showinfo("📝 Journal", "Document your happiness!")

    def play_comforting_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=comforting+peaceful+music")
        messagebox.showinfo("🎵 Comfort", "Playing music to comfort you...")
    
    def watch_comedy(self):
        webbrowser.open("https://www.youtube.com/results?search_query=funny+comedy+videos+2024")
        messagebox.showinfo("😄 Comedy", "Let's lift your spirits with some laughter!")
    
    def open_mood_lifting(self):
        webbrowser.open("https://www.youtube.com/results?search_query=mood+lifting+feel+good+videos")
        messagebox.showinfo("🌈 Feel Good", "Here's something to brighten your day!")
    
    def open_messaging(self):
        webbrowser.open("https://web.whatsapp.com")
        messagebox.showinfo("💬 Connect", "Reach out to someone!")
    
    def play_healing_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=healing+music+emotional")
        messagebox.showinfo("🎧 Healing", "Let this music help heal your heart...")
    
    def show_support_resources(self):
        messagebox.showinfo(
            " Support",
            "You're not alone. Support is available:\n\n"
            "• Mental Health Hotline: 1-800-662-4357\n"
            "• Crisis Text Line: Text HOME to 741741\n"
            "• Suicide Prevention: 988\n"
        )

    def play_calming_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=calming+meditation+music")
        messagebox.showinfo("🎵 Calm", "Soothing sounds for inner peace...")
    
    def open_fitness_app(self):
        webbrowser.open("https://www.youtube.com/results?search_query=quick+workout+anger+relief")
        messagebox.showinfo("🏃 Exercise", "Channel that energy into movement!")
    
    def open_stress_games(self):
        webbrowser.open("https://www.crazygames.com/t/stress-relief")
        messagebox.showinfo("🎮 Games", "Try these stress-relief games!")
    
    def open_stress_relief(self):
        webbrowser.open("https://www.youtube.com/results?search_query=virtual+stress+relief+activities")
        messagebox.showinfo("🥊 Relief", "Let it out in a healthy way!")
    
    def show_anger_tips(self):
        tips = [
            "Take 10 deep breaths slowly",
            "Count backwards from 100",
            "Go for a brisk walk",
            "Write down what's bothering you",
            "Listen to calming music",
            "Do some intense exercise",
            "Squeeze a stress ball",
            "Step away from the situation"
        ]
        messagebox.showinfo("💡 Anger Management", f"Try this:\n\n{np.random.choice(tips)}")
    
    def suggest_productive_activity(self):
        activities = [
            "Organize your workspace",
            "Clean a room",
            "Do a workout",
            "Learn a new skill online",
            "Work on a project",
            "Plan tomorrow's tasks"
        ]
        messagebox.showinfo("🎯 Productive", f"Channel your energy:\n\n{np.random.choice(activities)}")
    
    def show_cooldown_tips(self):
        messagebox.showinfo(
            " Cool Down",
            "• Splash cold water on your face\n"
            "• Take 5 slow, deep breaths\n"
            "• Count to 10 slowly\n"
            "• Step outside for fresh air\n"
            "• Drink cold water"
        )
    
    def open_mood_tracker(self):
        self.open_journal()
        messagebox.showinfo(" Track", "Document what triggered this feeling.")

    def show_safety_resources(self):
        messagebox.showinfo(
            "🔒 Safety",
            "Resources:\n• Crisis Helpline: 988\n• Emergency: 911\n• Crisis Text: HOME to 741741"
        )
    
    def open_support_chat(self):
        webbrowser.open("https://www.7cups.com")
        messagebox.showinfo("💬 Support", "Connect with trained listeners.")
    
    def play_anxiety_relief(self):
        webbrowser.open("https://www.youtube.com/results?search_query=anxiety+relief+calming+sounds")
        messagebox.showinfo("🎧 Calm", "Soothing sounds to ease anxiety...")
    
    def show_grounding_techniques(self):
        messagebox.showinfo("🛡️ Grounding", "5-4-3-2-1 Technique:\n5 see • 4 touch • 3 hear • 2 smell • 1 taste")
    
    def show_peace_guide(self):
        messagebox.showinfo("☮️ Peace", "Focus on breathing. You are safe. This will pass.")

    def open_video_recorder(self):
        if platform.system() == "Windows":
            try:
                subprocess.Popen("start microsoft.windows.camera:", shell=True)
                messagebox.showinfo("📹 Camera", "Capture this moment!")
            except:
                messagebox.showinfo("📹 Camera", "Open your camera app to record this moment!")
        else:
            messagebox.showinfo("📹 Camera", "Open your camera app to capture this moment!")
    
    def open_exciting_content(self):
        webbrowser.open("https://www.youtube.com/results?search_query=exciting+amazing+moments")
        messagebox.showinfo("⚡ Exciting", "More amazing content for you!")
    
    def show_reflection_prompt(self):
        prompts = [
            "What surprised you most?",
            "How does this make you feel?",
            "Who would you like to share this with?"
        ]
        messagebox.showinfo(" Reflect", f"Take a moment:\n\n{np.random.choice(prompts)}")

    def suggest_fresh_air(self):
        messagebox.showinfo(" Fresh Air","Step outside for 5 deep breaths and look at something green.")
    
    def show_cleansing_tips(self):
        tips = [
            "Take a refreshing shower",
            "Open windows for fresh air",
            "Organize your space",
            "Change into clean clothes"
        ]
        messagebox.showinfo("🧼 Cleansing", f"Try this:\n\n{np.random.choice(tips)}")
    
    def show_comfort_recipes(self):
        webbrowser.open("https://www.youtube.com/results?search_query=comfort+food+recipes")
        messagebox.showinfo("🍵 Comfort", "Find something comforting to make!")
    
    def open_art_therapy(self):
        webbrowser.open("https://www.youtube.com/results?search_query=art+therapy+relaxing")
        messagebox.showinfo("🎨 Art Therapy", "Express yourself through art!")
    
    def show_cleansing_visualization(self):
        messagebox.showinfo("🌊 Visualization","Imagine a waterfall washing away the negativity.")
    
    def show_space_reset_tips(self):
        messagebox.showinfo("💚 Space Reset","Open windows, clear clutter, play uplifting music.")

    def discover_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=music+discovery+mix")
        messagebox.showinfo("🎵 Discover", "Find new music you'll love!")
    
    def open_learning_resources(self):
        webbrowser.open("https://www.coursera.org")
        messagebox.showinfo("📚 Learn", "Explore free courses.")
    
    def open_productivity(self):
        webbrowser.open("https://todoist.com")
        messagebox.showinfo("🎯 Productivity", "Get organized.")
    
    def explore_interests(self):
        webbrowser.open("https://www.youtube.com/")
        messagebox.showinfo("🌐 Explore", "Discover something new today!")
    
    def open_planner(self):
        try:
            if platform.system() == "Windows":
                subprocess.Popen("notepad.exe")
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-a", "TextEdit"])
            else:
                subprocess.Popen(["gedit"])
            messagebox.showinfo("📊 Planning", "Plan your day ahead!")
        except:
            messagebox.showinfo("📊 Planning", "Open your favorite note app to plan your day!")
    
    def open_brain_games(self):
        webbrowser.open("https://www.lumosity.com")
        messagebox.showinfo("🧩 Brain Games", "Challenge your mind!")
    
    def open_reading(self):
        webbrowser.open("https://medium.com")
        messagebox.showinfo("📖 Reading", "Discover interesting articles.")
    
    def open_goal_setting(self):
        self.open_journal()
        messagebox.showinfo("🌟 Goals", "Write down your goals and aspirations.")
    
    def play_relaxing_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=relaxing+calm+music")
        messagebox.showinfo("🎵 Relax", "Peaceful music to calm your mind!")
    
    def play_nature_sounds(self):
        webbrowser.open("https://www.youtube.com/results?search_query=nature+sounds+rain+forest")
        messagebox.showinfo("🌿 Nature", "Immerse yourself in nature sounds!")
    
    def open_games(self):
        if platform.system() == "Windows":
            try:
                subprocess.Popen("start steam://", shell=True)
                messagebox.showinfo("🎮 Games", "Opening Steam!")
            except:
                webbrowser.open("https://www.crazygames.com")
                messagebox.showinfo("🎮 Games", "Time for some fun online games!")
        else:
            webbrowser.open("https://www.crazygames.com")
            messagebox.showinfo("🎮 Games", "Time for some fun!")
    
    def open_youtube(self):
        webbrowser.open("https://www.youtube.com")
        messagebox.showinfo("📺 YouTube", "Explore videos that interest you!")
    
    def open_social_media(self):
        webbrowser.open("https://www.twitter.com")
        messagebox.showinfo("📱 Social", "Connect with your network!")
    
    def open_camera_app(self):
        if platform.system() == "Windows":
            try:
                subprocess.Popen("start microsoft.windows.camera:", shell=True)
                messagebox.showinfo("📸 Camera", "Camera opened!")
            except:
                messagebox.showinfo("📸 Camera", "Open your camera app to capture moments!")
        elif platform.system() == "Darwin":
            messagebox.showinfo("📸 Camera", "Open Photo Booth to capture moments!")
        else:
            messagebox.showinfo("📸 Camera", "Open your camera app to capture moments!")
    
    def show_selfcare_tips(self):
        tips = [
            "Take a warm bath",
            "Deep breathing 5 minutes",
            "Go for a peaceful walk",
            "Call a friend",
            "Write in a journal",
            "Gentle stretching",
            "Make a comforting drink",
            "Read a chapter of a book"
        ]
        messagebox.showinfo("💝 Self-Care", f"Self-care tip:\n\n{np.random.choice(tips)}")
    
    def show_motivational_quotes(self):
        quotes = [
            "Every day is a new beginning!",
            "You are stronger than you think!",
            "This too shall pass.",
            "Believe in yourself!",
            "You've got this!",
            "Progress, not perfection.",
            "You are capable of amazing things!"
        ]
        messagebox.showinfo("✨ Motivation", np.random.choice(quotes))
    
    def open_meditation(self):
        webbrowser.open("https://www.youtube.com/results?search_query=guided+meditation+10+minutes")
        messagebox.showinfo("🧘 Meditation", "Find inner peace with guided meditation!")
    
    def start_breathing_exercise(self):
        messagebox.showinfo("🌬️ Breathing Exercise","IN 4 • HOLD 4 • OUT 6 • HOLD 2\nRepeat 5–10 times.")
    
    def suggest_exercise(self):
        exercises = [
            "10-minute brisk walk",
            "20 jumping jacks",
            "5-minute yoga session",
            "Dance to your favorite song",
            "10 push-ups",
            "Full-body stretch"
        ]
        messagebox.showinfo("🏃 Exercise", f"Try this:\n\n{np.random.choice(exercises)}")
    
    def open_journal(self):
        try:
            if platform.system() == "Windows":
                subprocess.Popen("notepad.exe")
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-a", "TextEdit"])
            else:
                subprocess.Popen(["gedit"])
            messagebox.showinfo("✍️ Journal", "Express your thoughts in writing!")
        except:
            webbrowser.open("https://docs.google.com")
            messagebox.showinfo("✍️ Journal", "Opening Google Docs for journaling!")
    
    def show_emergency_contacts(self):
        messagebox.showinfo(
            " Emergency Support",
            "Help is available 24/7:\n• 988 (US)\n• 911 (Emergency)\n• Text HOME to 741741"
        )
    
    def show_affirmations(self):
        affirmations = [
            "I am brave and capable.",
            "I focus on what I can control.",
            "I am safe right now.",
            "I am resilient.",
            "I deserve peace and happiness."
        ]
        messagebox.showinfo("💫 Affirmation", np.random.choice(affirmations))
    
    def show_celebration_ideas(self):
        ideas = [
            "Share with friends",
            "Treat yourself",
            "Take a photo",
            "Happy dance",
            "Journal the moment"
        ]
        messagebox.showinfo("🎉 Celebrate", f"Try this:\n\n{np.random.choice(ideas)}")
    
    def show_nature_content(self):
        webbrowser.open("https://www.youtube.com/results?search_query=beautiful+nature+scenery+4k+relaxing")
        messagebox.showinfo("🌸 Nature", "Immerse yourself in beautiful nature!")
    
    # REPORT GENERATION 
    def generate_report_dialog(self):
        """Show dialog to choose report format"""
        if not self.current_user:
            messagebox.showwarning("Login Required", "Please login to generate reports.")
            return
        
        if not self.emotion_log:
            messagebox.showwarning("No Data", "No emotion data to generate report. Start detection first!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Generate Report")
        dialog.configure(bg='#1a1a1a')
        dialog.geometry("400x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 200
        y = (dialog.winfo_screenheight() // 2) - 175
        dialog.geometry(f"400x350+{x}+{y}")
        
        frame = ttk.Frame(dialog, style='Dark.TFrame', padding=20)
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="📄 Generate Emotion Report", style='Title.TLabel').pack(pady=(0, 20))
        ttk.Label(frame, text=f"User: {self.current_user}", style='Dark.TLabel', font=('Segoe UI', 10)).pack(pady=5)
        ttk.Label(frame, text=f"Total Emotions Logged: {len(self.emotion_log)}", style='Dark.TLabel', font=('Segoe UI', 10)).pack(pady=5)
        
        wellbeing = self.analytics.calculate_wellbeing_score(self.emotion_log)
        ttk.Label(frame, text=f"Wellbeing Score: {wellbeing:.1f}/100", style='Dark.TLabel', font=('Segoe UI', 10)).pack(pady=5)
        
        ttk.Label(frame, text="\nSelect Report Format:", style='Dark.TLabel', font=('Segoe UI', 11, 'bold')).pack(pady=(15, 10))
        
        pdf_status = "✓ Available" if REPORTLAB_AVAILABLE else "✗ Install reportlab"
        ttk.Button(frame, text=f"📕 PDF Report {pdf_status}", style='Gesture.TButton' if REPORTLAB_AVAILABLE else 'Dark.TButton', command=lambda: self._generate_pdf_report(dialog), state='normal' if REPORTLAB_AVAILABLE else 'disabled').pack(fill='x', pady=5)
        
        excel_status = "✓ Available" if PANDAS_AVAILABLE else "✗ Install pandas"
        ttk.Button(frame, text=f"📗 Excel Report {excel_status}", style='Gesture.TButton' if PANDAS_AVAILABLE else 'Dark.TButton', command=lambda: self._generate_excel_report(dialog), state='normal' if PANDAS_AVAILABLE else 'disabled').pack(fill='x', pady=5)
        
        ttk.Button(frame, text="📄 JSON Data Export", style='Dark.TButton', command=lambda: self._generate_json_report(dialog)).pack(fill='x', pady=5)
        ttk.Label(frame, text="\n💡 Tip: Install missing libraries to enable all formats", style='Dark.TLabel', font=('Segoe UI', 8, 'italic')).pack(pady=(10, 0))
    
    def _generate_pdf_report(self, dialog):
        """Generate comprehensive PDF report"""
        try:
            filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")], initialfile=f"{self.current_user}_emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            if not filename:
                return
            ReportGenerator.generate_pdf_report(filename, self.current_user, self.emotion_log, self.analytics)
            dialog.destroy()
            messagebox.showinfo("Success", f"PDF report generated successfully!\n\nSaved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF report:\n{str(e)}")
    
    def _generate_excel_report(self, dialog):
        """Generate Excel report with multiple sheets"""
        try:
            filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")], initialfile=f"{self.current_user}_emotion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            if not filename:
                return
            ReportGenerator.generate_excel_report(filename, self.current_user, self.emotion_log, self.analytics)
            dialog.destroy()
            messagebox.showinfo("Success", f"Excel report generated successfully!\n\nSaved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate Excel report:\n{str(e)}")
    
    def _generate_json_report(self, dialog):
        """Generate JSON data export"""
        try:
            filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialfile=f"{self.current_user}_emotion_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            if not filename:
                return
            ReportGenerator.generate_json_report(filename, self.current_user, self.emotion_log, self.analytics)
            dialog.destroy()
            messagebox.showinfo("Success", f"JSON data exported successfully!\n\nSaved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export JSON data:\n{str(e)}")

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
        if hasattr(self, 'gesture_controller'):
            self.gesture_controller.stop()
        if hasattr(self, 'popup_window') and self.popup_window is not None:
            try:
                if self.popup_window.winfo_exists():
                    self.popup_window.destroy()
            except:
                pass
        if hasattr(self, 'notification_window') and self.notification_window is not None:
            try:
                if self.notification_window.winfo_exists():
                    self.notification_window.destroy()
            except:
                pass


def main():
    root = tk.Tk()
    app = EmotionRecognitionApp(root)

    def on_closing():
        app.detection_active = False
        app.gesture_controller.stop()
        # Stop voice assistant
        if getattr(app, "voice_assistant", None) is not None:
            try:
                app.voice_assistant.stop()
            except:
                pass
        if getattr(app, "popup_window", None) is not None:
            try:
                if app.popup_window.winfo_exists():
                    app.popup_window.destroy()
            except:
                pass
        if getattr(app, "notification_window", None) is not None:
            try:
                if app.notification_window.winfo_exists():
                    app.notification_window.destroy()
            except:
                pass
        if hasattr(app, 'cap') and app.cap is not None:
            app.cap.release()
        if hasattr(app, 'face_mesh') and app.face_mesh is not None:
            app.face_mesh.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
