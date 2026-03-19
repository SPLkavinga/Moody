import tkinter as tk
from tkinter import ttk, messagebox
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
warnings.filterwarnings('ignore')

# Model / features (your pipeline) 
import joblib
from mediapipe.python.solutions import face_mesh as mp_face_mesh
#from mediapipe_local.python.solutions import face_mesh as mp_face_mesh
from collections import deque
from live_emotion_inference import FEATURE_ORDER, compute_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model2")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")


class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root

        # App state
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.detection_active = False
        self._proba_window = deque(maxlen=10)

        # Canonical 7 labels used by UI/actions
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Suggested actions per emotion 
        self.emotion_actions = {
            'happy': [
                (' Play Upbeat Music', self.play_upbeat_music),
                (' Open Games Platform', self.open_games),
                (' Browse YouTube', self.open_youtube),
                (' Open Social Media', self.open_social_media),
                (' Launch Camera App', self.open_camera_app),
                (' Open Paint/Creative Tools', self.open_creative_apps),
                (' Start Video Call', self.open_video_call),
                (' Play Party Music', self.play_party_music),
                (' Create Happy Journal Entry', self.open_happy_journal),
            ],
            'sad': [
                (' Play Comforting Music', self.play_comforting_music),
                (' Watch Comedy/Feel-Good Shows', self.watch_comedy),
                (' View Self-Care Guide', self.show_selfcare_tips),
                (' Read Motivational Content', self.show_motivational_quotes),
                (' Open Meditation App', self.open_meditation),
                (' Connect with Friends', self.open_messaging),
                (' Write in Journal', self.open_journal),
                (' Watch Mood-Lifting Videos', self.open_mood_lifting),
                (' Listen to Healing Sounds', self.play_healing_music),
                (' View Support Resources', self.show_support_resources),
                (' Positive Affirmations', self.show_affirmations),
                (' Contact Helpline Info', self.show_emergency_contacts),
            ],
            'angry': [
                (' Play Calming Music', self.play_calming_music),
                (' Start Breathing Exercise', self.start_breathing_exercise),
                (' Open Workout/Exercise Videos', self.open_fitness_app),
                (' Vent in Journal', self.open_journal),
                (' Listen to Nature Sounds', self.play_nature_sounds),
                (' Play Stress-Relief Games', self.open_stress_games),
                (' Virtual Stress Relief', self.open_stress_relief),
                (' Anger Management Tips', self.show_anger_tips),
                (' Redirect Energy Productively', self.suggest_productive_activity),
                (' Physical Exercise Guide', self.suggest_exercise),
                (' Cool Down Technique', self.show_cooldown_tips),
                (' Track Your Triggers', self.open_mood_tracker),
            ],
            'fear': [
                (' Play Comforting Music', self.play_comforting_music),
                (' Guided Meditation', self.open_meditation),
                (' Emergency Contacts', self.show_emergency_contacts),
                (' Empowerment Content', self.show_motivational_quotes),
                (' Positive Affirmations', self.show_affirmations),
                (' Safety Resources', self.show_safety_resources),
                (' Support Chat', self.open_support_chat),
                (' Anxiety Relief Audio', self.play_anxiety_relief),
                (' Breathing Exercises', self.start_breathing_exercise),
                (' Write Your Worries', self.open_journal),
                (' Grounding Techniques', self.show_grounding_techniques),
                (' Peace and Calm Guide', self.show_peace_guide),
            ],
            'surprise': [
                (' Capture This Moment', self.open_camera_app),
                (' Share on Social Media', self.open_social_media),
                (' Play Energetic Music', self.play_upbeat_music),
                (' Document Your Thoughts', self.open_journal),
                (' Celebration Ideas', self.show_celebration_ideas),
                (' Record Video Message', self.open_video_recorder),
                (' Share Your Excitement', self.open_messaging),
                (' Explore Exciting Content', self.open_exciting_content),
                (' Channel This Energy', self.suggest_productive_activity),
                (' Reflect on the Moment', self.show_reflection_prompt),
            ],
            'disgust': [
                (' Fresh Air Reminder', self.suggest_fresh_air),
                (' Play Pleasant Music', self.play_relaxing_music),
                (' Mindfulness Exercise', self.open_meditation),
                (' Self-Care Routine Guide', self.show_selfcare_tips),
                (' View Beautiful Nature', self.show_nature_content),
                (' Cleansing Rituals', self.show_cleansing_tips),
                (' Comfort Recipes', self.show_comfort_recipes),
                (' Art Therapy', self.open_art_therapy),
                (' Cleansing Visualization', self.show_cleansing_visualization),
                (' Reset Your Space', self.show_space_reset_tips),
            ],
            'neutral': [
                (' Discover New Music', self.discover_music),
                (' Learn Something New', self.open_learning_resources),
                (' Browse Casual Games', self.open_games),
                (' Explore Entertainment', self.open_youtube),
                (' Start Mood Journal', self.open_journal),
                (' Open Productivity Apps', self.open_productivity),
                (' Explore Your Interests', self.explore_interests),
                (' Daily Planning Tool', self.open_planner),
                (' Brain Training Games', self.open_brain_games),
                (' Read Articles/News', self.open_reading),
                (' Creative Projects', self.open_creative_apps),
                (' Set New Goals', self.open_goal_setting),
            ],
        }

        # UI + model + camera
        self.setup_ui()
        self.setup_model()
        self.setup_camera()
        self.setup_responsive_layout()  # restore previous responsive behavior

        # Start maximized (optional)
        try:
            if platform.system() == "Windows":
                self.root.state('zoomed')
            else:
                self.root.attributes('-zoomed', True)
        except Exception:
            pass

    # UI (RESTORED to previous layout)
    def setup_ui(self):
        self.root.title("Emotion Recognition Assistant")
        self.root.configure(bg='#1a1a1a')
        self.root.minsize(1000, 700)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#1a1a1a')
        style.configure('Dark.TLabel', background='#1a1a1a', foreground='#ffffff', font=('Segoe UI', 10))
        style.configure('Title.TLabel', background='#1a1a1a', foreground='#ffffff', font=('Segoe UI', 18, 'bold'))
        style.configure('Emotion.TLabel', background='#1a1a1a', foreground='#00ff88', font=('Segoe UI', 16, 'bold'))
        style.configure('Dark.TButton', background='#333333', foreground='#ffffff', font=('Segoe UI', 9), padding=8)
        style.map('Dark.TButton', background=[('active', '#444444'), ('pressed', '#555555')])

        # Main container with 3 columns (left=2, middle=1, right=2 weights)
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill='both', expand=True, padx=15, pady=15)
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=2)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_columnconfigure(2, weight=2)

        # Title
        title_label = ttk.Label(main_container, text="Emotion Recognition Assistant", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15), sticky='ew')

        # LEFT COLUMN Camera
        left_frame = ttk.Frame(main_container, style='Dark.TFrame')
        left_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 10))
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        camera_label = ttk.Label(left_frame, text=" Live Camera Feed", style='Dark.TLabel',
                                 font=('Segoe UI', 12, 'bold'))
        camera_label.grid(row=0, column=0, pady=(0, 10), sticky='w')

        self.camera_container = ttk.Frame(left_frame, style='Dark.TFrame')
        self.camera_container.grid(row=1, column=0, sticky='nsew')
        self.camera_container.grid_rowconfigure(0, weight=1)
        self.camera_container.grid_columnconfigure(0, weight=1)

        # Video label fills container we’ll scale frames to fit this size
        self.video_label = ttk.Label(self.camera_container, style='Dark.TLabel', anchor='center')
        self.video_label.grid(row=0, column=0, sticky='nsew')

        control_frame = ttk.Frame(left_frame, style='Dark.TFrame')
        control_frame.grid(row=2, column=0, pady=(10, 0), sticky='ew')
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(control_frame, text="Start Detection", style='Dark.TButton',
                                    command=self.start_detection)
        self.start_btn.grid(row=0, column=0, padx=(0, 5), sticky='ew')

        self.stop_btn = ttk.Button(control_frame, text="Stop Detection", style='Dark.TButton',
                                   command=self.stop_detection, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=(5, 0), sticky='ew')

        # MIDDLE COLUMN Emotion Display (centered content)
        middle_frame = ttk.Frame(main_container, style='Dark.TFrame')
        middle_frame.grid(row=1, column=1, sticky='nsew', padx=10)
        middle_frame.grid_rowconfigure(2, weight=1)  # space beneath history
        middle_frame.grid_columnconfigure(0, weight=1)

        emotion_title = ttk.Label(middle_frame, text=" Current Emotion", style='Dark.TLabel',
                                  font=('Segoe UI', 12, 'bold'))
        emotion_title.grid(row=0, column=0, pady=(0, 15), sticky='ew')

        # Center stack for icon and labels
        emotion_display_frame = ttk.Frame(middle_frame, style='Dark.TFrame')
        emotion_display_frame.grid(row=1, column=0, pady=(0, 20), sticky='ew')
        self.emotion_icon_label = ttk.Label(emotion_display_frame, text="😐", font=('Segoe UI', 64),
                                            style='Dark.TLabel', anchor='center')
        self.emotion_icon_label.pack()
        self.emotion_text_label = ttk.Label(emotion_display_frame, text="Neutral", style='Emotion.TLabel')
        self.emotion_text_label.pack()
        self.confidence_label = ttk.Label(emotion_display_frame, text="Confidence: 0%", style='Dark.TLabel',
                                          font=('Segoe UI', 11))
        self.confidence_label.pack(pady=(5, 0))

        # History
        history_label = ttk.Label(middle_frame, text=" Recent Emotions", style='Dark.TLabel',
                                  font=('Segoe UI', 11, 'bold'))
        history_label.grid(row=2, column=0, pady=(20, 10), sticky='new')
        self.history_frame = ttk.Frame(middle_frame, style='Dark.TFrame')
        self.history_frame.grid(row=3, column=0, sticky='nsew')

        # RIGHT COLUMN Actions (scrollable)
        right_frame = ttk.Frame(main_container, style='Dark.TFrame')
        right_frame.grid(row=1, column=2, sticky='nsew', padx=(10, 0))
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        actions_label = ttk.Label(right_frame, text=" Suggested Actions", style='Dark.TLabel',
                                  font=('Segoe UI', 12, 'bold'))
        actions_label.grid(row=0, column=0, pady=(0, 10), sticky='w')

        self.actions_canvas = tk.Canvas(right_frame, bg='#2a2a2a', highlightthickness=0)
        actions_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.actions_canvas.yview)
        self.actions_scrollable_frame = ttk.Frame(self.actions_canvas, style='Dark.TFrame')

        self.actions_scrollable_frame.bind(
            "<Configure>", lambda e: self.actions_canvas.configure(scrollregion=self.actions_canvas.bbox("all"))
        )
        # Keep inner frame width in sync with canvas width (so buttons stretch)
        self.actions_canvas.bind('<Configure>', self._on_canvas_configure)

        self.actions_canvas.create_window((0, 0), window=self.actions_scrollable_frame, anchor="nw")
        self.actions_canvas.configure(yscrollcommand=actions_scrollbar.set)
        self.actions_canvas.grid(row=1, column=0, sticky='nsew')
        actions_scrollbar.grid(row=1, column=1, sticky='ns')

        self.update_action_suggestions()

    # Responsive helpers
    def _on_canvas_configure(self, event):
        # Ensure the scrollable frame matches the canvas width
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

    # Model / camera
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
            print("Camera + FaceMesh initialized")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.cap = None
            self.face_mesh = None

    # Label utilities 
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

    # Prediction (UNCHANGED) 
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

    # Loop (camera size effect RESTORED)
    def detect_emotions(self):
        while self.detection_active:
            if self.cap is None:
                break

            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            emotion, confidence = self.predict_emotion_from_frame(frame)
            cv2.putText(frame, f'{emotion}: {confidence:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Update emotion panel
            self.root.after(0, self.update_emotion_display, emotion, confidence)

            # Camera resizing effect fit frame to camera container with aspect preserved 
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

            frame_pil = Image.fromarray(frame_rgb).resize((new_width, new_height), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.root.after(0, self.update_video_display, frame_tk)

            time.sleep(0.03)  # ~33 FPS cap

    # UI update helpers 
    def update_video_display(self, frame_tk):
        self.video_label.configure(image=frame_tk)
        self.video_label.image = frame_tk

    def update_emotion_display(self, emotion, confidence):
        emotion = self._canonical_label(emotion)
        changed = (emotion != self.current_emotion)
        self.current_emotion = emotion
        self.emotion_confidence = confidence

        # Centered emoji and labels
        self.emotion_icon_label.configure(text=self.get_emotion_icon(emotion))
        self.emotion_text_label.configure(text=emotion.capitalize())
        self.confidence_label.configure(text=f"Confidence: {confidence:.1%}")

        if changed:
            self.update_action_suggestions()
        self.add_to_history(emotion, confidence)

    def add_to_history(self, emotion, confidence):
        for w in self.history_frame.winfo_children():
            w.destroy()
        history_text = f"{self.get_emotion_icon(emotion)} {emotion.capitalize()} - {confidence:.1%}"
        ttk.Label(self.history_frame, text=history_text, style='Dark.TLabel',
                  font=('Segoe UI', 10)).pack(anchor='w', pady=3)

    def update_action_suggestions(self):
        for w in self.actions_scrollable_frame.winfo_children():
            w.destroy()
        actions = self._actions_for(self.current_emotion)
        for text, func in actions:
            ttk.Button(self.actions_scrollable_frame, text=text, style='Dark.TButton',
                       command=func).pack(fill='x', pady=4, padx=6)
        self.actions_scrollable_frame.update_idletasks()
        self.actions_canvas.configure(scrollregion=self.actions_canvas.bbox("all"))
        self.actions_canvas.yview_moveto(0.0)

    # Small helpers
    def get_emotion_icon(self, emotion):
        icons = {'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😮'}
        return icons.get(emotion, '😐')

    def start_detection(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded")
            return
        if self.cap is None:
            messagebox.showerror("Error", "Camera not available")
            return
        self.detection_active = True
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        threading.Thread(target=self.detect_emotions, daemon=True).start()

    def stop_detection(self):
        self.detection_active = False
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')

    # Action methods
    # Happy
    def play_upbeat_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=upbeat+happy+music+playlist")
        messagebox.showinfo(" Music", "Opening upbeat music to match your mood!")
    def play_party_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=party+dance+music")
        messagebox.showinfo(" Party Time", "Let's get the party started!")
    def open_creative_apps(self):
        if platform.system() == "Windows":
            try: subprocess.Popen("mspaint.exe"); messagebox.showinfo(" Creative", "Opening Paint for you!")
            except: webbrowser.open("https://www.photopea.com"); messagebox.showinfo(" Creative", "Opening online creative tools!")
        else:
            webbrowser.open("https://www.photopea.com"); messagebox.showinfo(" Creative", "Opening creative tools for you!")
    def open_video_call(self):
        webbrowser.open("https://meet.google.com"); messagebox.showinfo(" Connect", "Share your happiness with friends!")
    def open_happy_journal(self):
        self.open_journal(); messagebox.showinfo(" Journal", "Document your happiness!")

    # Sad
    def play_comforting_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=comforting+peaceful+music")
        messagebox.showinfo(" Comfort", "Playing music to comfort you...")
    def watch_comedy(self):
        webbrowser.open("https://www.youtube.com/results?search_query=funny+comedy+videos+2024")
        messagebox.showinfo(" Comedy", "Let's lift your spirits with some laughter!")
    def open_mood_lifting(self):
        webbrowser.open("https://www.youtube.com/results?search_query=mood+lifting+feel+good+videos")
        messagebox.showinfo(" Feel Good", "Here's something to brighten your day!")
    def open_messaging(self):
        webbrowser.open("https://web.whatsapp.com"); messagebox.showinfo(" Connect", "Reach out to someone!")
    def play_healing_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=healing+music+emotional")
        messagebox.showinfo(" Healing", "Let this music help heal your heart...")
    def show_support_resources(self):
        messagebox.showinfo(" Support","You're not alone. Support is available:\n\n• Mental Health Hotline: 1-800-662-4357\n• Crisis Text Line: Text HOME to 741741\n• Suicide Prevention: 988\n")

    # Angry
    def play_calming_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=calming+meditation+music")
        messagebox.showinfo(" Calm", "Soothing sounds for inner peace...")
    def open_fitness_app(self):
        webbrowser.open("https://www.youtube.com/results?search_query=quick+workout+anger+relief")
        messagebox.showinfo(" Exercise", "Channel that energy into movement!")
    def open_stress_games(self):
        webbrowser.open("https://www.crazygames.com/t/stress-relief"); messagebox.showinfo(" Games", "Try these stress-relief games!")
    def open_stress_relief(self):
        webbrowser.open("https://www.youtube.com/results?search_query=virtual+stress+relief+activities"); messagebox.showinfo(" Relief", "Let it out in a healthy way!")
    def show_anger_tips(self):
        tips = ["Take 10 deep breaths slowly","Count backwards from 100","Go for a brisk walk","Write down what's bothering you","Listen to calming music","Do some intense exercise","Squeeze a stress ball","Step away from the situation"]
        messagebox.showinfo(" Anger Management", f"Try this:\n\n{np.random.choice(tips)}")
    def suggest_productive_activity(self):
        activities = ["Organize your workspace","Clean a room","Do a workout","Learn a new skill online","Work on a project","Plan tomorrow's tasks"]
        messagebox.showinfo(" Productive", f"Channel your energy:\n\n{np.random.choice(activities)}")
    def show_cooldown_tips(self):
        messagebox.showinfo(" Cool Down","• Splash cold water on your face\n• Take 5 slow, deep breaths\n• Count to 10 slowly\n• Step outside for fresh air\n• Drink cold water")
    def open_mood_tracker(self):
        self.open_journal(); messagebox.showinfo(" Track", "Document what triggered this feeling.")

    # Fear
    def show_safety_resources(self):
        messagebox.showinfo(" Safety", "Resources:\n• Crisis Helpline: 988\n• Emergency: 911\n• Crisis Text: HOME to 741741")
    def open_support_chat(self):
        webbrowser.open("https://www.7cups.com"); messagebox.showinfo(" Support", "Connect with trained listeners.")
    def play_anxiety_relief(self):
        webbrowser.open("https://www.youtube.com/results?search_query=anxiety+relief+calming+sounds"); messagebox.showinfo(" Calm", "Soothing sounds to ease anxiety...")
    def show_grounding_techniques(self):
        messagebox.showinfo(" Grounding", "5-4-3-2-1 Technique:\n5 see • 4 touch • 3 hear • 2 smell • 1 taste")
    def show_peace_guide(self):
        messagebox.showinfo(" Peace", "Focus on breathing. You are safe. This will pass.")

    # Surprise
    def open_video_recorder(self):
        if platform.system() == "Windows":
            try: subprocess.Popen("start microsoft.windows.camera:", shell=True); messagebox.showinfo(" Camera", "Capture this moment!")
            except: messagebox.showinfo(" Camera", "Open your camera app to record this moment!")
        else:
            messagebox.showinfo(" Camera", "Open your camera app to capture this moment!")
    def open_exciting_content(self):
        webbrowser.open("https://www.youtube.com/results?search_query=exciting+amazing+moments"); messagebox.showinfo(" Exciting", "More amazing content for you!")
    def show_reflection_prompt(self):
        prompts = ["What surprised you most?","How does this make you feel?","Who would you like to share this with?"]
        messagebox.showinfo(" Reflect", f"Take a moment:\n\n{np.random.choice(prompts)}")

    # Disgust
    def suggest_fresh_air(self):
        messagebox.showinfo(" Fresh Air","Step outside for 5 deep breaths and look at something green.")
    def show_cleansing_tips(self):
        tips = ["Take a refreshing shower","Open windows for fresh air","Organize your space","Change into clean clothes"]
        messagebox.showinfo(" Cleansing", f"Try this:\n\n{np.random.choice(tips)}")
    def show_comfort_recipes(self):
        webbrowser.open("https://www.youtube.com/results?search_query=comfort+food+recipes"); messagebox.showinfo(" Comfort", "Find something comforting to make!")
    def open_art_therapy(self):
        webbrowser.open("https://www.youtube.com/results?search_query=art+therapy+relaxing"); messagebox.showinfo(" Art Therapy", "Express yourself through art!")
    def show_cleansing_visualization(self):
        messagebox.showinfo(" Visualization","Imagine a waterfall washing away the negativity.")
    def show_space_reset_tips(self):
        messagebox.showinfo(" Space Reset","Open windows, clear clutter, play uplifting music.")

    # Neutral / Common
    def discover_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=music+discovery+mix"); messagebox.showinfo(" Discover", "Find new music you'll love!")
    def open_learning_resources(self):
        webbrowser.open("https://www.coursera.org"); messagebox.showinfo(" Learn", "Explore free courses.")
    def open_productivity(self):
        webbrowser.open("https://todoist.com"); messagebox.showinfo(" Productivity", "Get organized.")
    def explore_interests(self):
        webbrowser.open("https://www.youtube.com/"); messagebox.showinfo(" Explore", "Discover something new today!")
    def open_planner(self):
        try:
            if platform.system() == "Windows": subprocess.Popen("notepad.exe")
            elif platform.system() == "Darwin": subprocess.Popen(["open", "-a", "TextEdit"])
            else: subprocess.Popen(["gedit"])
            messagebox.showinfo(" Planning", "Plan your day ahead!")
        except: messagebox.showinfo(" Planning", "Open your favorite note app to plan your day!")
    def open_brain_games(self):
        webbrowser.open("https://www.lumosity.com"); messagebox.showinfo(" Brain Games", "Challenge your mind!")
    def open_reading(self):
        webbrowser.open("https://medium.com"); messagebox.showinfo(" Reading", "Discover interesting articles.")
    def open_goal_setting(self):
        self.open_journal(); messagebox.showinfo(" Goals", "Write down your goals and aspirations.")
    def play_relaxing_music(self):
        webbrowser.open("https://www.youtube.com/results?search_query=relaxing+calm+music"); messagebox.showinfo(" Relax", "Peaceful music to calm your mind!")
    def play_nature_sounds(self):
        webbrowser.open("https://www.youtube.com/results?search_query=nature+sounds+rain+forest"); messagebox.showinfo(" Nature", "Immerse yourself in nature sounds!")
    def open_games(self):
        if platform.system() == "Windows":
            try: subprocess.Popen("start steam://", shell=True); messagebox.showinfo(" Games", "Opening Steam!")
            except: webbrowser.open("https://www.crazygames.com"); messagebox.showinfo(" Games", "Time for some fun online games!")
        else:
            webbrowser.open("https://www.crazygames.com"); messagebox.showinfo(" Games", "Time for some fun!")
    def open_youtube(self):
        webbrowser.open("https://www.youtube.com"); messagebox.showinfo(" YouTube", "Explore videos that interest you!")
    def open_social_media(self):
        webbrowser.open("https://www.twitter.com"); messagebox.showinfo(" Social", "Connect with your network!")
    def open_camera_app(self):
        if platform.system() == "Windows":
            try: subprocess.Popen("start microsoft.windows.camera:", shell=True); messagebox.showinfo(" Camera", "Camera opened!")
            except: messagebox.showinfo(" Camera", "Open your camera app to capture moments!")
        elif platform.system() == "Darwin":
            messagebox.showinfo(" Camera", "Open Photo Booth to capture moments!")
        else:
            messagebox.showinfo(" Camera", "Open your camera app to capture moments!")
    def show_selfcare_tips(self):
        tips = ["Take a warm bath","Deep breathing 5 minutes","Go for a peaceful walk","Call a friend","Write in a journal","Gentle stretching","Make a comforting drink","Read a chapter of a book"]
        messagebox.showinfo(" Self-Care", f"Self-care tip:\n\n{np.random.choice(tips)}")
    def show_motivational_quotes(self):
        quotes = ["Every day is a new beginning!","You are stronger than you think!","This too shall pass.","Believe in yourself!","You've got this!","Progress, not perfection.","You are capable of amazing things!"]
        messagebox.showinfo(" Motivation", np.random.choice(quotes))
    def open_meditation(self):
        webbrowser.open("https://www.youtube.com/results?search_query=guided+meditation+10+minutes"); messagebox.showinfo(" Meditation", "Find inner peace with guided meditation!")
    def start_breathing_exercise(self):
        messagebox.showinfo(" Breathing Exercise","IN 4 • HOLD 4 • OUT 6 • HOLD 2\nRepeat 5–10 times.")
    def suggest_exercise(self):
        exercises = ["10-minute brisk walk","20 jumping jacks","5-minute yoga session","Dance to your favorite song","10 push-ups","Full-body stretch"]
        messagebox.showinfo(" Exercise", f"Try this:\n\n{np.random.choice(exercises)}")
    def open_journal(self):
        try:
            if platform.system() == "Windows": subprocess.Popen("notepad.exe")
            elif platform.system() == "Darwin": subprocess.Popen(["open", "-a", "TextEdit"])
            else: subprocess.Popen(["gedit"])
            messagebox.showinfo(" Journal", "Express your thoughts in writing!")
        except:
            webbrowser.open("https://docs.google.com"); messagebox.showinfo(" Journal", "Opening Google Docs for journaling!")
    def show_emergency_contacts(self):
        messagebox.showinfo(" Emergency Support","Help is available 24/7:\n• 988 (US)\n• 911 (Emergency)\n• Text HOME to 741741")
    def show_affirmations(self):
        affirmations = ["I am brave and capable.","I focus on what I can control.","I am safe right now.","I am resilient.","I deserve peace and happiness."]
        messagebox.showinfo(" Affirmation", np.random.choice(affirmations))
    def show_celebration_ideas(self):
        ideas = ["Share with friends","Treat yourself","Take a photo","Happy dance","Journal the moment"]
        messagebox.showinfo("Celebrate", f"Try this:\n\n{np.random.choice(ideas)}")
    def show_nature_content(self):
        webbrowser.open("https://www.youtube.com/results?search_query=beautiful+nature+scenery+4k+relaxing"); messagebox.showinfo(" Nature", "Immerse yourself in beautiful nature!")

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()


def main():
    root = tk.Tk()
    app = EmotionRecognitionApp(root)

    def on_closing():
        app.detection_active = False
        if hasattr(app, 'cap') and app.cap is not None:
            app.cap.release()
        if hasattr(app, 'face_mesh') and app.face_mesh is not None:
            app.face_mesh.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
