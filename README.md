# Moody
Emotional based recommendation which can control using hand gestures and voice commands

# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# voice model
pip install vosk
pip install setuptools>=75.0.0

# Run the app
python launcher/common_launcher.py

**Gesture Controls:**
- **Open palm (5 fingers)**: Toggle tracking ON/OFF
- **Thumb + Index pinch**: Left click (hold for drag)
- **Index + Pinky (rock sign)**: Right click
- **Index + Middle + Ring**: Scroll mode

# If PyAudio fails, install from wheel
pip install pipwin
pipwin install pyaudio

# voice model libraries
pip install setuptools
