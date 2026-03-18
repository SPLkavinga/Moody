"""
Centralized theme configuration for all modules
Provides dark and light themes with consistent color palettes
"""

import json
import os

# Theme definitions
THEMES = {
    "dark": {
        "bg_primary": "#1a1a1a",
        "bg_secondary": "#2a2a2a",
        "bg_tertiary": "#333333",
        "bg_hover": "#404040",
        "accent_primary": "#238636",
        "accent_secondary": "#2a4a7c",
        "accent_emotion": "#9d4edd",
        "accent_speech": "#2a9d8f",
        "accent_danger": "#da3633",
        "accent_warning": "#fb8500",
        "text_primary": "#ffffff",
        "text_secondary": "#c9d1d9",
        "text_muted": "#8b949e",
        "border": "#30363d",
    },
    "light": {
        "bg_primary": "#ffffff",
        "bg_secondary": "#f6f8fa",
        "bg_tertiary": "#e7e9eb",
        "bg_hover": "#d0d7de",
        "accent_primary": "#2da44e",
        "accent_secondary": "#0969da",
        "accent_emotion": "#8250df",
        "accent_speech": "#1a7f72",
        "accent_danger": "#cf222e",
        "accent_warning": "#d97706",
        "text_primary": "#1f2328",
        "text_secondary": "#57606a",
        "text_muted": "#8c959f",
        "border": "#d0d7de",
    }
}

# Path to store current theme preference
THEME_FILE = os.path.join(os.path.dirname(__file__), "current_theme.json")


def get_current_theme():
    """Get the currently active theme name (dark/light)"""
    try:
        if os.path.exists(THEME_FILE):
            with open(THEME_FILE, 'r') as f:
                data = json.load(f)
                return data.get('theme', 'dark')
    except Exception:
        pass
    return 'dark'  # Default to dark theme


def set_current_theme(theme_name):
    """Set the current theme (dark/light)"""
    if theme_name not in THEMES:
        raise ValueError(f"Invalid theme: {theme_name}. Must be 'dark' or 'light'")
    
    try:
        with open(THEME_FILE, 'w') as f:
            json.dump({'theme': theme_name}, f)
        return True
    except Exception as e:
        print(f"Error saving theme: {e}")
        return False


def get_theme_colors(theme_name=None):
    """Get color palette for specified theme or current theme"""
    if theme_name is None:
        theme_name = get_current_theme()
    
    if theme_name not in THEMES:
        theme_name = 'dark'
    
    return THEMES[theme_name].copy()


def toggle_theme():
    """Toggle between dark and light themes"""
    current = get_current_theme()
    new_theme = 'light' if current == 'dark' else 'dark'
    set_current_theme(new_theme)
    return new_theme
