"""
Moody Voice Assistant - Integrated voice control for the emotion module.
Wake word: "hey moody" or "moody"
Controls computer actions via voice commands with natural language understanding.
"""

import threading
import time
import os
import platform
import subprocess
import webbrowser
import warnings
import json
import re
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

warnings.filterwarnings('ignore')

# Speech Recognition and TTS
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from pynput.keyboard import Key, Controller as KeyboardController
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False



# Natural-language prefix,suffix strips

_NL_PREFIXES = [
    "can you", "could you", "would you", "will you",
    "please", "i want to", "i want you to", "i'd like to",
    "i'd like you to", "i need to", "i need you to",
    "go ahead and", "just", "kindly", "try to",
    "let's", "let me", "i wanna", "can we",
    "could we", "would you please", "can you please",
    "hey can you", "yo", "okay", "ok",
]
_NL_SUFFIXES = [
    "please", "now", "for me", "right now",
    "immediately", "quickly", "fast",
]

def _strip_natural_language(text: str) -> str:
    """Remove conversational filler so the core command remains."""
    t = text.lower().strip()
    changed = True
    while changed:
        changed = False
        for p in _NL_PREFIXES:
            if t.startswith(p + " "):
                t = t[len(p):].strip()
                changed = True
            elif t.startswith(p):
                t = t[len(p):].strip()
                changed = True
        for s in _NL_SUFFIXES:
            if t.endswith(" " + s):
                t = t[: -len(s)].strip()
                changed = True
    return t



# Common Speech-Recognition mis-hearings  intended word

_MISHEARD = {
    # minimize / minimise
    "minimise": "minimize", "minimalize": "minimize", "min eyes": "minimize",
    "minimum eyes": "minimize", "mini mice": "minimize", "minim eyes": "minimize",
    "mini mise": "minimize", "minimum ice": "minimize",
    # maximize / maximise
    "maximise": "maximize", "max eyes": "maximize", "maximum eyes": "maximize",
    "maxi mice": "maximize", "max mice": "maximize", "maximum ice": "maximize",
    # disable / dizable
    "dizable": "disable", "this able": "disable",
    "the sable": "disable", "does able": "disable",
    # enable
    "in able": "enable", "unable": "enable",
    # screenshot
    "screen short": "screenshot", "screen shot": "screenshot",
    "screen chart": "screenshot",
    # volume
    "wilume": "volume", "walume": "volume",
    # notepad
    "note pad": "notepad", "not pad": "notepad",
    # calculator
    "calcul8r": "calculator", "calculate her": "calculator",
    # browser
    "brows": "browser", "browzer": "browser",
    # YouTube
    "you tube": "youtube", "u tube": "youtube",
    # WhatsApp
    "what's up": "whatsapp", "whats app": "whatsapp",
    # close
    "clothes": "close", "kloz": "close",
    # open
    "upon": "open",
    # copy
    "kopy": "copy", "cappy": "copy",
    # paste
    "paced": "paste", "based": "paste", "pays": "paste",
    # undo
    "and do": "undo", "undone": "undo",
    # redo
    "read do": "redo", "ree do": "redo",
    # refresh
    "re fresh": "refresh",
    # escape
    "a scape": "escape", "a skip": "escape",
    # delete
    "the leet": "delete", "delight": "delete",
    # scroll
    "school": "scroll", "skrol": "scroll",
    # backspace
    "back space": "backspace",
    # mouse
    "mause": "mouse", "mouth": "mouse", "moose": "mouse",
    # gesture
    "just sure": "gesture", "just your": "gesture", "jester": "gesture",
    "chester": "gesture",
    # window
    "win doh": "window", "vindow": "window",
    # task
    "tusk": "task",
    # shutdown
    "shut down": "shutdown",
    # restart
    "re start": "restart",
    # lock
    "lok": "lock",
    # mute
    "moot": "mute", "mewed": "mute",
    # pause
    "paws": "pause", "pores": "pause",
    # next
    "necks": "next",
    # previous
    "previous lee": "previously",
    # paint
    "pint": "paint",
    # snipping
    "sniping": "snipping", "snip in": "snipping",
    # terminal
    "the terminal": "terminal",
    # command prompt
    "command promt": "command prompt",
    # explorer
    "file explore": "file explorer", "explore her": "explorer",
}

def _fix_misheard(text: str) -> str:
    """Replace common mis-heard words in-place."""
    result = text.lower()
    for wrong, right in sorted(_MISHEARD.items(), key=lambda x: len(x[0]), reverse=True):
        if wrong in result:
            result = result.replace(wrong, right)
    return result



# Fuzzy matching helper

def _fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()



# Voice Command Registry

class CommandRegistry:
    """Registry of all voice commands the assistant can handle."""

    def __init__(self, assistant, gesture_toggle_callback=None):
        self.assistant = assistant
        self.gesture_toggle_callback = gesture_toggle_callback
        self.commands = {}
        self._alias_map = {}
        self._register_all()

    # Registration helpers
    def _add(self, trigger, handler, response, takes_query=False, aliases=None):
        trigger = trigger.lower().strip()
        self.commands[trigger] = {
            'handler': handler,
            'response': response,
            'takes_query': takes_query,
        }
        if aliases:
            for a in aliases:
                self._alias_map[a.lower().strip()] = trigger

    def _register_all(self):

        # SYSTEM / OS
        self._add("open notepad", self._open_notepad, "Opening Notepad",
                   aliases=["launch notepad", "start notepad", "run notepad", "notepad open"])
        self._add("close notepad", lambda: self._close_app("notepad"), "Closing Notepad",
                   aliases=["kill notepad", "exit notepad", "shut notepad"])
        self._add("open calculator", self._open_calculator, "Opening Calculator",
                   aliases=["launch calculator", "start calculator", "open calc", "launch calc", "calc"])
        self._add("close calculator", lambda: self._close_app("CalculatorApp"), "Closing Calculator",
                   aliases=["kill calculator", "exit calculator", "close calc"])
        self._add("open file explorer", self._open_explorer, "Opening File Explorer",
                   aliases=["open explorer", "open my files", "open files", "show files",
                            "file manager", "open my computer", "open this pc", "show my files"])
        self._add("open task manager", self._open_task_manager, "Opening Task Manager",
                   aliases=["task manager", "show task manager", "launch task manager",
                            "show running apps", "show processes"])
        self._add("open settings", self._open_settings, "Opening Settings",
                   aliases=["open system settings", "system settings", "windows settings",
                            "launch settings", "show settings", "open preferences"])
        self._add("open control panel", self._open_control_panel, "Opening Control Panel",
                   aliases=["control panel", "launch control panel"])
        self._add("open command prompt", self._open_cmd, "Opening Command Prompt",
                   aliases=["open terminal", "open cmd", "launch terminal", "launch cmd",
                            "run cmd", "command line", "open powershell", "launch powershell",
                            "open console", "start terminal"])
        self._add("open paint", self._open_paint, "Opening Paint",
                   aliases=["launch paint", "start paint", "run paint", "open ms paint"])
        self._add("open snipping tool", self._open_snipping_tool, "Opening Snipping Tool",
                   aliases=["snipping tool", "launch snipping tool", "snip tool",
                            "screen capture tool", "open screen capture"])
        self._add("open word", self._open_word, "Opening Microsoft Word",
                   aliases=["launch word", "start word", "open microsoft word", "ms word"])
        self._add("open excel", self._open_excel, "Opening Microsoft Excel",
                   aliases=["launch excel", "start excel", "open microsoft excel", "ms excel"])
        self._add("open powerpoint", self._open_powerpoint, "Opening PowerPoint",
                   aliases=["launch powerpoint", "start powerpoint", "open ppt", "ms powerpoint"])
        self._add("open photos", self._open_photos, "Opening Photos",
                   aliases=["open photo app", "show photos", "launch photos"])
        self._add("open camera", self._open_camera, "Opening Camera",
                   aliases=["launch camera", "start camera", "turn on camera", "open webcam"])
        self._add("open clock", self._open_clock, "Opening Clock",
                   aliases=["open alarm", "open timer", "launch clock"])
        self._add("open maps", self._open_maps, "Opening Maps",
                   aliases=["launch maps", "show maps", "open google maps"])
        self._add("open store", self._open_store, "Opening Microsoft Store",
                   aliases=["open microsoft store", "app store", "open app store"])
        self._add("open downloads", self._open_downloads, "Opening Downloads folder",
                   aliases=["show downloads", "go to downloads", "downloads folder"])
        self._add("open desktop", self._open_desktop, "Opening Desktop folder",
                   aliases=["show desktop folder", "go to desktop"])
        self._add("open documents", self._open_documents, "Opening Documents folder",
                   aliases=["show documents", "go to documents", "documents folder", "my documents"])
        self._add("open recycle bin", self._open_recycle_bin, "Opening Recycle Bin",
                   aliases=["recycle bin", "show recycle bin", "open trash", "show trash"])
        self._add("empty recycle bin", self._empty_recycle_bin, "Emptying Recycle Bin",
                   aliases=["empty trash", "clear recycle bin", "clean recycle bin"])
        self._add("open device manager", self._open_device_manager, "Opening Device Manager",
                   aliases=["device manager", "show devices"])
        self._add("open disk management", self._open_disk_mgmt, "Opening Disk Management",
                   aliases=["disk management", "manage disks"])
        self._add("open event viewer", self._open_event_viewer, "Opening Event Viewer",
                   aliases=["event viewer", "show event logs"])

        # BROWSER and WEBSITES 
        self._add("open browser", lambda: webbrowser.open("https://www.google.com"), "Opening browser",
                   aliases=["launch browser", "start browser", "open chrome", "open edge",
                            "open firefox", "open internet", "go online"])
        self._add("open google", lambda: webbrowser.open("https://www.google.com"), "Opening Google",
                   aliases=["go to google", "launch google", "google.com"])
        self._add("open youtube", lambda: webbrowser.open("https://www.youtube.com"), "Opening YouTube",
                   aliases=["go to youtube", "launch youtube", "youtube.com"])
        self._add("open gmail", lambda: webbrowser.open("https://mail.google.com"), "Opening Gmail",
                   aliases=["go to gmail", "check email", "check mail", "open email",
                            "open mail", "launch gmail", "open my email", "open my mail"])
        self._add("open github", lambda: webbrowser.open("https://github.com"), "Opening GitHub",
                   aliases=["go to github", "launch github"])
        self._add("open spotify", lambda: webbrowser.open("https://open.spotify.com"), "Opening Spotify",
                   aliases=["launch spotify", "play spotify", "go to spotify"])
        self._add("open netflix", lambda: webbrowser.open("https://www.netflix.com"), "Opening Netflix",
                   aliases=["go to netflix", "launch netflix", "watch netflix"])
        self._add("open facebook", lambda: webbrowser.open("https://www.facebook.com"), "Opening Facebook",
                   aliases=["go to facebook", "launch facebook"])
        self._add("open instagram", lambda: webbrowser.open("https://www.instagram.com"), "Opening Instagram",
                   aliases=["go to instagram", "launch instagram"])
        self._add("open twitter", lambda: webbrowser.open("https://twitter.com"), "Opening Twitter",
                   aliases=["go to twitter", "launch twitter", "open x"])
        self._add("open whatsapp", lambda: webbrowser.open("https://web.whatsapp.com"), "Opening WhatsApp Web",
                   aliases=["go to whatsapp", "launch whatsapp", "open whatsapp web"])
        self._add("open linkedin", lambda: webbrowser.open("https://www.linkedin.com"), "Opening LinkedIn",
                   aliases=["go to linkedin", "launch linkedin"])
        self._add("open reddit", lambda: webbrowser.open("https://www.reddit.com"), "Opening Reddit",
                   aliases=["go to reddit", "launch reddit"])
        self._add("open stack overflow", lambda: webbrowser.open("https://stackoverflow.com"), "Opening Stack Overflow",
                   aliases=["go to stack overflow", "stackoverflow"])
        self._add("open chatgpt", lambda: webbrowser.open("https://chat.openai.com"), "Opening ChatGPT",
                   aliases=["go to chatgpt", "launch chatgpt", "open chat gpt", "open ai chat"])
        self._add("open amazon", lambda: webbrowser.open("https://www.amazon.com"), "Opening Amazon",
                   aliases=["go to amazon", "launch amazon"])
        self._add("open ebay", lambda: webbrowser.open("https://www.ebay.com"), "Opening eBay",
                   aliases=["go to ebay"])
        self._add("open wikipedia", lambda: webbrowser.open("https://www.wikipedia.org"), "Opening Wikipedia",
                   aliases=["go to wikipedia"])
        self._add("open twitch", lambda: webbrowser.open("https://www.twitch.tv"), "Opening Twitch",
                   aliases=["go to twitch", "launch twitch"])
        self._add("open pinterest", lambda: webbrowser.open("https://www.pinterest.com"), "Opening Pinterest",
                   aliases=["go to pinterest"])
        self._add("open tiktok", lambda: webbrowser.open("https://www.tiktok.com"), "Opening TikTok",
                   aliases=["go to tiktok", "launch tiktok"])
        self._add("open google drive", lambda: webbrowser.open("https://drive.google.com"), "Opening Google Drive",
                   aliases=["go to google drive", "open my drive", "open drive"])
        self._add("open google docs", lambda: webbrowser.open("https://docs.google.com"), "Opening Google Docs",
                   aliases=["go to google docs"])
        self._add("open google sheets", lambda: webbrowser.open("https://sheets.google.com"), "Opening Google Sheets",
                   aliases=["go to google sheets"])
        self._add("open google maps", lambda: webbrowser.open("https://maps.google.com"), "Opening Google Maps",
                   aliases=["go to google maps"])
        self._add("open google translate", lambda: webbrowser.open("https://translate.google.com"), "Opening Google Translate",
                   aliases=["translate", "open translator", "google translate"])
        self._add("open google calendar", lambda: webbrowser.open("https://calendar.google.com"), "Opening Google Calendar",
                   aliases=["go to calendar", "open calendar", "show calendar", "my calendar"])
        self._add("open outlook", lambda: webbrowser.open("https://outlook.live.com"), "Opening Outlook",
                   aliases=["go to outlook", "launch outlook"])
        self._add("open zoom", lambda: webbrowser.open("https://zoom.us"), "Opening Zoom",
                   aliases=["launch zoom", "open zoom meeting"])
        self._add("open discord", lambda: webbrowser.open("https://discord.com/app"), "Opening Discord",
                   aliases=["launch discord", "go to discord"])
        self._add("open telegram", lambda: webbrowser.open("https://web.telegram.org"), "Opening Telegram",
                   aliases=["launch telegram", "go to telegram"])

        # SEARCH
        self._add("search for", self._search_google, "Searching Google", takes_query=True,
                   aliases=["search", "google search", "look up", "find", "search google for",
                            "search the web for", "web search", "look for"])
        self._add("google", self._search_google, "Searching Google", takes_query=True)
        self._add("search youtube for", self._search_youtube, "Searching YouTube", takes_query=True,
                   aliases=["youtube search", "find on youtube", "look up on youtube",
                            "search youtube", "find video"])
        self._add("play on youtube", self._search_youtube, "Playing on YouTube", takes_query=True,
                   aliases=["play video", "play youtube"])
        self._add("open website", self._open_website, "Opening website", takes_query=True,
                   aliases=["go to website", "visit website", "visit"])

        #  VOLUME 
        self._add("volume up", self._volume_up, "Turning volume up",
                   aliases=["increase volume", "raise volume", "louder", "turn up volume",
                            "make it louder", "sound up", "turn up the volume",
                            "volume louder", "more volume", "pump up the volume"])
        self._add("volume down", self._volume_down, "Turning volume down",
                   aliases=["decrease volume", "lower volume", "quieter", "turn down volume",
                            "make it quieter", "sound down", "turn down the volume",
                            "volume quieter", "less volume", "reduce volume"])
        self._add("mute", self._mute, "Toggling mute",
                   aliases=["unmute", "toggle mute", "mute sound", "silence", "shut up sound",
                            "mute audio", "unmute audio", "mute volume", "unmute volume"])
        self._add("set volume to", self._set_volume, "Setting volume", takes_query=True,
                   aliases=["volume to", "change volume to", "volume level"])
        self._add("volume max", self._volume_max, "Setting volume to maximum",
                   aliases=["maximum volume", "full volume", "volume 100", "max volume"])
        self._add("volume min", self._volume_min, "Setting volume to minimum",
                   aliases=["minimum volume", "volume 0", "min volume"])

        # MEDIA 
        self._add("play music", self._media_play_pause, "Playing music",
                   aliases=["resume", "resume music", "resume playback",
                            "continue playing", "start playing", "unpause"])
        self._add("pause music", self._media_play_pause, "Pausing music",
                   aliases=["pause", "pause playback", "stop playing"])
        self._add("play", self._media_play_pause, "Toggling play/pause")
        self._add("next track", self._media_next, "Next track",
                   aliases=["next song", "skip", "skip song", "skip track",
                            "play next", "next one", "skip this"])
        self._add("previous track", self._media_prev, "Previous track",
                   aliases=["previous song", "go back", "last song", "play previous",
                            "previous one", "back one"])
        self._add("stop music", self._media_stop, "Stopping music",
                   aliases=["stop playback", "stop audio", "stop playing music"])

        # SCREENSHOT
        self._add("take screenshot", self._take_screenshot, "Taking screenshot",
                   aliases=["screenshot", "take a screenshot", "capture screen",
                            "screen capture", "print screen", "snap screen",
                            "grab screen", "save screenshot", "capture my screen"])

        # WINDOW MANAGEMENT
        self._add("minimize window", self._minimize_window, "Minimizing window",
                   aliases=["minimize", "minimise", "minimise window", "minimize this",
                            "hide window", "put window down", "shrink window",
                            "make it small", "window down", "minimize it"])
        self._add("maximize window", self._maximize_window, "Maximizing window",
                   aliases=["maximize", "maximise", "maximise window", "maximize this",
                            "full screen", "fullscreen", "make it big", "make it full",
                            "window up", "maximize it", "make it bigger", "expand window"])
        self._add("restore window", self._restore_window, "Restoring window",
                   aliases=["restore", "unmaximize", "restore down", "normal size",
                            "half screen", "exit full screen", "exit fullscreen"])
        self._add("close window", self._close_window, "Closing window",
                   aliases=["close this", "close it", "close app", "close application",
                            "exit app", "quit app", "close this window", "kill this"])
        self._add("alt tab", self._alt_tab, "Switching window",
                   aliases=["switch window", "switch app", "next window", "change window",
                            "switch to next", "toggle window", "swap window"])
        self._add("snap window left", self._snap_left, "Snapping window left",
                   aliases=["snap left", "window left", "move window left", "half left"])
        self._add("snap window right", self._snap_right, "Snapping window right",
                   aliases=["snap right", "window right", "move window right", "half right"])
        self._add("show desktop", self._show_desktop, "Showing desktop",
                   aliases=["minimize all", "minimise all", "hide all windows",
                            "go to desktop", "desktop"])
        self._add("close all windows", self._close_all_windows, "",
                   aliases=["close everything"])

        # TYPING
        self._add("type", self._type_text, "Typing text", takes_query=True,
                   aliases=["write", "type out", "type this", "write this", "input text",
                            "enter text"])

        # KEYBOARD SHORTCUTS
        self._add("copy", self._copy, "Copying",
                   aliases=["copy that", "copy this", "copy text", "copy it"])
        self._add("paste", self._paste, "Pasting",
                   aliases=["paste that", "paste this", "paste it", "paste text"])
        self._add("cut", self._cut, "Cutting",
                   aliases=["cut that", "cut this", "cut text", "cut it"])
        self._add("undo", self._undo, "Undoing",
                   aliases=["undo that", "undo it", "reverse that",
                            "take that back", "undo last"])
        self._add("redo", self._redo, "Redoing",
                   aliases=["redo that", "redo it", "redo last"])
        self._add("select all", self._select_all, "Selecting all",
                   aliases=["select everything", "highlight all", "select all text"])
        self._add("save", self._save, "Saving",
                   aliases=["save file", "save it", "save this", "save document",
                            "save changes", "save my work"])
        self._add("save as", self._save_as, "Opening Save As dialog",
                   aliases=["save file as", "save as new"])
        self._add("new tab", self._new_tab, "Opening new tab",
                   aliases=["open new tab", "add tab", "create new tab"])
        self._add("close tab", self._close_tab, "Closing tab",
                   aliases=["close this tab", "close current tab"])
        self._add("new window", self._new_window, "Opening new window",
                   aliases=["open new window"])
        self._add("reopen tab", self._reopen_tab, "Reopening closed tab",
                   aliases=["reopen closed tab", "restore tab", "bring back tab",
                            "undo close tab"])
        self._add("refresh", self._refresh, "Refreshing",
                   aliases=["reload", "reload page", "refresh page", "hard refresh"])
        self._add("zoom in", self._zoom_in, "Zooming in",
                   aliases=["make bigger", "make text bigger", "enlarge", "increase zoom"])
        self._add("zoom out", self._zoom_out, "Zooming out",
                   aliases=["make smaller", "make text smaller", "decrease zoom"])
        self._add("reset zoom", self._reset_zoom, "Resetting zoom",
                   aliases=["normal zoom", "zoom 100", "default zoom"])
        self._add("find", self._find, "Opening Find dialog",
                   aliases=["find text", "search text", "control f", "find on page",
                            "search in page", "find in page"])
        self._add("find and replace", self._find_replace, "Opening Find and Replace",
                   aliases=["replace", "replace text", "search and replace"])
        self._add("print", self._print, "Opening Print dialog",
                   aliases=["print this", "print page", "print document"])

        # Key presses
        self._add("press enter", self._press_enter, "Pressing Enter",
                   aliases=["enter", "hit enter", "press return"])
        self._add("press escape", self._press_escape, "Pressing Escape",
                   aliases=["escape", "hit escape", "press esc", "esc"])
        self._add("press backspace", self._press_backspace, "Pressing Backspace",
                   aliases=["backspace", "back space", "erase"])
        self._add("press delete", self._press_delete, "Pressing Delete",
                   aliases=["delete", "press del", "del key"])
        self._add("press tab", self._press_tab, "Pressing Tab",
                   aliases=["tab key", "hit tab"])
        self._add("press space", self._press_space, "Pressing Space",
                   aliases=["space", "space bar", "spacebar", "hit space"])
        self._add("press home", self._press_home, "Pressing Home",
                   aliases=["home key"])
        self._add("press end", self._press_end, "Pressing End",
                   aliases=["end key"])
        self._add("press f5", self._refresh, "Pressing F5",
                   aliases=["f5"])
        self._add("press f11", self._press_f11, "Pressing F11",
                   aliases=["f11", "toggle fullscreen"])

        # SCROLL
        self._add("scroll up", self._scroll_up, "Scrolling up",
                   aliases=["scroll upward", "move up", "page scroll up"])
        self._add("scroll down", self._scroll_down, "Scrolling down",
                   aliases=["scroll downward", "move down", "page scroll down"])
        self._add("scroll left", self._scroll_left, "Scrolling left")
        self._add("scroll right", self._scroll_right, "Scrolling right")
        self._add("page up", self._page_up, "Page up",
                   aliases=["one page up", "scroll page up"])
        self._add("page down", self._page_down, "Page down",
                   aliases=["one page down", "scroll page down"])
        self._add("go to top", self._go_to_top, "Going to top",
                   aliases=["scroll to top", "top of page", "beginning of page",
                            "go to beginning", "scroll to beginning"])
        self._add("go to bottom", self._go_to_bottom, "Going to bottom",
                   aliases=["scroll to bottom", "bottom of page", "end of page",
                            "go to end", "scroll to end"])

        # MOUSE GESTURE CONTROL
        self._add("enable mouse", self._enable_gesture_mouse, "Enabling hand gesture mouse control",
                   aliases=["enable gestures", "enable gesture", "enable hand gestures",
                            "enable gesture control", "enable mouse control",
                            "turn on mouse", "turn on gestures", "turn on gesture control",
                            "activate mouse", "activate gestures", "start mouse control",
                            "start gesture", "start gestures", "start hand gestures",
                            "gesture on", "gestures on", "mouse on",
                            "enable hand mouse", "hand control on",
                            "enable hand control", "switch on mouse",
                            "switch on gestures", "mouse control on"])
        self._add("disable mouse", self._disable_gesture_mouse, "Disabling hand gesture mouse control",
                   aliases=["disable gestures", "disable gesture", "disable hand gestures",
                            "disable gesture control", "disable mouse control",
                            "turn off mouse", "turn off gestures", "turn off gesture control",
                            "deactivate mouse", "deactivate gestures", "stop mouse control",
                            "stop gesture", "stop gestures", "stop hand gestures",
                            "gesture off", "gestures off", "mouse off",
                            "disable hand mouse", "hand control off",
                            "disable hand control", "switch off mouse",
                            "switch off gestures", "mouse control off"])
        self._add("mouse click", self._mouse_click, "Clicking",
                   aliases=["click", "left click", "single click", "click here"])
        self._add("double click", self._double_click, "Double clicking",
                   aliases=["double click here"])
        self._add("right click", self._right_click, "Right clicking",
                   aliases=["right click here", "context menu", "show menu"])
        self._add("mouse move up", self._mouse_move_up, "Moving mouse up",
                   aliases=["move mouse up", "cursor up"])
        self._add("mouse move down", self._mouse_move_down, "Moving mouse down",
                   aliases=["move mouse down", "cursor down"])
        self._add("mouse move left", self._mouse_move_left, "Moving mouse left",
                   aliases=["move mouse left", "cursor left"])
        self._add("mouse move right", self._mouse_move_right, "Moving mouse right",
                   aliases=["move mouse right", "cursor right"])

        # SYSTEM ACTIONS
        self._add("lock screen", self._lock_screen, "Locking screen",
                   aliases=["lock computer", "lock my computer", "lock this computer",
                            "lock pc", "lock my pc", "lock it"])
        self._add("shutdown computer", self._shutdown, "",
                   aliases=["shut down", "shutdown", "turn off computer", "turn off pc",
                            "power off", "shut down computer"])
        self._add("restart computer", self._restart, "",
                   aliases=["restart", "reboot", "restart pc", "reboot computer"])
        self._add("sleep computer", self._sleep_computer, "Putting computer to sleep",
                   aliases=["sleep mode", "put to sleep", "hibernate",
                            "computer sleep", "pc sleep"])
        self._add("increase brightness", self._brightness_up, "Increasing brightness",
                   aliases=["brightness up", "brighter", "more brightness",
                            "turn up brightness", "raise brightness"])
        self._add("decrease brightness", self._brightness_down, "Decreasing brightness",
                   aliases=["brightness down", "dimmer", "less brightness",
                            "turn down brightness", "lower brightness", "dim screen"])
        self._add("open wifi settings", self._open_wifi, "Opening WiFi settings",
                   aliases=["wifi settings", "open wifi", "show wifi", "connect wifi",
                            "turn on wifi", "network settings"])
        self._add("open bluetooth settings", self._open_bluetooth, "Opening Bluetooth settings",
                   aliases=["bluetooth settings", "open bluetooth", "show bluetooth",
                            "connect bluetooth", "turn on bluetooth"])
        self._add("open display settings", self._open_display_settings, "Opening Display settings",
                   aliases=["display settings", "screen settings", "resolution settings"])
        self._add("open sound settings", self._open_sound_settings, "Opening Sound settings",
                   aliases=["sound settings", "audio settings"])
        self._add("battery status", self._battery_status, "",
                   aliases=["battery level", "battery percentage", "how much battery",
                            "check battery", "battery life"])

        # DATE and TIME 
        self._add("what time is it", self._tell_time, "",
                   aliases=["what's the time", "tell me the time", "current time",
                            "time now", "what time", "time please", "show time"])
        self._add("what date is it", self._tell_date, "",
                   aliases=["what's the date", "what day is it", "tell me the date",
                            "current date", "today's date", "what date", "date please",
                            "what is today"])
        self._add("set timer for", self._set_timer, "Setting timer", takes_query=True,
                   aliases=["timer for", "start timer for", "countdown"])
        self._add("set alarm for", self._set_alarm, "Setting alarm", takes_query=True,
                   aliases=["alarm for", "wake me up at"])

        # MOODY ASSISTANT 
        self._add("go to sleep", self._assistant_sleep, "Going to sleep. Say 'Hey Moody' to wake me up.",
                   aliases=["stop listening", "sleep now", "go sleep", "bye", "goodbye",
                            "see you later", "that's all", "that is all", "i'm done",
                            "no more commands"])
        self._add("thank you", lambda: None, "You're welcome!",
                   aliases=["thanks", "thank you moody", "thanks moody"])
        self._add("hello", lambda: None, "Hello! How can I help you?",
                   aliases=["hi", "hey", "hi moody", "hello moody", "howdy",
                            "good morning", "good afternoon", "good evening"])
        self._add("how are you", lambda: None, "I'm doing great! Ready to help you.",
                   aliases=["how are you doing", "how do you feel", "are you okay",
                            "how's it going"])
        self._add("what's your name", lambda: None, "I'm Moody, your voice assistant!",
                   aliases=["who are you", "your name", "what are you called"])
        self._add("what can you do", self._show_help, "",
                   aliases=["help", "show help", "help me", "list commands",
                            "show commands", "what are your commands", "what do you do"])
        self._add("tell me a joke", self._tell_joke, "",
                   aliases=["joke", "say something funny", "make me laugh"])
        self._add("motivate me", self._motivate, "",
                   aliases=["motivation", "inspire me", "give me motivation",
                            "say something inspiring", "motivational quote"])

        # CLOSE SPECIFIC APPS
        self._add("close chrome", lambda: self._close_app("chrome"), "Closing Chrome",
                   aliases=["kill chrome", "exit chrome"])
        self._add("close edge", lambda: self._close_app("msedge"), "Closing Edge",
                   aliases=["kill edge", "exit edge"])
        self._add("close firefox", lambda: self._close_app("firefox"), "Closing Firefox",
                   aliases=["kill firefox", "exit firefox"])
        self._add("close word", lambda: self._close_app("WINWORD"), "Closing Word",
                   aliases=["kill word", "exit word"])
        self._add("close excel", lambda: self._close_app("EXCEL"), "Closing Excel",
                   aliases=["kill excel", "exit excel"])
        self._add("close powerpoint", lambda: self._close_app("POWERPNT"), "Closing PowerPoint",
                   aliases=["kill powerpoint", "exit powerpoint"])
        self._add("close vlc", lambda: self._close_app("vlc"), "Closing VLC",
                   aliases=["kill vlc", "exit vlc"])

 
    #  MATCHING ENGINE

    def match(self, text):
        """Smart matching: normalisation -> aliases -> fuzzy fallback."""
        raw = text.lower().strip()

        # Fix commonly misheard words
        normalised = _fix_misheard(raw)

        # Strip natural-language fluff
        core = _strip_natural_language(normalised)

        # Exact substring on canonical triggers (longest first)
        result = self._exact_match(core)
        if result[0]:
            return result

        # Exact substring on aliases
        result = self._alias_match(core)
        if result[0]:
            return result

        # Try on normalised but unstripped version
        if core != normalised:
            result = self._exact_match(normalised)
            if result[0]:
                return result
            result = self._alias_match(normalised)
            if result[0]:
                return result

        # Try on unprocessed raw text (handles edge cases)
        if raw != normalised:
            result = self._exact_match(raw)
            if result[0]:
                return result
            result = self._alias_match(raw)
            if result[0]:
                return result

        # Fuzzy match (threshold 0.62)
        result = self._fuzzy_match(core)
        if result[0]:
            return result

        return None, ""

    def _exact_match(self, text):
        sorted_triggers = sorted(self.commands.keys(), key=len, reverse=True)
        for trigger in sorted_triggers:
            if text == trigger or text.startswith(trigger + " ") or text.startswith(trigger) or trigger in text:
                cmd = self.commands[trigger]
                query = ""
                if cmd['takes_query']:
                    idx = text.find(trigger)
                    query = text[idx + len(trigger):].strip()
                return cmd, query
        return None, ""

    def _alias_match(self, text):
        sorted_aliases = sorted(self._alias_map.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if text == alias or text.startswith(alias + " ") or text.startswith(alias) or alias in text:
                trigger = self._alias_map[alias]
                cmd = self.commands[trigger]
                query = ""
                if cmd['takes_query']:
                    idx = text.find(alias)
                    query = text[idx + len(alias):].strip()
                return cmd, query
        return None, ""

    def _fuzzy_match(self, text, threshold=0.62):
        best_score = 0
        best_cmd = None
        best_query = ""

        for trigger, cmd in self.commands.items():
            score = _fuzzy_score(text, trigger)
            words_t = text.split()
            words_tr = trigger.split()
            if len(words_t) >= len(words_tr):
                partial = " ".join(words_t[:len(words_tr)])
                score = max(score, _fuzzy_score(partial, trigger))
            if score > best_score:
                best_score = score
                best_cmd = cmd
                if cmd['takes_query']:
                    best_query = text[len(trigger):].strip()

        for alias, trigger in self._alias_map.items():
            score = _fuzzy_score(text, alias)
            words_t = text.split()
            words_a = alias.split()
            if len(words_t) >= len(words_a):
                partial = " ".join(words_t[:len(words_a)])
                score = max(score, _fuzzy_score(partial, alias))
            if score > best_score:
                best_score = score
                cmd = self.commands[trigger]
                best_cmd = cmd
                if cmd['takes_query']:
                    best_query = text[len(alias):].strip()

        if best_score >= threshold and best_cmd:
            return best_cmd, best_query
        return None, ""


    #  COMMAND HANDLERS
    

    # App Launchers
    def _open_notepad(self):
        if platform.system() == "Windows":
            subprocess.Popen(["notepad.exe"])
        else:
            subprocess.Popen(["gedit"])

    def _open_calculator(self):
        if platform.system() == "Windows":
            subprocess.Popen(["calc.exe"])
        else:
            subprocess.Popen(["gnome-calculator"])

    def _open_explorer(self):
        if platform.system() == "Windows":
            subprocess.Popen(["explorer.exe"])
        else:
            subprocess.Popen(["nautilus"])

    def _open_task_manager(self):
        if platform.system() == "Windows":
            subprocess.Popen(["taskmgr.exe"])

    def _open_settings(self):
        if platform.system() == "Windows":
            subprocess.Popen(["start", "ms-settings:"], shell=True)

    def _open_control_panel(self):
        if platform.system() == "Windows":
            subprocess.Popen(["control.exe"])

    def _open_cmd(self):
        if platform.system() == "Windows":
            subprocess.Popen(["cmd.exe"])
        else:
            subprocess.Popen(["gnome-terminal"])

    def _open_paint(self):
        if platform.system() == "Windows":
            subprocess.Popen(["mspaint.exe"])

    def _open_snipping_tool(self):
        if platform.system() == "Windows":
            try:
                subprocess.Popen(["SnippingTool.exe"])
            except FileNotFoundError:
                subprocess.Popen(["start", "ms-screenclip:"], shell=True)

    def _open_word(self):
        if platform.system() == "Windows":
            try:
                subprocess.Popen(["start", "winword"], shell=True)
            except Exception:
                self.assistant.speak("Microsoft Word doesn't seem to be installed.")

    def _open_excel(self):
        if platform.system() == "Windows":
            try:
                subprocess.Popen(["start", "excel"], shell=True)
            except Exception:
                self.assistant.speak("Microsoft Excel doesn't seem to be installed.")

    def _open_powerpoint(self):
        if platform.system() == "Windows":
            try:
                subprocess.Popen(["start", "powerpnt"], shell=True)
            except Exception:
                self.assistant.speak("PowerPoint doesn't seem to be installed.")

    def _open_photos(self):
        if platform.system() == "Windows":
            subprocess.Popen(["start", "ms-photos:"], shell=True)

    def _open_camera(self):
        if platform.system() == "Windows":
            subprocess.Popen(["start", "microsoft.windows.camera:"], shell=True)

    def _open_clock(self):
        if platform.system() == "Windows":
            subprocess.Popen(["start", "ms-clock:"], shell=True)

    def _open_maps(self):
        webbrowser.open("https://maps.google.com")

    def _open_store(self):
        if platform.system() == "Windows":
            subprocess.Popen(["start", "ms-windows-store:"], shell=True)

    def _open_downloads(self):
        path = os.path.join(os.path.expanduser("~"), "Downloads")
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _open_desktop(self):
        path = os.path.join(os.path.expanduser("~"), "Desktop")
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _open_documents(self):
        path = os.path.join(os.path.expanduser("~"), "Documents")
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _open_recycle_bin(self):
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", "shell:RecycleBinFolder"])

    def _empty_recycle_bin(self):
        if platform.system() == "Windows":
            try:
                subprocess.Popen(["PowerShell", "-Command", "Clear-RecycleBin -Force"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.assistant.speak("Recycle bin emptied.")
            except Exception:
                self.assistant.speak("Could not empty the recycle bin.")

    def _open_device_manager(self):
        if platform.system() == "Windows":
            subprocess.Popen(["devmgmt.msc"], shell=True)

    def _open_disk_mgmt(self):
        if platform.system() == "Windows":
            subprocess.Popen(["diskmgmt.msc"], shell=True)

    def _open_event_viewer(self):
        if platform.system() == "Windows":
            subprocess.Popen(["eventvwr.msc"], shell=True)

    def _close_app(self, name):
        if platform.system() == "Windows":
            subprocess.Popen(["taskkill", "/IM", f"{name}.exe", "/F"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Search  Website
    def _search_google(self, query=""):
        if query:
            webbrowser.open(f"https://www.google.com/search?q={query}")
        else:
            self.assistant.speak("What should I search for?")

    def _search_youtube(self, query=""):
        if query:
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
        else:
            self.assistant.speak("What should I search on YouTube?")

    def _open_website(self, query=""):
        if query:
            url = query.strip().replace(" ", "")
            if not url.startswith("http"):
                url = "https://" + url
            webbrowser.open(url)
        else:
            self.assistant.speak("Which website should I open?")

    # Volume
    def _volume_up(self):
        if PYAUTOGUI_AVAILABLE:
            for _ in range(5):
                pyautogui.press('volumeup')

    def _volume_down(self):
        if PYAUTOGUI_AVAILABLE:
            for _ in range(5):
                pyautogui.press('volumedown')

    def _mute(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('volumemute')

    def _set_volume(self, query=""):
        try:
            level = int(re.search(r'\d+', query).group())
            level = max(0, min(100, level))
        except Exception:
            self.assistant.speak("I didn't catch the volume level. Say a number from 0 to 100.")
            return
        if PYAUTOGUI_AVAILABLE:
            # Mute then unmute then raise to approximate level
            pyautogui.press('volumemute')
            time.sleep(0.1)
            pyautogui.press('volumemute')
            steps = level // 2
            for _ in range(steps):
                pyautogui.press('volumeup')
            self.assistant.speak(f"Volume set to approximately {level} percent.")

    def _volume_max(self):
        if PYAUTOGUI_AVAILABLE:
            for _ in range(50):
                pyautogui.press('volumeup')

    def _volume_min(self):
        if PYAUTOGUI_AVAILABLE:
            for _ in range(50):
                pyautogui.press('volumedown')

    # Media
    def _media_play_pause(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('playpause')

    def _media_next(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('nexttrack')

    def _media_prev(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('prevtrack')

    def _media_stop(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('stop')

    # Screenshot
    def _take_screenshot(self):
        if PYAUTOGUI_AVAILABLE:
            screenshot_dir = os.path.join(os.path.expanduser("~"), "Pictures", "Moody_Screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(screenshot_dir, filename)
            img = pyautogui.screenshot()
            img.save(filepath)
            self.assistant.speak(f"Screenshot saved as {filename}")

    # Window Management
    def _minimize_window(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey('win', 'down')
            time.sleep(0.15)
            pyautogui.hotkey('win', 'down')

    def _maximize_window(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey('win', 'up')

    def _restore_window(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey('win', 'down')

    def _close_window(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey('alt', 'F4')

    def _alt_tab(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey('alt', 'tab')

    def _snap_left(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey('win', 'left')

    def _snap_right(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey('win', 'right')

    def _show_desktop(self):
        if PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey('win', 'd')

    def _close_all_windows(self):
        self.assistant.speak("For safety, I won't close all windows. Please do that manually.")

    # Typing
    def _type_text(self, query=""):
        if query and PYAUTOGUI_AVAILABLE:
            pyautogui.typewrite(query, interval=0.03)
        elif not query:
            self.assistant.speak("What should I type?")

    # Keyboard Shortcuts
    def _copy(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'c')

    def _paste(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'v')

    def _cut(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'x')

    def _undo(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'z')

    def _redo(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'y')

    def _select_all(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'a')

    def _save(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 's')

    def _save_as(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'shift', 's')

    def _new_tab(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 't')

    def _close_tab(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'w')

    def _new_window(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'n')

    def _reopen_tab(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'shift', 't')

    def _refresh(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('f5')

    def _zoom_in(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'plus')

    def _zoom_out(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'minus')

    def _reset_zoom(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', '0')

    def _find(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'f')

    def _find_replace(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'h')

    def _print(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'p')

    def _press_enter(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('enter')

    def _press_escape(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('escape')

    def _press_backspace(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('backspace')

    def _press_delete(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('delete')

    def _press_tab(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('tab')

    def _press_space(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('space')

    def _press_home(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('home')

    def _press_end(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('end')

    def _press_f11(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('f11')

    # Scroll
    def _scroll_up(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.scroll(7)

    def _scroll_down(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.scroll(-7)

    def _scroll_left(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hscroll(-5)

    def _scroll_right(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hscroll(5)

    def _page_up(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('pageup')

    def _page_down(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.press('pagedown')

    def _go_to_top(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'Home')

    def _go_to_bottom(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.hotkey('ctrl', 'End')

    # Mouse Gesture
    def _enable_gesture_mouse(self):
        cb = self.gesture_toggle_callback
        if cb:
            cb("enable")
        else:
            self.assistant.speak("Gesture control is not available right now.")

    def _disable_gesture_mouse(self):
        cb = self.gesture_toggle_callback
        if cb:
            cb("disable")
        else:
            self.assistant.speak("Gesture control is not available right now.")

    def _mouse_click(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.click()

    def _double_click(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.doubleClick()

    def _right_click(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.rightClick()

    def _mouse_move_up(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.moveRel(0, -80, duration=0.2)

    def _mouse_move_down(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.moveRel(0, 80, duration=0.2)

    def _mouse_move_left(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.moveRel(-80, 0, duration=0.2)

    def _mouse_move_right(self):
        if PYAUTOGUI_AVAILABLE: pyautogui.moveRel(80, 0, duration=0.2)

    # System
    def _lock_screen(self):
        if platform.system() == "Windows":
            subprocess.Popen(["rundll32.exe", "user32.dll,LockWorkStation"])

    def _shutdown(self):
        self.assistant.speak("For safety, I won't shut down automatically. Please use the Start menu.")

    def _restart(self):
        self.assistant.speak("For safety, I won't restart automatically. Please use the Start menu.")

    def _sleep_computer(self):
        if platform.system() == "Windows":
            subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"])

    def _brightness_up(self):
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["powershell", "-Command",
                    "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, [Math]::Min(100, (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness + 10))"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            self.assistant.speak("Could not change brightness.")

    def _brightness_down(self):
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["powershell", "-Command",
                    "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, [Math]::Max(0, (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness - 10))"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            self.assistant.speak("Could not change brightness.")

    def _open_wifi(self):
        if platform.system() == "Windows":
            subprocess.Popen(["start", "ms-settings:network-wifi"], shell=True)

    def _open_bluetooth(self):
        if platform.system() == "Windows":
            subprocess.Popen(["start", "ms-settings:bluetooth"], shell=True)

    def _open_display_settings(self):
        if platform.system() == "Windows":
            subprocess.Popen(["start", "ms-settings:display"], shell=True)

    def _open_sound_settings(self):
        if platform.system() == "Windows":
            subprocess.Popen(["start", "ms-settings:sound"], shell=True)

    def _battery_status(self):
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery:
                pct = battery.percent
                plugged = "plugged in" if battery.power_plugged else "on battery"
                self.assistant.speak(f"Battery is at {pct} percent, {plugged}.")
            else:
                self.assistant.speak("Could not detect a battery. This might be a desktop computer.")
        except ImportError:
            self.assistant.speak("Battery info requires the psutil library.")
        except Exception:
            self.assistant.speak("Could not read battery status.")

    # Date and Time
    def _tell_time(self):
        now = datetime.now().strftime("%I:%M %p")
        self.assistant.speak(f"The current time is {now}")

    def _tell_date(self):
        now = datetime.now().strftime("%A, %B %d, %Y")
        self.assistant.speak(f"Today is {now}")

    def _set_timer(self, query=""):
        try:
            minutes = int(re.search(r'\d+', query).group())
        except Exception:
            self.assistant.speak("How many minutes?")
            return
        self.assistant.speak(f"Timer set for {minutes} minutes.")
        def _timer():
            time.sleep(minutes * 60)
            self.assistant.speak(f"Time's up! Your {minutes} minute timer is done.")
        threading.Thread(target=_timer, daemon=True).start()

    def _set_alarm(self, query=""):
        self.assistant.speak("Alarm feature is coming soon. I've noted your request!")

    # Moody specific
    def _assistant_sleep(self):
        self.assistant.set_awake(False)

    def _show_help(self):
        help_text = (
            "I can do lots of things! Here are some examples: "
            "Open apps like Notepad, Calculator, Word, Excel, or any browser. "
            "Search Google or YouTube for anything. "
            "Control volume: louder, quieter, mute, or set to a level. "
            "Play, pause, skip, or go back on music. "
            "Take screenshots saved to your Pictures folder. "
            "Type text, copy, paste, cut, undo, redo, select all, save. "
            "Scroll up, down, or jump to top or bottom. "
            "Minimize, maximize, close, or snap windows. "
            "Enable or disable hand gesture mouse control. "
            "Click, double click, or right click with mouse commands. "
            "Lock screen, check battery, change brightness. "
            "Tell you the time and date. Set timers. "
            "Open websites, WiFi, Bluetooth, or display settings. "
            "Just speak naturally: I understand phrases like "
            "'Can you open notepad please' or 'I want to search YouTube'. "
            "Say 'Hey Moody' followed by your command!"
        )
        self.assistant.speak(help_text)

    _JOKES = [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "Why was the computer cold? It left its Windows open!",
        "What's a computer's favorite snack? Microchips!",
        "Why did the developer go broke? Because he used up all his cache!",
        "How do trees access the internet? They log in!",
        "Why do Java developers wear glasses? Because they don't C sharp!",
    ]

    def _tell_joke(self):
        import random
        joke = random.choice(self._JOKES)
        self.assistant.speak(joke)

    _MOTIVATIONS = [
        "You're doing amazing! Keep pushing forward!",
        "Every expert was once a beginner. Keep going!",
        "Believe in yourself. You've got this!",
        "The only way to do great work is to love what you do.",
        "Don't watch the clock; do what it does: keep going!",
        "Success is not final, failure is not fatal. It is the courage to continue that counts.",
    ]

    def _motivate(self):
        import random
        quote = random.choice(self._MOTIVATIONS)
        self.assistant.speak(quote)



# Voice Assistant Engine

class MoodyVoiceAssistant:
    """Core voice assistant with wake word detection and command execution."""

    WAKE_WORDS = ["hey moody", "moody", "hay moody", "hey movie", "hey modi",
                  "a moody", "hey moti", "hey muddy", "he moody", "hey woody",
                  "hey buddy", "hey mobi", "hey mody", "hey mooney",
                  "hey money", "hey morning", "hey body"]

    def __init__(self, on_status_change=None, on_log=None, on_wake=None,
                 gesture_toggle_callback=None):
        self.on_status_change = on_status_change or (lambda s: None)
        self.on_log = on_log or (lambda t, tag: None)
        self.on_wake = on_wake or (lambda: None)
        self.gesture_toggle_callback = gesture_toggle_callback

        # State
        self.running = False
        self.listening = False
        self.awake = False
        self.background_mode = False
        self._lock = threading.Lock()
        self._listen_thread = None

        # Speech recognizer
        if SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.microphone = None
        else:
            self.recognizer = None
            self.microphone = None

        # TTS engine
        self._tts_engine = None
        self._tts_lock = threading.Lock()

        # Command registry
        self.command_registry = CommandRegistry(self, gesture_toggle_callback=gesture_toggle_callback)

        # Awake timeout
        self.awake_timeout = 45
        self._last_command_time = 0

        # Command history
        self.command_history = []

    # TTS
    def _get_tts(self):
        if self._tts_engine is None and TTS_AVAILABLE:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 170)
                self._tts_engine.setProperty('volume', 0.9)
                voices = self._tts_engine.getProperty('voices')
                for v in voices:
                    if 'female' in v.name.lower() or 'zira' in v.name.lower():
                        self._tts_engine.setProperty('voice', v.id)
                        break
            except Exception:
                self._tts_engine = None
        return self._tts_engine

    def speak(self, text):
        if not text:
            return
        self.on_log(f"Moody: {text}", "assistant")

        def _speak():
            with self._tts_lock:
                engine = self._get_tts()
                if engine:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception:
                        pass

        threading.Thread(target=_speak, daemon=True).start()

    # State
    def set_awake(self, awake):
        self.awake = awake
        if awake:
            self._last_command_time = time.time()
            self.on_status_change("Awake – Listening for commands...")
            self.on_wake()
        else:
            self.on_status_change("💤 Sleeping – Say 'Hey Moody' to wake")

    # Start Stop
    def start(self):
        if not SR_AVAILABLE:
            self.on_log("SpeechRecognition not installed. Install it with: pip install SpeechRecognition", "error")
            return False

        if self.running:
            return True

        self.running = True
        self.listening = True
        self.awake = False
        self._last_command_time = 0

        self.on_status_change(" Sleeping – Say 'Hey Moody' to wake")
        self.on_log("Voice assistant started. Say 'Hey Moody' to begin!", "system")

        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        return True

    def stop(self):
        self.running = False
        self.listening = False
        self.awake = False
        self.on_status_change("Voice assistant stopped")
        self.on_log("Voice assistant stopped.", "system")

    def toggle_background(self, enabled):
        self.background_mode = enabled
        if enabled:
            self.on_log("Background mode enabled – assistant will keep listening.", "system")
        else:
            self.on_log("Background mode disabled.", "system")

    # core Listening Loop 
    def _listen_loop(self):
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.on_log("Adjusting for ambient noise... please wait.", "system")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                self.on_log("Ready! Say 'Hey Moody' to wake me up.", "system")

            while self.running:
                try:
                    self._listen_once()
                except Exception as e:
                    if self.running:
                        self.on_log(f"Listening error: {str(e)}", "error")
                        time.sleep(1)

                if self.awake and self._last_command_time > 0:
                    if time.time() - self._last_command_time > self.awake_timeout:
                        self.set_awake(False)
                        self.speak("Going to sleep. Say 'Hey Moody' when you need me.")

                time.sleep(0.05)

        except Exception as e:
            self.on_log(f"Microphone error: {str(e)}", "error")
            self.on_status_change("Microphone not available")
            self.running = False

    def _listen_once(self):
        if not self.running:
            return

        try:
            with sr.Microphone() as source:
                if self.awake:
                    self.on_status_change("Listening for command...")
                    audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=12)
                else:
                    self.on_status_change("Waiting for 'Hey Moody'...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            return
        except Exception:
            return

        if not self.running:
            return

        try:
            self.on_status_change("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            if not text:
                return
            text = text.strip()
        except sr.UnknownValueError:
            return
        except sr.RequestError as e:
            self.on_log(f"Speech API error: {str(e)}", "error")
            return

        text_lower = text.lower()

        if not self.awake:
            for wake in self.WAKE_WORDS:
                if wake in text_lower:
                    self.set_awake(True)
                    self.speak("I'm listening!")
                    idx = text_lower.find(wake)
                    remaining = text[idx + len(wake):].strip()
                    if remaining and len(remaining) > 2:
                        self._process_command(remaining)
                    return
        else:
            for wake in self.WAKE_WORDS:
                if text_lower.startswith(wake):
                    self._last_command_time = time.time()
                    remaining = text[len(wake):].strip()
                    if remaining and len(remaining) > 2:
                        self._process_command(remaining)
                    else:
                        self.speak("Yes? I'm here!")
                    return

            self._process_command(text)

    def _process_command(self, text):
        self.on_log(f"🗣️ You: {text}", "user")
        self._last_command_time = time.time()

        self.command_history.append({
            'time': datetime.now().isoformat(),
            'command': text,
        })

        cmd_info, query = self.command_registry.match(text)

        if cmd_info:
            response = cmd_info['response']
            if response:
                self.speak(response)
            try:
                if cmd_info['takes_query']:
                    cmd_info['handler'](query)
                else:
                    cmd_info['handler']()
            except Exception as e:
                self.on_log(f"Command error: {str(e)}", "error")
                self.speak("Sorry, I had trouble with that command.")
        else:
            self.speak("I'm not sure how to do that. Try saying 'help' for a list of commands.")
            self.on_log(f"Unrecognized: {text}", "system")
