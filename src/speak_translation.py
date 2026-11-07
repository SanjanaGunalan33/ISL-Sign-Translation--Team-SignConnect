"""Step 8: speak (or save) the English translation with offline TTS.

Tries pyttsx3 first. If that fails (a common COM error on Windows), it falls
back to Windows' built-in speech engine through PowerShell, which needs no
extra Python packages.

Usage:
    python -m src.speak_translation "bring water for me"
    python -m src.speak_translation "bring water for me" --save out.wav
"""
import argparse
import platform
import subprocess


def _speak_windows(text: str, save_path: str | None = None, rate: int = 0) -> bool:
    safe = text.replace("'", "''")
    out = ""
    if save_path:
        out = f"$s.SetOutputToWaveFile('{str(save_path).replace(chr(39), chr(39) * 2)}');"
    cmd = ("Add-Type -AssemblyName System.Speech;"
           "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
           f"$s.Rate = {rate};{out}$s.Speak('{safe}');$s.Dispose()")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception as e:
        print(f"[Windows TTS unavailable: {e}]")
        return False


def speak(text: str, save_path: str | None = None, rate: int = 160) -> bool:
    """Returns True on success. Never raises: the app must still show the text if TTS fails."""
    if not text.strip():
        return False
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        if save_path:
            engine.save_to_file(text, save_path)
        else:
            engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        if platform.system() == "Windows":
            return _speak_windows(text, save_path)
        print(f"[TTS unavailable: {e}]")
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    print("spoke" if speak(a.text, a.save) else "speech failed")