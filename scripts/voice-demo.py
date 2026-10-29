import subprocess
import urllib.parse
import webbrowser
from typing import Literal

import numpy as np
import sounddevice as sd
import needle
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CONFIDENCE_TO_ACT = 0.5  # below this (but above Needle's own floor) we ask first


# ---------------------------------------------------------------- tools
# One narrow tool per action, with plain descriptions: Needle reads these literally.

@needle.tool
def open_gmail():
    "Open Gmail, the user's email inbox, in the web browser."
    webbrowser.open("https://mail.google.com")
    return "opened Gmail"


@needle.tool
def open_website(site: str):
    "Open a website by name in the browser, e.g. 'github', 'youtube', 'wikipedia', 'linkedin'."
    site = site.lower().strip().replace(" ", "")
    url = site if "." in site else f"{site}.com"
    webbrowser.open(f"https://{url}")
    return f"opened {url}"


@needle.tool
def search_google(query: str):
    "Search Google for a query."
    webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote_plus(query))
    return f"searched Google for {query}"


@needle.tool
def search_youtube(query: str):
    "Search YouTube for videos about a query."
    webbrowser.open("https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query))
    return f"searched YouTube for {query}"


@needle.tool
def open_app(app: Literal["notepad", "calculator", "file explorer"]):
    "Open a desktop app: notepad, calculator or file explorer."
    commands = {"notepad": "notepad", "calculator": "calc", "file explorer": "explorer"}
    subprocess.Popen(commands[app])
    return f"opened {app}"


TOOLS = [open_gmail, open_website, search_google, search_youtube, open_app]
FUNCTIONS = {f.__name__: f for f in TOOLS}


# ---------------------------------------------------------------- speech
def record_until_enter() -> np.ndarray:
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    input("\nPress Enter to START speaking... ")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input("Recording... press Enter to STOP. ")
    if not chunks:
        return np.zeros(0, dtype="float32")
    return np.concatenate(chunks).flatten()


def transcribe(whisper: WhisperModel, audio: np.ndarray) -> str:
    segments, _ = whisper.transcribe(audio, language="en", vad_filter=True, beam_size=1)
    return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------- main loop
def main():
    print("Loading Whisper (first run downloads the model)...")
    whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    print("Loading Needle...")
    agent = needle.Needle(tools=TOOLS)

    print("\nReady. Try: 'open Gmail', 'search YouTube for machine learning',")
    print("'open GitHub', 'open notepad', 'search Google for small language models'.")

    while True:
        audio = record_until_enter()
        if audio.size < SAMPLE_RATE // 2:
            print("Too short, try again.")
            continue

        text = transcribe(whisper, audio)
        print(f'\n[Whisper] heard: "{text}"')
        if not text:
            continue
        if text.lower().strip(" .!") in ("quit", "exit", "stop"):
            break

        result = agent.complete(text)
        calls = result.get("function_calls") or []
        held = result.get("suppressed_calls") or []
        confidence = result.get("confidence", 0.0)
        print(f"[Needle] calls={calls} confidence={confidence:.2f}")

        if not calls and not held:
            print("[Action] No matching tool, nothing done.")
            continue

        to_run = calls or held
        if not calls or confidence < CONFIDENCE_TO_ACT:
            answer = input(f"Not fully sure. Run {to_run}? (y/n) ").strip().lower()
            if answer != "y":
                continue

        for call in to_run:
            fn = FUNCTIONS.get(call["name"])
            if fn is None:
                print(f"[Action] Unknown tool {call['name']}")
                continue
            print("[Action]", fn(**call["arguments"]))


if __name__ == "__main__":
    main()