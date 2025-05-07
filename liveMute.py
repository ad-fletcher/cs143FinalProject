#!/usr/bin/env python3
"""
#codede with assistance from ChatGPT

Dependencies
------------
pip install sounddevice numpy librosa scikit‑learn tensorflow tensorflow‑addons

Audio routing
-------------
• Install BlackHole (2‑ch) via Homebrew      →  brew install blackhole-2ch
• In macOS Audio MIDI Setup create a Multi‑Output device that
  includes BlackHole + your speakers/headphones.
• Set system output to the Multi‑Output; this script listens
  on BlackHole (BH_IDX below).

"""

import collections
import queue
import subprocess
import threading
import time
import numpy as np
import librosa
import joblib
import sounddevice as sd
import tensorflow as tf
import tensorflow_addons as tfa

# ──────────────────────────── Load assets ────────────────────────────────
model = tf.keras.models.load_model(
    "avgAll.keras",
    custom_objects={"SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy},
)
scaler = joblib.load("avgAll.pkl")

WINDOW_MODEL = 3  # model trained on 3 × 3 s windows
FEATURE_NAMES = [
    "shot_length",
    "ste_mean",
    "zcr_mean",
    "centroid_mean",
    "rolloff_mean",
    "flux_mean",
    "fundfreq_mean",
]
buf = collections.deque(maxlen=WINDOW_MODEL)
prediction_history = collections.deque(maxlen=4) # Stores the last 4 predictions
COMMERCIALS_IN_HISTORY_THRESHOLD = 3
HISTORY_LENGTH = 4

# ─────────────────────── Audio/streaming parameters ──────────────────────
WINDOW_SEC = np.random.uniform(3.0, 3.5)  # choose once at startup
SR, CHANNELS = 48_000, 2
FRAMES = int(WINDOW_SEC * SR)
BH_IDX = -1  # Will be set by user input

# ─────────────────────────── macOS mute helper ───────────────────────────
MUTE_SECONDS = 270

_is_muting = threading.Event()  # guard against overlap

# Device names will be populated by argparse
LOOPBACK_SPEAKER_DEVICE_NAME = ""
HEADPHONES_DEVICE_NAME = ""

def _get_user_device_selection(prompt_message: str, devices_list: list, device_type_filter: str = "any"):
    print(prompt_message)
    valid_devices_for_prompt = []
    for i, device in enumerate(devices_list):
        is_valid = False
        if device_type_filter == "input" and device['max_input_channels'] > 0:
            is_valid = True
        elif device_type_filter == "output" and device['max_output_channels'] > 0:
            is_valid = True
        elif device_type_filter == "any":
            is_valid = True
        
        if is_valid:
            print(f"  {i}: {device['name']} (Inputs: {device['max_input_channels']}, Outputs: {device['max_output_channels']})")
            valid_devices_for_prompt.append((i, device['name']))

    if not valid_devices_for_prompt:
        print(f"No suitable {device_type_filter} devices found. Exiting.")
        exit()

    while True:
        try:
            choice = int(input("Enter device number: "))
            for index, name in valid_devices_for_prompt:
                if choice == index:
                    print(f"Selected: {name}")
                    return choice, name # Return both index and name
            print("Invalid selection. Please choose from the listed numbers.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def _mute_for(seconds: int):
    """Mute macOS for <seconds> then restore previous state by switching devices."""
    global LOOPBACK_SPEAKER_DEVICE_NAME, HEADPHONES_DEVICE_NAME # Ensure we use the global (parsed) names
    if _is_muting.is_set():
        return
    _is_muting.set()
    
    switched_to_headphones = False
    original_system_output_device = LOOPBACK_SPEAKER_DEVICE_NAME # Assume this is the default for the script

    try:
        print(f"Attempting to mute for {seconds}s by switching to {HEADPHONES_DEVICE_NAME}...")
        # Switch system output to headphones
        subprocess.run(["SwitchAudioSource", "-s", HEADPHONES_DEVICE_NAME], check=True)
        switched_to_headphones = True
        print(f"System output switched to {HEADPHONES_DEVICE_NAME}.")

        # Mute the output
        subprocess.run(["osascript", "-e", "set volume output muted true"], check=True)
        print(f"{HEADPHONES_DEVICE_NAME} muted.")
        
        # During this sleep, the script will not be processing new audio via BlackHole
        # because system audio is not routed through LoopbackSpeaker.
        time.sleep(seconds)

    except subprocess.CalledProcessError as e:
        print(f"Error during mute/device switch: {e}")
        # Attempt to restore to LoopbackSpeaker if anything went wrong during the switch/mute
        if switched_to_headphones: # Only try to switch back if we know we switched away
            try:
                print(f"Restoring output to {original_system_output_device} due to error...")
                subprocess.run(["SwitchAudioSource", "-s", original_system_output_device], check=True)
                print(f"System output restored to {original_system_output_device}.")
            except subprocess.CalledProcessError as e_restore:
                print(f"Error restoring output device: {e_restore}")
    except Exception as e:
        print(f"An unexpected error occurred in _mute_for: {e}")
    finally:
        if switched_to_headphones:
            try:
                # Unmute the output (which should still be headphones)
                subprocess.run(["osascript", "-e", "set volume output muted false"], check=True)
                print(f"{HEADPHONES_DEVICE_NAME} unmuted.")

                # Switch system output back to the original device (LoopbackSpeaker)
                subprocess.run(["SwitchAudioSource", "-s", original_system_output_device], check=True)
                print(f"System output restored to {original_system_output_device}.")
            except subprocess.CalledProcessError as e:
                print(f"Error during unmute/restore device switch: {e}")
            except Exception as e_unex:
                 print(f"An unexpected error occurred during finally block in _mute_for: {e_unex}")
        
        _is_muting.clear()


# ─────────────────────────── Feature extractor ───────────────────────────
def extract_features(y: np.ndarray, sr: int):
    """Return a dict of features for one window."""
    # 1) shot_length = mean length of non‑silent chunks
    intervals = librosa.effects.split(y, top_db=20)
    durations = (intervals[:, 1] - intervals[:, 0]) / sr
    shot_length = float(np.mean(durations)) if durations.size else 0.0

    # 2) STE
    ste = y**2
    ste_mean = float(np.mean(ste))

    # 3) ZCR
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
    zcr_mean = float(np.mean(zcr))

    # 4) Spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
    centroid_mean = float(np.mean(cent))

    # 5) Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048, hop_length=512)
    rolloff_mean = float(np.mean(rolloff))

    # 6) Spectral flux
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    flux = librosa.onset.onset_strength(S=S, sr=sr)
    flux_mean = float(np.mean(flux))

    # 7) Fundamental frequency
    f0, *_ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )
    f0_clean = f0[~np.isnan(f0)]
    fundfreq_mean = float(np.mean(f0_clean)) if f0_clean.size else 0.0

    return {
        "shot_length": shot_length,
        "ste_mean": ste_mean,
        "zcr_mean": zcr_mean,
        "centroid_mean": centroid_mean,
        "rolloff_mean": rolloff_mean,
        "flux_mean": flux_mean,
        "fundfreq_mean": fundfreq_mean,
    }


# ─────────────────────── Streaming infrastructure ────────────────────────
audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, flush=True)  # xruns, overruns, etc.
    audio_q.put(indata.copy())  # PortAudio will reuse its buffer


# Main script execution / Argument parsing
if __name__ == "__main__":
    all_devices = sd.query_devices()
    
    print("Device Selection:")
    print("="*20)

    BH_IDX, _ = _get_user_device_selection(
        "1. Select your BlackHole input device (used for listening):", 
        all_devices, 
        device_type_filter="input"
    )
    
    _, HEADPHONES_DEVICE_NAME = _get_user_device_selection(
        "\n2. Select your Headphones output device (will be muted/unmuted directly):", 
        all_devices, 
        device_type_filter="output"
    )

    _, LOOPBACK_SPEAKER_DEVICE_NAME = _get_user_device_selection(
        "\n3. Select your Loopback/Multi-Output device (should include BlackHole and Headphones):", 
        all_devices, 
        device_type_filter="output"
    )
    print("="*20)
    print("Configuration complete.")

    sd.default.device, sd.default.samplerate, sd.default.channels = (BH_IDX, None), SR, CHANNELS
    
    print(f"Using BlackHole Index: {BH_IDX}")
    print(f"Using Headphones Device for muting: '{HEADPHONES_DEVICE_NAME}'")
    print(f"Using Loopback/Multi-Output Device: '{LOOPBACK_SPEAKER_DEVICE_NAME}'")
    print("Ensure your Mac's system audio output is set to your Loopback/Multi-Output device before proceeding.")
    input("Press Enter to start monitoring after confirming system output...")

    print(
        f"→ Buffering… need {WINDOW_MODEL} × {WINDOW_SEC:.1f}s "
        "of audio before first prediction"
    )

    with sd.InputStream(
        device=BH_IDX,
        channels=CHANNELS,
        samplerate=SR,
        blocksize=FRAMES,
        dtype="float32",
        callback=audio_callback,
    ):
        while True:
            block = audio_q.get()  # waits exactly WINDOW_SEC seconds
            y = block.mean(axis=1)  # down‑mix to mono

            # Print RMS of the current audio block
            rms = np.sqrt(np.mean(y**2))
            print(f"Audio block RMS: {rms:.6f}", flush=True)

            # Feature extraction
            feats = extract_features(y, SR)
            buf.append([feats[k] for k in FEATURE_NAMES])

            if len(buf) < WINDOW_MODEL:
                print(f"…{len(buf)}/{WINDOW_MODEL} windows collected")
                continue  # still filling buffer

            # Prepare + predict
            x = np.array(buf, dtype=np.float32)[None, ...]  # (1, 3, 7)
            n, w, f = x.shape
            x_scaled = scaler.transform(x.reshape(-1, f)).reshape(n, w, f)

            prob = model.predict(x_scaled, verbose=0).item()
            label = "COMMERCIAL" if prob > 0.42 else "PROGRAMME"
            print(f"[{label}]  p={prob:.3f}", flush=True)
            prediction_history.append(label)

            # Mute trigger based on 3 out of last 4 predictions
            if len(prediction_history) == HISTORY_LENGTH and \
               prediction_history.count("COMMERCIAL") >= COMMERCIALS_IN_HISTORY_THRESHOLD and \
               not _is_muting.is_set():
                print(f"🛑 Detected {prediction_history.count('COMMERCIAL')} of last {HISTORY_LENGTH} as ads – muting for {MUTE_SECONDS}s")
                threading.Thread(target=_mute_for, args=(MUTE_SECONDS,), daemon=True).start()
                prediction_history.clear() # Clear history after triggering mute
