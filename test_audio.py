import sounddevice as sd
import numpy as np
import time

DEVICE_IDX = 3
DURATION = 10  # seconds to monitor

def audio_callback(indata):
    volume_norm = np.linalg.norm(indata) * 10
    print(f"Audio level: {volume_norm:.2f}", flush=True)
    
print(f"Monitoring audio from BlackHole (device {DEVICE_IDX}) for {DURATION} seconds...")
with sd.InputStream(device=DEVICE_IDX, channels=2, callback=audio_callback):
    time.sleep(DURATION)

print("Done monitoring.") 