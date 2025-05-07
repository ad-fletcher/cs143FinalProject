# Audio Routing Setup Guide

This project requires routing audio on macOS using BlackHole and a Multi-Output Device. Follow the steps below to get started.

---

## Prerequisites

- **Homebrew** (If you don't have it, install from [brew.sh](https://brew.sh/))
- **BlackHole (2ch)**
- **SwitchAudio-OSX** (for muting functionality)

---

## 1. Install BlackHole

Open your terminal and run:

```sh
brew install blackhole-2ch
```

---

## 1b. Install SwitchAudio-OSX (Optional - for muting)

If you need muting capabilities within the project, install `SwitchAudio-OSX`:

```sh
brew install switchaudio-osx
```

---

## 2. Configure a Multi-Output Device on macOS

1. **Open** `Audio MIDI Setup` (find it via Spotlight or in Applications > Utilities).
2. **Create a Multi-Output Device**:
   - Click the `+` button at the bottom left and select `Create Multi-Output Device`.
3. **Add Devices**:
   - In the right panel, check both `BlackHole 2ch` and your real output device (e.g., `MacBook Pro Speakers` or `EarPods`) in the "Use" column.
4. **Set Primary Device**:
   - Click the "Primary Device" drop-down and select your real output device (e.g., `EarPods` or `MacBook Pro Speakers`).
5. **Enable Drift Correction**:
   - In the "Drift Correction" column, check only `BlackHole 2ch`. Leave it unchecked for your real output device.

---

## 3. Set the Multi-Output Device as System Output

- Go to `System Settings` > `Sound` > `Output`.
- Select your new Multi-Output Device as the output device.

---

## 4. Install Python Dependencies

In your project directory, run:

```sh
pip install -r requirements.txt
```

---
 
## 5. Run the Project

Before running the project, you may need to identify the correct audio device index for BlackHole. You can typically list available audio devices using the list_devices script. Ensure the in settings that audio output is initially set to your Loopback or Multi-Output device.


---

## Troubleshooting

- If you don't hear audio, double-check the Multi-Output Device settings.
- Make sure Drift Correction is enabled only for BlackHole.
- Ensure your Python environment is using the correct version and all dependencies are installed.

---

## References

- [BlackHole GitHub](https://github.com/ExistentialAudio/BlackHole)
- [Audio MIDI Setup Guide](https://support.apple.com/en-us/HT202000)
- GPT assisted with code and debugging: [ChatGPT](https://chat.openai.com/chat)




