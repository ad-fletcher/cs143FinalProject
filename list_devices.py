import sounddevice as sd

# Print all available audio devices
print("Available audio devices:")
print("-" * 50)
devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"Device {i}: {device['name']}")
    print(f"  Max input channels: {device['max_input_channels']}")
    print(f"  Max output channels: {device['max_output_channels']}")
    print(f"  Default sample rate: {device['default_samplerate']}")
    print()

# Print default devices
print("Default input device:", sd.default.device[0])
print("Default output device:", sd.default.device[1]) 