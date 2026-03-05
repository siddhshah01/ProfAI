import sounddevice as sd
import soundfile as sf

# --- Configuration ---
fs = 16000      # Sample rate
duration = 22   # Duration of each recorded chunk in seconds
channels = 1    # Mono audio


def record_chunks(filenames_queue, stop_event):
    """Records audio in chunks continuously, putting filenames into the queue as each chunk finishes."""
    iteration = 0
    while not stop_event.is_set():
        filename = f"chunk_{iteration}.wav"
        print(f"Recording {filename} ({duration}s)...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float32')
        sd.wait()
        sf.write(filename, audio, fs)
        del audio
        filenames_queue.put(filename)
        print(f"Saved {filename}")
        iteration += 1
