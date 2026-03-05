import whisper
import os


def transcribe_chunks(filenames_queue, stop_event, output_file="transcription.txt"):
    """Loads Whisper, waits for filenames from the queue, transcribes each chunk, then deletes the wav."""
    model = whisper.load_model("turbo")
    while not stop_event.is_set() or not filenames_queue.empty():
        try:
            filename = filenames_queue.get(timeout=2)
        except Exception:
            continue
        print(f"Transcribing {filename}...")
        result = model.transcribe(filename, fp16=False)
        text = result["text"].strip()
        print(f"  -> {text}")

        # Clean up wav file after transcription
        os.remove(filename)
        print(f"Deleted {filename}")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    del model
