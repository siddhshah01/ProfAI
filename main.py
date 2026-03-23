import queue
import threading

from record import record_chunks
from transcribe import transcribe_chunks
from embed import run_embed
from query import run_query, get_llm_client


def end_function(stop_event):
    """Waits for the user to press Enter, then signals all threads to stop."""
    print("Press Enter to stop recording...")
    input()
    stop_event.set()
    print("Stopping recording. Waiting for remaining audio to finish transcribing...")


if __name__ == "__main__":
    filenames_queue = queue.Queue()
    stop_event = threading.Event()

    input("Press Enter to start recording...")

    threads = [
        threading.Thread(target=record_chunks, args=(filenames_queue, stop_event)),
        threading.Thread(target=end_function, args=(stop_event,)),
        threading.Thread(target=transcribe_chunks, args=(filenames_queue, stop_event)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\nDone recording. Full transcription saved to transcription.txt")

    print("\nEmbedding and grouping lecture content...")
    full_docs, embed_client = run_embed()

    print("\nReady to answer questions about your lecture!")
    llm_client = get_llm_client()
    run_query(full_docs, embed_client, llm_client)

    print("\nSession complete.")
