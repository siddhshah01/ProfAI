import os
import queue
import threading

from record import record_chunks
from transcribe import transcribe_chunks
from embed import run_embed
from query import run_query, get_llm_client

COURSES_DIR = os.path.join(os.path.dirname(__file__), "courses")


def end_function(stop_event):
    """Waits for the user to press Enter, then signals all threads to stop."""
    print("Press Enter to stop recording...")
    input()
    stop_event.set()
    print("Stopping recording. Waiting for remaining audio to finish transcribing...")


def do_record():
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


def pick_from_list(items, label):
    """Prints a numbered list and returns the user's chosen item."""
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    while True:
        raw = input(f"\nSelect a {label} (number): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return items[int(raw) - 1]
        print(f"Please enter a number between 1 and {len(items)}.")


def do_answer():
    if not os.path.isdir(COURSES_DIR):
        print(f"No courses directory found at '{COURSES_DIR}'. Create it and add course folders with .txt files.")
        return

    courses = sorted(
        d for d in os.listdir(COURSES_DIR)
        if os.path.isdir(os.path.join(COURSES_DIR, d))
    )
    if not courses:
        print("No courses found. Add subdirectories to the 'courses/' folder.")
        return

    print("\nCourses:")
    course = pick_from_list(courses, "course")

    course_path = os.path.join(COURSES_DIR, course)
    txt_files = sorted(f for f in os.listdir(course_path) if f.endswith(".txt"))
    if not txt_files:
        print(f"No .txt files found in '{course}'.")
        return

    print(f"\nFiles in {course}:")
    chosen_file = pick_from_list(txt_files, "file")

    selected_path = os.path.join(course_path, chosen_file)
    print(f"\nEmbedding '{chosen_file}'...")
    full_docs, embed_client = run_embed(transcription_file=selected_path)

    print(f"\nReady to answer questions about '{chosen_file}'!")
    llm_client = get_llm_client()
    run_query(full_docs, embed_client, llm_client)


if __name__ == "__main__":
    print("What would you like to do?")
    print("  1. Record a lecture")
    print("  2. Answer questions from a saved file")

    while True:
        choice = input("\nEnter 1 or 2: ").strip()
        if choice == "1":
            do_record()
            break
        elif choice == "2":
            do_answer()
            break
        else:
            print("Please enter 1 or 2.")

    print("\nSession complete.")
