import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import queue
import re

load_dotenv()

# --- Configuration ---
THRESHOLD = 0.60        # Cosine similarity threshold for grouping chunks
MAX_GROUP_SIZE = 15     # Max number of chunks per semantic group
MIN_WORDS = 4           # Drop chunks shorter than this

FILLERS = {
    "yes", "yeah", "okay", "ok", "so", "anyway", "right",
    "i don't know", "uh", "um", "hmm", "..."
}


def get_embedding_client():
    return InferenceClient(api_key=os.getenv("HF_TOKEN"))


def get_embedding(text, client):
    embedding = client.feature_extraction(
        text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    arr = np.array(embedding, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def cosine_similarity(a, b):
    return np.dot(a, b)


def load_chunks(transcription_file="transcription.txt"):
    """Reads transcription.txt, filters fillers and short chunks, returns a queue of cleaned sentences."""
    chunks = queue.Queue()

    with open(transcription_file, encoding="utf-8") as f:
        buffer = ""
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Merge lines that don't end with punctuation
            buffer = (buffer + " " + line).strip() if buffer else line

            if buffer.endswith(('.', '!', '?')):
                for s in re.split(r'(?<=[.!?])\s+', buffer):
                    cleaned = s.strip()
                    lowered = cleaned.lower().strip(".!?")
                    if lowered in FILLERS or len(cleaned.split()) < MIN_WORDS:
                        continue
                    chunks.put(cleaned)
                buffer = ""

        # Catch any leftover text without trailing punctuation
        if buffer:
            for s in re.split(r'(?<=[.!?])\s+', buffer):
                cleaned = s.strip()
                lowered = cleaned.lower().strip(".!?")
                if lowered in FILLERS or len(cleaned.split()) < MIN_WORDS:
                    continue
                chunks.put(cleaned)

    chunks.put(None)  # Sentinel to signal end of queue
    return chunks


def group_chunks(chunks, client):
    """Groups semantically similar chunks by comparing embeddings to a running mean per group."""
    full_docs = []  # Format: [(mean_emb, [chunk1, chunk2, ...], count), ...]

    while True:
        chunk = chunks.get()
        if chunk is None:
            break

        chunk_embedding = get_embedding(chunk, client)
        found = False

        for i, (mean_emb, chunk_list, count) in enumerate(full_docs):
            if cosine_similarity(chunk_embedding, mean_emb) > THRESHOLD and count < MAX_GROUP_SIZE:
                # Update running mean embedding
                new_mean = (mean_emb * count + chunk_embedding) / (count + 1)
                new_mean /= np.linalg.norm(new_mean)
                full_docs[i] = (new_mean, chunk_list + [chunk], count + 1)
                found = True
                break

        if not found:
            full_docs.append((chunk_embedding, [chunk], 1))

    return full_docs


def save_groups(full_docs, output_file="grouped_chunks.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (_, chunk_list, count) in enumerate(full_docs):
            f.write(f"Group {i} ({count} chunks): {chunk_list}\n")


def run_embed(transcription_file="transcription.txt", output_file="grouped_chunks.txt"):
    client = get_embedding_client()
    print("Loading and cleaning transcription...")
    chunks = load_chunks(transcription_file)
    print("Grouping semantically similar chunks...")
    full_docs = group_chunks(chunks, client)
    save_groups(full_docs, output_file)
    print(f"Grouped into {len(full_docs)} topic clusters. Saved to {output_file}")
    return full_docs, client


if __name__ == "__main__":
    run_embed()
