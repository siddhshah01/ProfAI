# ProfAI — Lecture Q&A Assistant

ProfAI records live lectures, transcribes them locally, groups the content by topic, and lets you ask questions about what was said — all from the command line.

## How It Works

```
Record audio → Transcribe with Whisper → Embed & group by topic → Q&A with Llama 3.1
```

1. **Record** — Captures your microphone in 22-second chunks while the lecture is live
2. **Transcribe** — Each chunk is transcribed locally using OpenAI Whisper (no data leaves your machine)
3. **Embed & Group** — Transcribed sentences are embedded with `sentence-transformers/all-MiniLM-L6-v2` and clustered by semantic similarity
4. **Q&A** — You ask questions; the system retrieves the most relevant topic clusters and answers using Llama 3.1 8B via HuggingFace

## Project Structure

```
ProfAI/
├── main.py           # Entry point — runs the full pipeline
├── record.py         # Mic recording in audio chunks
├── transcribe.py     # Whisper transcription of each chunk
├── embed.py          # Embedding + semantic grouping of transcription
├── query.py          # Q&A over grouped lecture content
├── prompts/
│   ├── professor.txt # System prompt for the answering LLM
│   ├── summarizer.txt# System prompt for the summarization step
│   └── verifier.txt  # System prompt for LLM-as-judge verification (optional)
└── requirements.txt
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/ProfAI.git
cd ProfAI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your HuggingFace API token**

Create a `.env` file in the project root from the `.env.example` skeleton:
```
HF_TOKEN=your_huggingface_token_here
```

You can get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Make sure it has access to `meta-llama/Llama-3.1-8B-Instruct` (you may need to accept the model license on HuggingFace first).

## Usage

Run the full pipeline with:
```bash
python main.py
```

- Press **Enter** to start recording
- Press **Enter** again to stop
- Once recording stops, the transcript is embedded and grouped automatically
- A Q&A prompt appears — type your question or `q` to exit

### Running stages individually

```bash
python embed.py   # Re-embed an existing transcription.txt
python query.py   # Q&A over an existing grouped_chunks.txt
```

## Models Used

| Task | Model | Runs Locally? |
|------|-------|---------------|
| Transcription | openai/whisper-turbo | Yes |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 | Via HuggingFace API |
| Q&A | meta-llama/Llama-3.1-8B-Instruct | Via HuggingFace API |

## Customizing Prompts

The LLM system prompts live in `prompts/` as plain text files. Edit them to change how the model answers, summarizes, or verifies responses — no code changes needed.

## Requirements

- Python 3.10+
- A microphone
- A HuggingFace account with API access
