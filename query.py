import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from embed import get_embedding, cosine_similarity

load_dotenv()


def _load_prompt(filename):
    """Loads a prompt template from the prompts/ directory."""
    path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def get_llm_client():
    return InferenceClient(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key=os.getenv("HF_TOKEN")
    )


def rank_documents(question_emb, full_docs):
    data_list = [
        (cosine_similarity(question_emb, mean_emb), chunk_list)
        for mean_emb, chunk_list, _ in full_docs
    ]
    return sorted(data_list, key=lambda x: x[0], reverse=True)


def top_k(k, question_emb, full_docs):
    return [doc for _, doc in rank_documents(question_emb, full_docs)[:k]]


def summarize_docs(question, flattened_docs, llm_client):
    """Summarizes the retrieved lecture content relative to the question."""
    prompt = _load_prompt("summarizer.txt")
    response = llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"{prompt}\n\nContent: {flattened_docs}."},
            {"role": "user", "content": question}
        ],
        max_tokens=1500,
        temperature=0.2
    )
    return response.choices[0].message.content


def ask_llm_question(question, full_docs, embed_client, llm_client):
    """Retrieves relevant chunks, summarizes them, then answers the question using the professor prompt."""
    question_embedding = get_embedding(question, embed_client)
    top_docs = top_k(10, question_embedding, full_docs)
    flattened_docs = "\n".join(f"- {chunk}" for doc in top_docs for chunk in doc)

    summarized = summarize_docs(question, flattened_docs, llm_client)

    prompt = _load_prompt("professor.txt")
    response = llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"{prompt}\n\nCONTENT: {summarized}."},
            {"role": "user", "content": question}
        ],
        max_tokens=1000,
        temperature=0.3
    )
    return response.choices[0].message.content


def run_query(full_docs, embed_client, llm_client=None):
    """Interactive Q&A loop over the embedded lecture content."""
    if llm_client is None:
        llm_client = get_llm_client()

    question = input("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\nType 'q' to exit, otherwise enter your question: ")
    while question != 'q':
        print("\n" + ask_llm_question(question, full_docs, embed_client, llm_client))
        question = input("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\nType 'q' to exit, otherwise enter your question: ")


if __name__ == "__main__":
    from embed import run_embed
    full_docs, embed_client = run_embed()
    run_query(full_docs, embed_client)
