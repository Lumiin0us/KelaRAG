import chromadb
from chromadb.utils import embedding_functions
import ollama

DB_DIR = "chroma_db"
COLLECTION_NAME = "kelarag_chunks"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

from openai import OpenAI

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"  # LM Studio default
client_llm = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
LLM_MODEL = "phi-3.1-mini-instruct" 

def format_context(docs, metas, max_chars=2000):
    blocks = []
    total = 0
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        chunk_id = meta.get("chunk_id", "?")
        block = f"[SOURCE: {source}, page {page}, chunk {chunk_id}]\n{doc}\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n---\n".join(blocks)

def answer_question(question: str, top_k=2):
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    res = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    context = format_context(docs, metas)

    system = (
        "You are a careful assistant. Answer ONLY using the provided context.\n"
        "If the context does not contain the answer, say: "
        "'I couldn't find this in the provided Kela documents.'\n"
        "Always include citations as: (Source: <pdf>, p.<page>).\n"
        "Do not invent laws, numbers, or eligibility rules."
    )

    user = f"Question: {question}\n\nContext:\n{context}"

    resp = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=180,
    )
    return resp.choices[0].message.content, metas

if __name__ == "__main__":
    while True:
        q = input("\nAsk (or exit): ").strip()
        if q.lower() == "exit":
            break
        ans, sources = answer_question(q)
        print("\nANSWER:\n", ans)