import os, json
import chromadb
from chromadb.utils import embedding_functions

CHUNKS_PATH = "chunks/chunks.json"
DB_DIR = "chroma_db"
COLLECTION_NAME = "kelarag_chunks"

def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [x["text"] for x in data]
    metas = [x["meta"] for x in data]
    ids = [f"{m['source']}|p{m['page']}|c{m.get('chunk_id',0)}|i{i}" for i, m in enumerate(metas)]

    # 2) Create  Chroma client
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)

    # 3) Embedding function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # 4) Create / get collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    # 5) Add documents
    batch_size = 200
    for i in range(0, len(texts), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metas[i:i+batch_size]
        )
        print(f"Added {min(i+batch_size, len(texts))}/{len(texts)}")

    print("✅ Chroma index built at:", DB_DIR)

if __name__ == "__main__":
    main()
