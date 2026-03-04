import chromadb
from chromadb.utils import embedding_functions

DB_DIR = "chroma_db"
COLLECTION_NAME = "kelarag_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def main():
    # 1) Loading the persisted DB
    client = chromadb.PersistentClient(path=DB_DIR)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    print("✅ Loaded collection:", COLLECTION_NAME)

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue

        res = collection.query(
            query_texts=[q],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]  # smaller = closer (usually)

        print("\n--- Top results (lower distance = better) ---")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            chunk_id = meta.get("chunk_id", "?")

            print(f"\n#{i}  dist={dist:.4f}  |  {source}  p.{page}  (chunk {chunk_id})")
            print(doc[:700].replace("\n", " ") + ("..." if len(doc) > 700 else ""))

if __name__ == "__main__":
    main()