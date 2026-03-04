import os, json

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks


all_chunks = []  # list of {"text": ..., "meta": ...}

for file in os.listdir("preprocessedDataset"):
    if not file.endswith(".json"):
        continue

    path = os.path.join("preprocessedDataset", file)
    data = json.load(open(path, "r", encoding="utf-8"))

    for key, page_obj in data.items():
        text = page_obj["content"]
        source = page_obj["source"]
        page = page_obj["pageNumber"]

        chunks = chunk_text(text, chunk_size=1200, overlap=200)
        for i, c in enumerate(chunks):
            all_chunks.append({
                "text": c,
                "meta": {"source": source, "page": page, "chunk_id": i}
            })

os.makedirs("chunks", exist_ok=True)
with open("chunks/chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("Total chunks:", len(all_chunks))
