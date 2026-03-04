# KelaRAG

This project implements a Retrieval-Augmented Generation (RAG) system built on official Kela (Finnish Social Insurance Institution) documentation. It involves the datasets used (PDFs) from their official source, sentence tokinizer, vector database (ChromaDB), and LM Studio is used to run a local Large Language Model (LLM) for answer generation. 
After retrieving, several preprocessing techniques were applied to clean the data, such as removing table-of-contents pages, or pages that did not contain relevant information. Followed by splitting pages into smaller chunks for easier embeddings and vector storage. 

---

# Retrieval Process

When a user asks a question:

1. The query is converted into an embedding
2. ChromaDB retrieves the most relevant chunks
3. Retrieved context is assembled into a prompt
4. The prompt is sent to the LLM

---

# LLM Generation

A local LLM running through **LM Studio** generates the final answer.

The model is instructed to:

- answer only using retrieved context
- avoid hallucinating information
- cite document sources where possible

---

## Project Structure

dataset/  
preprocessedDataset/  
chunks/  
chroma_db/  
preprocess.py  
chunking.py  
chromaBuild.py  
ragAnswer.py  
  
---

## Workflow

User Query  
   ↓  
Embedding Model  
   ↓  
ChromaDB Vector Search  
   ↓  
Top-K Context  
   ↓  
Local LLM (LM Studio)  
   ↓  
Answer  


---

## Future Improvements

- RAG evaluation metrics (faithfulness / relevance)
- Streaming responses
- Web interface with Streamlit
- Cloud deployment
