# Naive RAG with Gemini + Chroma

A minimal Retrieval-Augmented Generation (RAG) pipeline built with LangChain, Google Gemini (`gemini-embedding-001` + `gemini-2.5-flash`), and ChromaDB as the vector store. Includes per-stage latency instrumentation (embedding, retrieval, context assembly, and LLM generation with time-to-first-token).

## Features

-  Ingests `.txt` documents from a local folder
-  Chunks documents with `RecursiveCharacterTextSplitter`
-  Embeds and stores chunks in a persistent Chroma vector store (cosine similarity)
-  Answers questions using retrieved context, grounded strictly in the source documents
-  Tracks latency per stage: embedding, retrieval, context assembly, LLM (TTFT + total)

## Project Structure

```
.
├── ingestion_pipeline.py           # Loads, chunks, and embeds documents into Chroma
├── retrieval_pipeline.py           # Retrieves relevant chunks and answers a question, with latency tracking
├── docs/                           # Put your .txt source documents here
├── db/                             # Persisted Chroma vector store (created after ingestion)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your `.env` file**
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Add your source documents**
   Place `.txt` files inside a `docs/` folder in the project root.

## Usage

**1. Ingest your documents** (builds the vector store):
```bash
python ingestion_pipeline.py
```

**2. Query the system**:
```bash
python retrieval_pipeline.py
```


## Example Output

```
--- Generated Response ---
Deep learning is a subfield of machine learning that uses neural networks
with many layers to learn increasingly abstract representations of data...

--- Latency Breakdown ---
Embedding latency:            120.45 ms
Retrieval latency:             38.20 ms
Context assembly:               0.15 ms
LLM TTFT:                      410.30 ms
LLM generation (post-1st):    980.60 ms
LLM total:                   1390.90 ms
----------------------------------------
End-to-end total:            1549.70 ms
```

