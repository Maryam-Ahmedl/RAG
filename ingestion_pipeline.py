import hashlib
import time
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from latency_log import log_ingest

load_dotenv()

PERSIST_DIRECTORY = "db/chroma_db"

def _doc_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def ingest_file(path: Path, vectorstore, chunk_size=1000, chunk_overlap=0) -> int:
    # --- Load + split ("extraction") ---
    t0 = time.perf_counter()
    raw_doc = TextLoader(str(path), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(raw_doc)
    t1 = time.perf_counter()
    extraction_seconds = t1 - t0

    # --- Embed + write to Chroma ---
    t2 = time.perf_counter()
    vectorstore.add_documents(chunks)
    t3 = time.perf_counter()
    embed_seconds = t3 - t2

    n_chunks = max(len(chunks), 1)
    full_text = "\n".join(c.page_content for c in chunks)

    log_ingest(file=path.name, doc_hash=_doc_hash(full_text), chunks=len(chunks),
        timings={
            "avg_extraction_seconds": f"{extraction_seconds / n_chunks:.4f}",
            "avg_chunk_embed_seconds": f"{embed_seconds / n_chunks:.4f}",
            "entity_embed_seconds": "0.0000",  # naive RAG has no knowledge-graph step
            "total_seconds": f"{extraction_seconds + embed_seconds:.2f}",
        },
    )
    return len(chunks)

def load_docs(docs_path: str = "docs") -> list[Path]:
    docs_dir = Path(docs_path)
    if not docs_dir.exists():
        raise FileNotFoundError("Path did not exist")
    txt_files = list(docs_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError("no documents found")
    return txt_files


def main(docs_path: str = "docs"):
    txt_files = load_docs(docs_path)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Ingestion Pipeline: found {len(txt_files)} files")
    for f in txt_files:
        n = ingest_file(f, vectorstore)
        print(f"  {f.name}: {n} chunks")
    print("End of Ingestion Pipeline")


if __name__ == "__main__":
    main()