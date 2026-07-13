import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from latency_log import log_query, query_report, ingest_report

load_dotenv()

PERSIST_DIRECTORY = "db/chroma_db"


_embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=_embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)
_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def query(question: str, mode: str = "naive", k: int = 5) -> dict:
    # --- 1. Embedding latency ---
    t0 = time.perf_counter()
    query_embedding = _embedding_model.embed_query(question)
    t1 = time.perf_counter()
    embedding_seconds = t1 - t0

    # --- 2. Retrieval latency ---
    t2 = time.perf_counter()
    relevant_docs = _db.similarity_search_by_vector(query_embedding, k=k)
    t3 = time.perf_counter()
    retrieval_seconds = t3 - t2

    # --- 3. Context assembly ---
    context = "\n".join(f"- {doc.page_content}" for doc in relevant_docs)
    combined_input = f"""Based on the following documents, please answer this question: {question}
    Documents:
    {context}
    Please provide a clear, helpful answer using only the information from these documents. If you cannot find the answer in the documents,
    say 'I do not have enough information to answer that question based on the provided documents'
    """
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content=combined_input),
    ]

    # --- 4. LLM latency (TTFT + total generation) ---
    start = time.perf_counter()
    first_token_time = None
    chunks = []
    for event in _llm.stream(messages):
        if event.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(event.content)
    end = time.perf_counter()
    answer = "".join(chunks)

    ttft_seconds = (first_token_time or end) - start
    llm_seconds = end - start
    end_to_end_seconds = embedding_seconds + retrieval_seconds + llm_seconds

    log_query(mode=mode, question=question,
        timings={
            "embedding_seconds": f"{embedding_seconds:.4f}",
            "retrieval_seconds": f"{retrieval_seconds:.4f}",
            "ttft_seconds": f"{ttft_seconds:.4f}",
            "llm_seconds": f"{llm_seconds:.4f}",
            "end_to_end_seconds": f"{end_to_end_seconds:.4f}",
        },
    )
    return {"answer": answer, "context": context, "sources": relevant_docs}


if __name__ == "__main__":
    print("Type a question, or 'exit' to quit.")
    print("Type 'report' to see query latency trends. Type 'ingest_report' for ingestion history.\n")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question.lower() == "report":
            print("\n" + query_report() + "\n")
            continue
        if question.lower() == "ingest_report":
            print("\n" + ingest_report() + "\n")
            continue
        result = query(question)
        print("\n--- Generated Response ---")
        print(result["answer"])
        print()