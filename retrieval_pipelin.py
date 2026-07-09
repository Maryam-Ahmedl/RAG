import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# User query
query = "what is Deep learning?"

latencies = {}

# --- 1. Embedding latency ---
t0 = time.perf_counter()
query_embedding = embedding_model.embed_query(query)
t1 = time.perf_counter()
latencies["embedding"] = t1 - t0

# --- 2. Retrieval latency 
t2 = time.perf_counter()
relevant_docs = db.similarity_search_by_vector(query_embedding, k=5)
t3 = time.perf_counter()
latencies["retrieval"] = t3 - t2

# --- 3. Context assembly latency ---
t4 = time.perf_counter()
combined_input = f"""Based on the following documents, please answer this question: {query}
Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
Please provide a clear, helpful answer using only the information from these documnets. if you can not find the answer in the documents, 
say 'I do not have enough information to answer that question based on the pointed documents'
"""
t5 = time.perf_counter()
latencies["context_assembly"] = t5 - t4

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = [
    SystemMessage(content="you are a helpful assistant"),
    HumanMessage(content=combined_input)
]

# --- 4. LLM latency (TTFT + total generation) ---
start = time.perf_counter()
first_token_time = None
chunks = []

for event in model.stream(messages):
    if event.content:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        chunks.append(event.content)

end = time.perf_counter()
answer = "".join(chunks)

latencies["llm_ttft"] = first_token_time - start
latencies["llm_generation"] = end - first_token_time  # time after first token to last token
latencies["llm_total"] = end - start

# --- Report ---
end_to_end = latencies["embedding"] + latencies["retrieval"] + latencies["context_assembly"] + latencies["llm_total"]

print("\n --- Generated Response ---")
print(answer)

print("\n--- Latency Breakdown ---")
print(f"Embedding latency:       {latencies['embedding']*1000:8.2f} ms")
print(f"Retrieval latency:       {latencies['retrieval']*1000:8.2f} ms")
print(f"Context assembly:        {latencies['context_assembly']*1000:8.2f} ms")
print(f"LLM TTFT:                {latencies['llm_ttft']*1000:8.2f} ms")
print(f"LLM generation (post-1st):{latencies['llm_generation']*1000:8.2f} ms")
print(f"LLM total:                {latencies['llm_total']*1000:8.2f} ms")
print(f"{'-'*40}")
print(f"End-to-end total:        {end_to_end*1000:8.2f} ms")