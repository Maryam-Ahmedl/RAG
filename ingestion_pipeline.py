import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_docs(docs_path):
    if not os.path.exists(docs_path):
        raise FileNotFoundError("Path did not exist")
    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError("no documents founded")
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

def main():
    print("Ingestion Pipline......")
    documents = load_docs("docs")
    print(f'found {len(documents)} documnets')
    chunks = split_docs(documents)
    vector_store = create_vector_store(chunks)
    print("End of Ingestion Pipline")

if __name__ == "__main__":
    main()