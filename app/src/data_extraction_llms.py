from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from project root (assuming you run from /rag_llms)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Add it to .env or export it.")

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

def get_embedding_function():
    # You can also omit api_key=... and just rely on env
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY,
    )

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

def split_pdf(pages, chunk_size=1500, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(pages)

def create_vectorstore_faiss(chunks, embedding_function, vectorstore_path):
    unique_chunks = []
    seen_content = set()
    for chunk in chunks:
        if chunk.page_content not in seen_content:
            seen_content.add(chunk.page_content)
            unique_chunks.append(chunk)

    vectorstore = FAISS.from_documents(
        documents=unique_chunks,
        embedding=embedding_function,
    )
    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore.save_local(vectorstore_path)
    return vectorstore

def load_vectorstore_faiss(vectorstore_path, embedding_function):
    return FAISS.load_local(
        vectorstore_path,
        embedding_function,
        allow_dangerous_deserialization=True,
    )

# 🔹 HERE is the new function you were missing
def build_rag_chain(retriever):
    """
    Build a simple RAG chain: retriever -> prompt -> ChatOpenAI.
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant for questions about a research paper.
Use the provided context to answer the question clearly and concisely.
If the context is not enough, say you are not sure.

Context:
{context}

Question:
{question}

Answer in 3–6 short sentences, in simple English.
        """.strip()
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Runnable pipeline:
    #   input question -> {"context": retrieved_docs, "question": question}
    #   -> prompt -> llm
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return rag_chain
