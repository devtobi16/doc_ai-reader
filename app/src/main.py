import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

from data_extraction_llms import (
    PyPDFLoader,
    RecursiveCharacterTextSplitter,
    get_embedding_function,
    create_vectorstore_faiss,
    load_vectorstore_faiss,
    build_rag_chain,   
)

load_dotenv()


def main():
    st.title("Research Paper Q&A App")
    st.write("Upload a PDF and ask questions about its content.")

    st.sidebar.write("Has OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    question = st.text_input("Ask a question about the paper:")

    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(pages)

        embedding_function = get_embedding_function()
        vectorstore = create_vectorstore_faiss(
            chunks, embedding_function, "vectorstore_faiss"
        )
        retriever = vectorstore.as_retriever(search_type="similarity")

        if question:
            rag_chain = build_rag_chain(retriever)
            answer = rag_chain.invoke(question)

            st.subheader("Answer")
            st.write(answer.content)

            st.subheader("Top relevant chunks (context used)")
            relevant_chunks = retriever.invoke(question)
            for i, chunk in enumerate(relevant_chunks, start=1):
                with st.expander(f"Chunk {i}"):
                    st.write(chunk.page_content)

if __name__ == "__main__":
    main()
