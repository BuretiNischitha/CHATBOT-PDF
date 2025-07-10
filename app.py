import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from transformers import pipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import List, Any
from pydantic import PrivateAttr

# Load environment variables from .env file
load_dotenv()

# Custom wrapper for Hugging Face transformer-based models
class HFTransformerLLM(LLM):
    _model_pipeline: Any = PrivateAttr()

    def __init__(self, model_pipeline: Any):
        super().__init__()
        self._model_pipeline = model_pipeline  # Internally store pipeline

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        result = self._model_pipeline(prompt)
        return result[0]["generated_text"]

    @property
    def _llm_type(self) -> str:
        return "custom-huggingface-pipeline"

# Extract raw text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page_text := page.extract_text():
                text += page_text
    return text

# Split large text into smaller overlapping chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# Create and save FAISS vector store using sentence-transformer embeddings
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load question-answering chain using prompt and custom LLM
def get_conversational_chain():
    prompt_template = """
Use the following context to answer the user's question in detail and in paragraph form.
Always include all relevant information from the context.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=2048,
    temperature=0.3,
    repetition_penalty=1.1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)

    llm = HFTransformerLLM(model_pipeline=pipe)
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# Handle user question and generate an answer using vector similarity + LLM
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists("faiss_index"):
        st.warning("No vector store found. Please upload and process a PDF first.")
        return
    
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=2)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("💬 **Reply:**", response["output_text"])

# Streamlit App Entry Point
def main():
    st.set_page_config("Chat with PDF - HF Pipeline")
    st.header("Chat with your PDF using FLAN-T5 🤖")

    # Input box for user to ask questions
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        user_input(user_question)

    # Sidebar for uploading and processing PDFs
    with st.sidebar:
        st.title("📄 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("✅ PDF processed and vector store saved!")

if __name__ == "__main__":
    main()

