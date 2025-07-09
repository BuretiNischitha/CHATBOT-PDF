# CHATBOT-PDF
# 📄 Chat with your PDF using Hugging Face & LangChain

A Streamlit-based chatbot that allows users to upload PDF documents and ask questions. The chatbot uses a custom Hugging Face transformer model (`flan-t5-base`) to generate contextual answers.

---

## 🚀 Features

- Upload multiple PDFs
- Semantic search using FAISS + Sentence Transformers
- Contextual Q&A with HuggingFace LLM (Flan-T5)
- Easy-to-use Streamlit interface

---

## 🛠️ Tech Stack

- LangChain
- Hugging Face Transformers (`flan-t5-base`)
- FAISS Vector Store
- Streamlit
- PyPDF2

---

## How to Run Locally

```bash
git clone https://github.com/<your-username>/pdf-chatbot.git
cd pdf-chatbot
pip install -r requirements.txt
streamlit run app.py
