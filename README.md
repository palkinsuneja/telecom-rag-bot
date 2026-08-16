# 📡 Telecom 3GPP RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from **3GPP 5G standards documentation** with minimal to near-zero hallucinations.

Built as part of a Graduate Engineer Trainee assignment for Mavenir.

## 🎯 Key Focus: Minimal Hallucinations

The system is designed to strictly answer from the provided documentation only:
- If the answer exists in the docs → accurate, source-backed answer
- If the answer is NOT in the docs → explicitly says "This information is not available in the provided 3GPP documentation" instead of making something up

This is achieved through:
- **Strict prompt engineering** — LLM is explicitly instructed to use ONLY the retrieved context
- **temperature=0** — zero creativity, maximum factual accuracy
- **Source transparency** — every answer shows exactly which document chunks were used

## 🚀 Features

- **3GPP 5G Documentation Q&A** — query across 4 3GPP 5G standards documents simultaneously
- **Near-zero hallucinations** — strict context-only answering with explicit fallback message
- **Source transparency** — expandable "View Sources" section for every answer
- **Persistent vector store** — ChromaDB stores embeddings so documents aren't re-processed on every run
- **Clean chat interface** — Streamlit UI with chat history

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Document Loader | Docx2txtLoader (for .docx 3GPP specs) |
| UI | Streamlit |

## ⚙️ How It Works

3GPP .docx files → Load → Chunk (1000 tokens, 200 overlap) → Embed → ChromaDB
↓
User Query → Embed Query → Retrieve Top-5 Similar Chunks → Groq LLM → Answer + Sources


**Why chunk_size=1000 and overlap=200?**
Technical 3GPP documents need larger context windows than general text — a single concept often spans multiple paragraphs. Higher overlap ensures no critical information is lost at chunk boundaries, reducing hallucinations.

## 📦 Setup

```bash
# Clone the repo
git clone https://github.com/palkinsuneja/telecom-rag-bot.git
cd telecom-rag-bot

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set Groq API key
export GROQ_API_KEY="your_api_key_here"
```

## ▶️ Usage

1. Place your 3GPP 5G standards `.docx` files in the project folder (4 documents used in this implementation)
2. Run the pipeline once to build the vector store:
```bash
python3 rag_pipeline.py
```
3. Launch the Streamlit app:
```bash
streamlit run app.py
```
4. Ask questions about 3GPP 5G standards in the chat interface!

## 📁 Project Structure

telecom-rag-bot/
├── app.py # Streamlit chat UI
├── rag_pipeline.py # Core RAG pipeline
├── requirements.txt
├── .gitignore
└── README.md


## 👩‍💻 Author

**Palkin Suneja** — Final Year BTech (Industrial IoT), VIPS Delhi







