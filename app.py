import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Telecom 3GPP RAG Bot", page_icon="📡", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(160deg, #0E1117 0%, #0a1628 50%, #0E1117 100%); }
.user-bubble { background-color: #0066cc; color: white; padding: 12px 16px; border-radius: 16px 16px 4px 16px; margin: 8px 0; max-width: 80%; margin-left: auto; }
.bot-bubble { background-color: #1A1D24; color: #FAFAFA; padding: 12px 16px; border-radius: 16px 16px 16px 4px; margin: 8px 0; max-width: 80%; border: 1px solid #0066cc33; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📡 About")
    st.markdown("""
    This chatbot answers questions based on **Telecom 3GPP standards documentation**.
    
    **Stack:**
    - LangChain
    - ChromaDB
    - HuggingFace Embeddings
    - Groq (Llama 3.3-70b)
    - Streamlit
    """)
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<h1>📡 Telecom 3GPP Assistant</h1>', unsafe_allow_html=True)
st.caption("Ask anything from the 3GPP standards documentation.")

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    prompt_template = """You are an expert assistant on Telecom 3GPP standards documentation.
Answer the question using ONLY the context provided below.
If the answer is not present in the context, say "This information is not available in the provided 3GPP documentation."
Do NOT make up any information.

Context: {context}
Question: {question}
Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain

qa_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📄 View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.text(f"{i}. {src[:300]}...")

question = st.chat_input("Ask a question about 3GPP standards...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f'<div class="user-bubble">{question}</div>', unsafe_allow_html=True)
    
    with st.spinner("Searching 3GPP documentation..."):
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        sources = [doc.page_content for doc in result.get("source_documents", [])]
    
    st.markdown(f'<div class="bot-bubble">{answer}</div>', unsafe_allow_html=True)
    if sources:
        with st.expander("📄 View Sources"):
            for i, src in enumerate(sources, 1):
                st.text(f"{i}. {src[:300]}...")
    
    st.session_state.messages.append({"role": "bot", "content": answer, "sources": sources})