from langchain_community.document_loaders import Docx2txtLoader

from langchain_community.document_loaders import DirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate 

import os
import shutil

loader = DirectoryLoader(
    path=".",                          
    glob="*.docx",                     
    loader_cls=Docx2txtLoader  
)

docs = loader.load()
print(f"Total documents loaded: {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
)

chunks = text_splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

if os.path.exists("./chroma_db"):
    vectordb = Chroma(
        persist_directory = "./chroma_db",
        embedding_function = embeddings
    )

else:
    vectordb = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = "./chroma_db"
    )

print("Vector store ready!")

prompt_template = """You are an expert assistant on Telecom 3GPP standards documentation.
Answer the question using ONLY the information explicitly present in the context below.
Do NOT use any external knowledge, make assumptions, or add any information beyond what is in the context.
Do NOT say "However" or add extra context after saying information is not available.
If the answer is not explicitly present in the context, respond with exactly this sentence and nothing else:
"This information is not available in the provided 3GPP documentation."

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,          # 0 = maximum factual, zero creativity - hallucination minimum!
)

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(
        search_kwargs={"k": 8}  
    ),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

print("Chain ready!")

# Test query
question = input("Enter your question: ")
result = qa_chain.invoke({"query": question})
print("\nAnswer:", result["result"])


print("\nSources used:")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"{i}. {doc.page_content[:200]}...")






