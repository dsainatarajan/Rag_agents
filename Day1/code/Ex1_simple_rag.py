# -*- coding: utf-8 -*-
"""
Ex1_simple_rag.py

The most basic RAG pipeline: load a PDF, chunk it, embed the chunks,
retrieve the top-k most similar chunks for a question, and ask the LLM to
answer using just that context. No query rewriting, no re-ranking, no
compression -- those are added in later exercises (Ex2 onward).
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# 1. Load the document
# ---------------------------------------------------------------------------
loader = PyPDFLoader("EC-Council-University-catalog.pdf")   # your PDF
documents = loader.load()
print(f"Pages loaded: {len(documents)}")

# ---------------------------------------------------------------------------
# 2. Split into chunks
# ---------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Chunks created: {len(chunks)}")

# ---------------------------------------------------------------------------
# 3. Embed chunks + store in a vector database
# ---------------------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./pdf_rag_db"
)

# ---------------------------------------------------------------------------
# 4. Retriever - plain similarity search, top 4 chunks
# ---------------------------------------------------------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ---------------------------------------------------------------------------
# 5. LLM (local vLLM endpoint, OpenAI-compatible)
# ---------------------------------------------------------------------------
VLLM_BASE_URL = "http://192.168.51.100:8000/v1"
VLLM_API_KEY = "not-needed"          # vLLM ignores this unless launched with --api-key
MODEL = "openai/gpt-oss-120b"

llm = ChatOpenAI(
    model=MODEL,
    temperature=0,
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

# ---------------------------------------------------------------------------
# 6. Prompt + answer generation
# ---------------------------------------------------------------------------
answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

def simple_rag(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    response = llm.invoke(answer_prompt.format(context=context, question=question))
    return response.content

if __name__ == "__main__":
    question = "What are the key courses mentioned?"
    answer = simple_rag(question)
    print("\n===== Answer =====")
    print(answer)
