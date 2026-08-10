# -*- coding: utf-8 -*-
"""
Ex3_rag_similarity_hybrid.py

Extends Ex2_rag_openai.py to demonstrate two retrieval strategies side by side:

1. SIMILARITY (dense) retrieval  -> Chroma vector store + embeddings, cosine similarity search
2. HYBRID retrieval              -> dense (Chroma) + sparse (BM25 keyword) combined with an
                                     EnsembleRetriever (reciprocal rank fusion of both result lists)

Same PDF, chunking, embedding model, and local vLLM endpoint as Ex2 so results are comparable.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# 1. Load + chunk the document (identical to Ex2)
# ---------------------------------------------------------------------------
loader = PyPDFLoader("EC-Council-University-catalog.pdf")   # your PDF
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# ---------------------------------------------------------------------------
# 2. SIMILARITY RETRIEVER (dense, embedding-based)
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

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

# ---------------------------------------------------------------------------
# 3. SPARSE RETRIEVER (keyword-based, BM25) -- needed for hybrid search
# ---------------------------------------------------------------------------
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 8

# ---------------------------------------------------------------------------
# 4. HYBRID RETRIEVER = similarity + BM25, fused via EnsembleRetriever
#    weights control how much each retriever contributes to the final ranking
# ---------------------------------------------------------------------------
hybrid_retriever = EnsembleRetriever(
    retrievers=[similarity_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# ---------------------------------------------------------------------------
# 5. LLM setup (same local vLLM endpoint as Ex2)
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

def generate_answer(question, context):
    response = llm.invoke(
        answer_prompt.format(context=context, question=question)
    )
    return response.content

# ---------------------------------------------------------------------------
# 6. Helper to show what each retriever returns for a given query
# ---------------------------------------------------------------------------
def show_retrieved(label, docs):
    print(f"\n--- {label} ({len(docs)} chunks) ---")
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", "?")
        preview = d.page_content.strip().replace("\n", " ")[:150]
        print(f"[{i}] page={page} | {preview}...")

def compare_retrievers(question):
    print(f"\n===== Query: {question!r} =====")

    sim_docs = similarity_retriever.invoke(question)
    show_retrieved("Similarity (dense) retrieval", sim_docs)

    bm25_docs = bm25_retriever.invoke(question)
    show_retrieved("BM25 (keyword) retrieval", bm25_docs)

    hybrid_docs = hybrid_retriever.invoke(question)
    show_retrieved("Hybrid retrieval (dense + BM25 fused)", hybrid_docs)

    return sim_docs, hybrid_docs

# ---------------------------------------------------------------------------
# 7. End-to-end RAG using hybrid retrieval
# ---------------------------------------------------------------------------
def rag_answer(question, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    return generate_answer(question, context)

if __name__ == "__main__":
    question = "What are the key courses mentioned?"

    sim_docs, hybrid_docs = compare_retrievers(question)

    print("\n===== Answer using SIMILARITY retrieval only =====")
    print(rag_answer(question, similarity_retriever))

    print("\n===== Answer using HYBRID retrieval (dense + BM25) =====")
    print(rag_answer(question, hybrid_retriever))
