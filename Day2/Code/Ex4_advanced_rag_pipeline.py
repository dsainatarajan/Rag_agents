# -*- coding: utf-8 -*-
"""
Ex5_advanced_rag_pipeline.py

Advanced RAG pipeline combining four techniques:
    1. Query rewriting   - LLM cleans up the raw question into a search-friendly query
    2. HyDE               - LLM writes a hypothetical answer; that answer (not the
                             raw question) is embedded and used to search
    3. Re-ranking          - LLM re-scores the broad candidate set and keeps only
                             the most relevant chunks
    4. Contextual compression - each kept chunk is trimmed down to just the
                             sentences that are actually relevant to the question

WHY THIS DOCUMENT (best-fit use case):
Advanced retrieval techniques earn their keep on long, jargon-heavy documents where
(a) users ask in plain language but the source uses specialized/legal/financial
terminology (the "vocabulary gap" HyDE is designed to close), and (b) a single
question is answered by a small paragraph buried inside dozens of pages of
boilerplate (exactly what re-ranking + contextual compression are for). Public
company SEC 10-K filings are a canonical example of this in industry RAG use
cases (financial/legal document Q&A). This script uses a real excerpt of
Apple Inc.'s FY2025 Form 10-K (source: SEC EDGAR, sec.gov) - see
Apple_10K_FY2025.pdf in this folder - which contains dense Risk Factors and
MD&A sections full of financial/legal jargon (tariffs, indemnification,
impairment, macroeconomic conditions, IP litigation, etc.).

For a simple, clean document like the EC-Council course catalog (Ex2-Ex4), plain
similarity search is usually good enough - these techniques would add cost and
latency without much benefit. That's why this exercise switches documents.
"""

import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

try:
    # LangChain >= 1.0 (langchain-classic package)
    from langchain_classic.retrievers.document_compressors.chain_extract import (
        LLMChainExtractor,
    )
except ImportError:
    # LangChain < 1.0
    from langchain.retrievers.document_compressors import LLMChainExtractor

# ---------------------------------------------------------------------------
# 1. Load + chunk the document
# ---------------------------------------------------------------------------
loader = PyPDFLoader("Apple_10K_FY2025.pdf")   # real SEC 10-K excerpt, see note above
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# ---------------------------------------------------------------------------
# 2. Vector store (dense embeddings) - same embedding model as Ex2-Ex4
# ---------------------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./apple10k_rag_db"
)

# ---------------------------------------------------------------------------
# 3. LLM setup (same local vLLM endpoint as Ex2)
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

compressor = LLMChainExtractor.from_llm(llm)

# ---------------------------------------------------------------------------
# 4. STEP 1 - Query rewriting
# ---------------------------------------------------------------------------
rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Rewrite the user question to be clear, specific, and optimal for document
retrieval. Keep any financial/legal terms the user implied. Return ONLY the
rewritten query, nothing else.

Original question:
{question}

Rewritten query:
"""
)

def rewrite_query(question):
    return llm.invoke(rewrite_prompt.format(question=question)).content.strip()

# ---------------------------------------------------------------------------
# 5. STEP 2 - HyDE (Hypothetical Document Embeddings)
# ---------------------------------------------------------------------------
hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Write a short, factual-sounding paragraph (3-5 sentences) that WOULD answer the
question below, as if it were an excerpt from an SEC Form 10-K filing. It does
not need to be accurate - it only needs to read like real 10-K language, so it
can be used to search for the real matching section.

Question:
{question}

Hypothetical 10-K excerpt:
"""
)

def generate_hyde_document(question):
    return llm.invoke(hyde_prompt.format(question=question)).content.strip()

def hyde_retrieve(question, k=12):
    """Embed the hypothetical document (not the raw question) and search."""
    hyde_doc = generate_hyde_document(question)
    hyde_vector = embedding_model.embed_query(hyde_doc)
    candidates = vectorstore.similarity_search_by_vector(hyde_vector, k=k)
    return candidates, hyde_doc

# ---------------------------------------------------------------------------
# 6. STEP 3 - Re-ranking
#    Ask the LLM to rank the candidate chunks by relevance to the ORIGINAL
#    question (not the hypothetical doc) and keep only the top few.
# ---------------------------------------------------------------------------
rerank_prompt = PromptTemplate(
    input_variables=["question", "numbered_docs"],
    template="""
Question: {question}

Below are numbered candidate excerpts from a document. Rank them from MOST to
LEAST relevant for answering the question. Return ONLY a comma-separated list
of the excerpt numbers in ranked order, e.g. "4,1,7,2". Do not explain.

Excerpts:
{numbered_docs}

Ranked order:
"""
)

def rerank(question, docs, top_n=5):
    numbered = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    response = llm.invoke(
        rerank_prompt.format(question=question, numbered_docs=numbered)
    ).content
    order = [int(n) for n in re.findall(r"\d+", response)]
    seen, ranked_docs = set(), []
    for n in order:
        idx = n - 1
        if 0 <= idx < len(docs) and idx not in seen:
            seen.add(idx)
            ranked_docs.append(docs[idx])
    # fall back to original order for any docs the LLM didn't mention
    for i, d in enumerate(docs):
        if i not in seen:
            ranked_docs.append(d)
    return ranked_docs[:top_n]

# ---------------------------------------------------------------------------
# 7. STEP 4 - Contextual compression
#    Trim each kept chunk down to just the sentences relevant to the question.
# ---------------------------------------------------------------------------
def compress(question, docs):
    compressed = compressor.compress_documents(docs, question)
    return list(compressed)

# ---------------------------------------------------------------------------
# 8. Final answer generation
# ---------------------------------------------------------------------------
answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful financial-documents assistant.
Answer the question using ONLY the context below. If the context does not
contain the answer, say so.

Context:
{context}

Question:
{question}

Answer:
"""
)

def generate_answer(question, context):
    return llm.invoke(answer_prompt.format(context=context, question=question)).content

# ---------------------------------------------------------------------------
# 9. Helper to inspect chunks at each stage
# ---------------------------------------------------------------------------
def show(label, docs):
    print(f"\n--- {label} ({len(docs)} chunks) ---")
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", "?")
        preview = d.page_content.strip().replace("\n", " ")[:150]
        print(f"[{i}] page={page} | {preview}...")

# ---------------------------------------------------------------------------
# 10. Full pipeline
# ---------------------------------------------------------------------------
def advanced_rag(question, k_candidates=12, top_n_rerank=5):
    print(f"\n===== Question: {question!r} =====")

    rewritten = rewrite_query(question)
    print(f"\n[1] Rewritten query: {rewritten}")

    candidates, hyde_doc = hyde_retrieve(rewritten, k=k_candidates)
    print(f"\n[2] HyDE hypothetical excerpt used for search:\n{hyde_doc}")
    show("HyDE candidate retrieval", candidates)

    reranked = rerank(question, candidates, top_n=top_n_rerank)
    show("After re-ranking", reranked)

    compressed = compress(question, reranked)
    show("After contextual compression", compressed)

    context = "\n\n".join(d.page_content for d in compressed) if compressed else \
        "\n\n".join(d.page_content for d in reranked)

    answer = generate_answer(question, context)
    return answer

if __name__ == "__main__":
    question = "What risks does Apple disclose related to tariffs and its global supply chain?"
    answer = advanced_rag(question)
    print("\n===== Final Answer (full advanced pipeline) =====")
    print(answer)
