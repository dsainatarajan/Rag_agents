# -*- coding: utf-8 -*-
"""
Ex2_rag_reranking_ragas.py

RAG pipeline (query-rewrite -> retrieve -> LLM re-rank -> generate) over an
INLINE telecom billing-complaints knowledge base, evaluated with RAGAS.

Changes vs. the original:
  * Removed the PyPDFLoader / external PDF dependency.
  * Added an inline multi-paragraph knowledge base (telecom billing complaints
    + resolutions), authored so the four core RAGAS metrics behave differently
    across the eval questions (see notes at the bottom).
  * Added RAGAS evaluation (faithfulness, answer_relevancy,
    context_precision, context_recall) using the SAME local vLLM model as judge.

Install (once):
  pip install -U ragas langchain-openai langchain-huggingface langchain-chroma \
                 langchain-community langchain-text-splitters chromadb rank_bm25
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Inline knowledge base  (replaces the PDF)
#    Each paragraph is a self-contained complaint-type + resolution policy so
#    that one chunk cleanly answers one question — good for interpretable metrics.
# ─────────────────────────────────────────────────────────────────────────────
KB_TEXT = """
Duplicate charge complaints occur when a customer is billed twice for the same monthly plan fee, usually due to a payment-gateway retry or an account-migration error. Our policy is to reverse the duplicate charge within 5 to 7 business days once it is verified against the transaction log. If the duplicate remained unresolved for more than one billing cycle, the customer is also credited a 10 percent goodwill adjustment on the affected bill.

International roaming disputes arise when a customer is charged premium roaming rates despite believing roaming was disabled. If the account shows no active roaming pack, agents verify the tower-connection logs; charges confirmed as accidental roaming while roaming was toggled off are fully waived. If a roaming pack was active but the customer exceeded its data cap, only the overage above the pack is billable, and we offer a one-time 50 percent reduction on that overage.

When a customer upgrades or downgrades a plan in the middle of a billing cycle, the bill reflects prorated charges: the old plan is charged for the days used before the change and the new plan for the remaining days. This often makes the first bill after a change look higher than expected. The resolution is to send an itemized proration breakdown; no refund is due because the prorated total is correct, though agents may apply a courtesy credit if the plan change was caused by an agent error.

Late payment fees of 9 dollars are applied automatically when a balance is not cleared by the due date. A first-time late fee is waived on request for customers in good standing with at least twelve months of on-time payments. Repeat late fees within a six-month window are not waivable, but customers can enroll in autopay to avoid future fees and receive a 2 dollar monthly autopay discount.

Data overage charges apply when a customer exceeds the high-speed data allowance on a capped plan, billed at 10 dollars per additional gigabyte. Customers who were not notified at the 90 percent usage threshold because of a system error are eligible to have the overage charges credited. Otherwise agents recommend upgrading to an unlimited plan, which stops overage billing going forward but does not retroactively remove past overage charges.

Autopay failures happen when a saved card expires or is declined, causing the account to be treated as unpaid and sometimes incurring a late fee. When the failure is traced to an expired card the customer had not updated, the late fee stands but any service-interruption charge is reversed. If the failure was caused by our payment system being down on the charge date, both the late fee and any reconnection charge are fully refunded.

A 3 dollar paper-bill fee is charged to accounts that receive mailed statements instead of e-bills. Customers are frequently unaware they were opted into paper billing during store sign-up. The resolution is to switch the account to e-billing, waive the current month's paper-bill fee, and refund up to three prior months of paper-bill fees if the customer states they never requested paper statements.

A reconnection fee of 20 dollars is charged when service is restored after a suspension for non-payment. This fee is waived if the suspension resulted from a billing error on our side, such as a misapplied payment. If the suspension was due to genuine non-payment, the reconnection fee is valid, but agents may split it into two installments for customers facing financial hardship.

Promotional discounts, such as a 15 dollar monthly new-customer credit, are time-limited and expire after the stated promo period, usually 12 months. When the promo ends the bill rises by the discount amount, which customers often mistake for an unexplained price increase. Agents explain the promo end date from the original contract; the charge is valid, but retention agents may offer a new loyalty discount to offset the increase.
""".strip()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Chunk the inline text into Documents
# ─────────────────────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""],   # "\n\n" first keeps paragraphs whole
)
chunks = text_splitter.create_documents([KB_TEXT])
print(f"Created {len(chunks)} chunks from the inline knowledge base")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Embeddings + vector store + retriever
# ─────────────────────────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma                # 1.x home
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback (deprecated)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./telecom_billing_db",
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},        # retrieve a few; precision/recall become meaningful
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM (local vLLM) + prompts  — unchanged pipeline logic
# ─────────────────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

VLLM_BASE_URL = "http://192.168.51.102:8000/v1"
VLLM_API_KEY  = "not-needed"
MODEL         = "openai/gpt-oss-120b"

llm = ChatOpenAI(model=MODEL, temperature=0, base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

query_rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Rewrite the user question to be optimal for semantic document retrieval.
Keep it concise and specific.

Original question:
{question}

Rewritten query:
""",
)

def rewrite_query(question):
    return llm.invoke(query_rewrite_prompt.format(question=question)).content

rerank_prompt = PromptTemplate(
    input_variables=["question", "documents"],
    template="""
Given the question and retrieved documents, rank the documents by relevance.
Return only the most relevant content.

Question:
{question}

Documents:
{documents}

Relevant content:
""",
)

def rerank(question, docs):
    docs_text = "\n\n".join(d.page_content for d in docs)
    return llm.invoke(rerank_prompt.format(question=question, documents=docs_text)).content

answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a telecom billing support assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the billing policy."

Context:
{context}

Question:
{question}

Answer:
""",
)

def generate_answer(question, context):
    return llm.invoke(answer_prompt.format(context=context, question=question)).content

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline — now also RETURNS the retrieved contexts (RAGAS needs them)
# ─────────────────────────────────────────────────────────────────────────────
def rag_pipeline(question):
    rewritten_query = rewrite_query(question)
    retrieved_docs  = retriever.invoke(rewritten_query)          # raw retrieval
    contexts        = [d.page_content for d in retrieved_docs]   # -> for RAGAS
    ranked_context  = rerank(question, retrieved_docs)           # condensed
    answer          = generate_answer(question, ranked_context)
    return answer, contexts

# quick smoke test
ans, ctx = rag_pipeline("I was billed twice for the same month. What happens?")
print("\nSample answer:\n", ans)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation set  — questions + reference answers (ground truth)
#    Q1–Q4 are answerable from the KB; Q5 is intentionally OUT OF SCOPE.
# ─────────────────────────────────────────────────────────────────────────────
eval_questions = [
    "I was billed twice for the same month. How is this resolved and how long does it take?",
    "I was charged roaming fees but I had roaming switched off. Will I get a refund?",
    "Why is my bill higher after I upgraded my plan mid-cycle, and will I be refunded?",
    "Can my late payment fee be waived?",
    "How do I reset my voicemail PIN?",          # <-- NOT in the billing KB (on purpose)
]

eval_references = [
    "The duplicate charge is reversed within 5 to 7 business days after it is verified against the transaction log, plus a 10 percent goodwill credit if it was unresolved for more than one billing cycle.",
    "If no roaming pack was active and the tower logs confirm accidental roaming while roaming was toggled off, the charges are fully waived.",
    "The higher amount is due to prorated charges for the old and new plans for their respective days; it is correct and no refund is due, though a courtesy credit applies if the change resulted from an agent error.",
    "A first-time late fee is waived for customers with at least twelve months of on-time payments; repeat late fees within a six-month window are not waivable.",
    "Dial your voicemail access number, open settings, choose reset PIN, and set a new four-digit PIN.",  # true answer lives OUTSIDE the KB
]

# ─────────────────────────────────────────────────────────────────────────────
# 6. Build the RAGAS dataset by running the pipeline on each question
# ─────────────────────────────────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

samples = []
for q, ref in zip(eval_questions, eval_references):
    answer, contexts = rag_pipeline(q)
    samples.append({
        "user_input":         q,          # the question
        "retrieved_contexts": contexts,   # list[str] the retriever returned
        "response":           answer,     # your pipeline's generated answer
        "reference":          ref,        # ground-truth answer (for context_recall)
    })

dataset = EvaluationDataset.from_list(samples)

# Use the SAME local vLLM model as the judge, and local embeddings — fully offline.
judge_llm  = LangchainLLMWrapper(llm)
judge_emb  = LangchainEmbeddingsWrapper(embedding_model)

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_emb,
    run_config=RunConfig(timeout=240, max_workers=4),   # local models are slower; be patient
)

print("\n================ RAGAS aggregate scores ================")
print(result)

df = result.to_pandas()
import pandas as pd
pd.set_option("display.max_colwidth", 60)
print("\n================ Per-question breakdown ================")
print(df[["user_input", "faithfulness", "answer_relevancy",
          "context_precision", "context_recall"]])

df.to_csv("ragas_results.csv", index=False)
print("\nSaved full results (with contexts + answers) to ragas_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# WHAT TO LOOK FOR  (why the data was written this way)
# ─────────────────────────────────────────────────────────────────────────────
# faithfulness      : high for all — temperature=0 + "use ONLY the context" keeps
#                     the model grounded (no hallucination), including Q5 where it
#                     should answer "I don't have that information...".
# answer_relevancy  : high for Q1–Q4; LOWER for Q5 because the refusal is not a
#                     substantive answer to "reset voicemail PIN".
# context_precision : reflects whether the ONE relevant paragraph ranked at the top
#                     of the 4 retrieved chunks. Clean single-topic paragraphs keep
#                     this high for Q1–Q4.
# context_recall    : high for Q1–Q4 (the reference facts exist in a retrieved chunk)
#                     but ~0.0 for Q5 — the voicemail-PIN answer is NOT in the KB, so
#                     retrieval could not possibly cover the reference. This is the
#                     key teaching contrast: a RETRIEVAL gap (low context_recall) vs.
#                     a GENERATION gap (low faithfulness) look completely different.
# ─────────────────────────────────────────────────────────────────────────────