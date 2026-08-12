import re, os
import tiktoken

from bs4 import BeautifulSoup

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings          # LOCAL embeddings (was langchain_openai.OpenAIEmbeddings)
from langchain_community.vectorstores import SKLearnVectorStore

# ── Local model config ───────────────────────────────────────────────────────
# Local sentence-transformers model runs in-process (CPU/GPU), no API key needed.
# NOTE: 384-dim vs text-embedding-3-large's 3072-dim — the vector store MUST be
# rebuilt (this script does that) and every reader must use the SAME model.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def count_tokens(text, model="cl100k_base"):
    """Count the number of tokens in the text using tiktoken."""
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    main_content = soup.find("article", class_="md-content__inner")
    content = main_content.get_text() if main_content else soup.text
    content = re.sub(r"\n\n+", "\n\n", content).strip()
    return content


def load_langgraph_docs():
    """Load LangGraph documentation from the official website."""
    print("Loading LangGraph documentation...")

    urls = ["https://langchain-ai.github.io/langgraph/concepts/",
     "https://langchain-ai.github.io/langgraph/how-tos/",
     "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
     "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
     "https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/",
    ]

    docs = []
    for url in urls:
        loader = RecursiveUrlLoader(url, max_depth=5, extractor=bs4_extractor)
        for d in loader.lazy_load():
            docs.append(d)

    print(f"Loaded {len(docs)} documents from LangGraph documentation.")
    print("\nLoaded URLs:")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.metadata.get('source', 'Unknown URL')}")

    total_tokens = 0
    tokens_per_doc = []
    for doc in docs:
        total_tokens += count_tokens(doc.page_content)
        tokens_per_doc.append(count_tokens(doc.page_content))
    print(f"Total tokens in loaded documents: {total_tokens}")

    return docs, tokens_per_doc


def save_llms_full(documents):
    """Save the documents to a file."""
    output_filename = "llms_full.txt"
    with open(output_filename, "w") as f:
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Unknown URL')
            f.write(f"DOCUMENT {i+1}\n")
            f.write(f"SOURCE: {source}\n")
            f.write("CONTENT:\n")
            f.write(doc.page_content)
            f.write("\n\n" + "="*80 + "\n\n")
    print(f"Documents concatenated into {output_filename}")


def split_documents(documents):
    """Split documents into smaller chunks for improved retrieval."""
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=8000,
        chunk_overlap=500,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} chunks from documents.")

    total_tokens = 0
    for doc in split_docs:
        total_tokens += count_tokens(doc.page_content)
    print(f"Total tokens in split documents: {total_tokens}")
    return split_docs


def create_vectorstore(splits):
    """Create a vector store from document chunks using SKLearnVectorStore."""
    print("Creating SKLearnVectorStore...")

    embeddings = get_embeddings()          # LOCAL embeddings

    persist_path = os.getcwd() + "/sklearn_vectorstore.parquet"
    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet",
    )
    print("SKLearnVectorStore created successfully.")
    vectorstore.persist()
    print("SKLearnVectorStore was persisted to", persist_path)
    return vectorstore


# Load the documents
documents, tokens_per_doc = load_langgraph_docs()

# Save the documents to a file
save_llms_full(documents)

# Split the documents
split_docs = split_documents(documents)

# Create the vector store  (rebuilds the parquet with LOCAL 384-dim embeddings)
vectorstore = create_vectorstore(split_docs)

# Create retriever to get relevant documents (k=3 means return top 3 matches)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Get relevant documents for the query
query = "What is LangGraph?"
relevant_docs = retriever.invoke(query)
print(f"Retrieved {len(relevant_docs)} relevant documents")

for d in relevant_docs:
    print(d.metadata['source'])
    print(d.page_content[0:500])
    print("\n--------------------------------\n")
