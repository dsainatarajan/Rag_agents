import os

#from mcp.server.fastmcp import FastMCP
from mcp.server.mcpserver import MCPServer
from langchain_huggingface import HuggingFaceEmbeddings          # LOCAL embeddings (was OpenAIEmbeddings)
from langchain_community.vectorstores import SKLearnVectorStore

# Portable path: the folder this script lives in (was a hardcoded Windows path).
PATH = os.path.dirname(os.path.abspath(__file__)) + os.sep

# No OpenAI key needed anymore — embeddings run locally.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


# Create an MCP server
mcp = MCPServer("LangGraph-Docs-MCP-Server")


@mcp.tool()
def langgraph_query_tool(query: str):
    """
    Query the LangGraph documentation using a retriever.

    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    retriever = SKLearnVectorStore(
        embedding=get_embeddings(),
        persist_path=os.path.join(PATH, "sklearn_vectorstore.parquet"),
        serializer="parquet",
    ).as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join(
        [f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
    )
    return formatted_context


@mcp.tool()
def demo_query_tool():
    """
    Returns demo

    Args:
        None

    Returns:
        str: A str of the retrieved documents
    """
    return "demo tool"


@mcp.resource("docs://langgraph/full")
def get_all_langgraph_docs() -> str:
    """
    Get all the LangGraph documentation. Returns the contents of llms_full.txt.

    Returns:
        str: The contents of the LangGraph documentation
    """
    doc_path = os.path.join(PATH, "llms_full.txt")
    try:
        with open(doc_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading log file: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport='stdio')
