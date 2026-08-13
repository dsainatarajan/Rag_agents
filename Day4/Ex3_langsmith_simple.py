import os
from langchain_openai import ChatOpenAI

# --- LangSmith (set before the call) ---
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "lsv2_...your_free_key..."
os.environ["LANGSMITH_PROJECT"] = "gpt-oss-demo"      # optional; groups runs in the UI

# --- Local gpt-oss-120b via vLLM ---
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="http://192.168.51.102:8000/v1",
    api_key="not-needed",
    temperature=0,
)

resp = llm.invoke(input("Ask something: "))
print(resp.content)

# flush trace before the script exits (short scripts die before upload otherwise)
from langchain_core.tracers.langchain import wait_for_all_tracers
wait_for_all_tracers()