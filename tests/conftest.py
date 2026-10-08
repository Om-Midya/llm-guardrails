import os

os.environ.setdefault("LLM_CASSETTE_MODE", "replay")
os.environ.setdefault("LLM_CASSETTE_PATH", "evals/cassettes/llm.json")
# Empty keys win over .env (load_dotenv never overrides), so tests never send traces.
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
