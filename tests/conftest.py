import os

os.environ.setdefault("LLM_CASSETTE_MODE", "replay")
os.environ.setdefault("LLM_CASSETTE_PATH", "evals/cassettes/llm.json")
