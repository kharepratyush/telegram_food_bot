import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
LLM = os.getenv("LLM")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
