from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.agents.utils.config import (
    GOOGLE_MODEL,
    LLM,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENAI_MODEL,
)


def llm_selector(openai_model=None):
    if LLM == "google-gemini":
        llm = ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm

    elif LLM == "ollama":
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm

    elif LLM == "openai":
        llm = ChatOpenAI(
            model=openai_model if openai_model else OPENAI_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm

    else:
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm
