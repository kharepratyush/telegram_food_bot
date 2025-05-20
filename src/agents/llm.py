from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.utils.config import LLM
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from src.agents.utils.config import OLLAMA_MODEL, OLLAMA_URL
from src.agents.utils.config import OPENAI_MODEL, GOOGLE_MODEL


def llm_selector(openai_model=None):
    if LLM == "google-gemini":
        llm = ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    elif LLM == "ollama":
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    elif LLM == "openai":
        llm = ChatOpenAI(
            model=openai_model if openai_model else OPENAI_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    else:
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    return llm
