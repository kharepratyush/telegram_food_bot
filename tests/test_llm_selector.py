import importlib

import pytest

# skip if required LLM libraries not available
pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_google_genai")
pytest.importorskip("langchain_ollama")


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in ["LLM", "GOOGLE_MODEL", "OPENAI_MODEL", "OLLAMA_MODEL", "OLLAMA_URL"]:
        monkeypatch.delenv(key, raising=False)
    yield


def test_default_selector_openai(monkeypatch):
    monkeypatch.setenv("LLM", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    import src.agents.llm as llm_module

    importlib.reload(llm_module)
    sel = llm_module.llm_selector()
    from langchain_openai import ChatOpenAI

    assert isinstance(sel, ChatOpenAI)


def test_selector_google(monkeypatch):
    monkeypatch.setenv("LLM", "google-gemini")
    monkeypatch.setenv("GOOGLE_MODEL", "g-test")
    import src.agents.utils.config as config_module
    import src.agents.llm as llm_module
    importlib.reload(config_module)
    importlib.reload(llm_module)
    sel = llm_module.llm_selector()
    from langchain_google_genai import ChatGoogleGenerativeAI

    assert isinstance(sel, ChatGoogleGenerativeAI)


def test_selector_ollama(monkeypatch):
    monkeypatch.setenv("LLM", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "o-test")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost")
    import src.agents.utils.config as config_module
    import src.agents.llm as llm_module
    importlib.reload(config_module)
    importlib.reload(llm_module)
    sel = llm_module.llm_selector()
    from langchain_ollama import ChatOllama

    assert isinstance(sel, ChatOllama)
