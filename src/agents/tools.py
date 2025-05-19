from datetime import datetime, timedelta
import requests
from langchain.tools import tool
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
import os, certifi
from langchain_core.tools import Tool

load_dotenv()

# point Requests (and duckduckgo_search under-the-hood) at certifi’s CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
import requests

requests.packages.urllib3.disable_warnings()
session = requests.Session()
session.verify = False


@tool("internet_search_DDGO", return_direct=False)
def internet_search_DDGO(query: str) -> str:
    """Searches the internet using DuckDuckGo."""

    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return results if results else "No results found."


@tool("process_content", return_direct=False)
def process_content(url: str) -> str:
    """Processes content from a webpage."""

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()


@tool("internet_search", return_direct=False)
def internet_search(query: str) -> str:
    """Searches the internet using Tavily."""
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
    results = search_tool.invoke(query)

    # Log the raw results for debugging purposes
    print("Raw results:", results)

    if isinstance(results, list) and all(
        isinstance(result, dict) for result in results
    ):
        formatted_results = ""
        references = []
        for i, result in enumerate(results):
            title = result.get("title", "No Title")
            url = result.get("url", "No URL")
            snippet = result.get("snippet", "No Snippet")
            formatted_results += f"{i+1}. {title}\n{snippet} [^{i+1}]\n\n"
            references.append(f"[^{i+1}]: [{title}]({url})")

        references_section = "\n**References:**\n" + "\n".join(references)
        return formatted_results + references_section

    else:
        return (
            "Unexpected result format. Please check the Tavily API response structure."
        )


@tool("get_today")
def get_today() -> str:
    """Provides information about today's date."""
    now = datetime.now()
    return "Today is "+now.strftime("%Y-%m-%d, %A")

@tool("get_tomorrow")
def get_tomorrow() -> str:
    """Provides information about tomorrow's date."""
    tomorrow = datetime.now() + timedelta(days=1)
    return "Tomorrow is "+tomorrow.strftime("%Y-%m-%d, %A")


def get_tools():
    return [
        internet_search,
        process_content,
        Tool.from_function(
            get_today,
            name="get_today",
            description="Provides information about today's date.",
        ),
        Tool.from_function(
            get_tomorrow,
            name="get_tomorrow",
            description="Provides information about tomorrow's date.",
        ),
    ]  # Uncomment this and comment the line below to use Tavily instead of DuckDuckGo Search.

    # return [internet_search_DDGO, process_content]  # Uncomment this and comment the line above to use DuckDuckGo Search instead of Tavily.


def main():
    print(internet_search("paneer"))


if __name__ == "__main__":
    main()
