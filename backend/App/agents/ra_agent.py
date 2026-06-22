import os
import arxiv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.messages import SystemMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

# Build custom tools
@tool
def search_arxiv(query: str) -> str:
    """
    Search the arXiv database for academic literature, preprints, and papers.
    Use this tool when the query requires academic, formal, or scientific research papers.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )
    try:
        results = []
        for result in client.results(search):
            results.append(
                f"Title: {result.title}\n"
                f"Authors: {', '.join([a.name for a in result.authors])}\n"
                f"Published: {result.published.strftime('%Y-%m-%d')}\n"
                f"ID: {result.get_short_id()}\n"
                f"URL: {result.pdf_url}\n"
                f"Abstract: {result.summary.strip()}\n"
            )
        if not results:
            return "No arXiv papers found matching the query."
        return "\n---\n".join(results)
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"

@tool
def fetch_arxiv_by_id(arxiv_id: str) -> str:
    """
    Fetch the detailed metadata and abstract of a specific academic paper using its arXiv ID.
    """
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    try:
        result = next(client.results(search))
        return (
            f"Title: {result.title}\n"
            f"Authors: {', '.join([a.name for a in result.authors])}\n"
            f"Published: {result.published.strftime('%Y-%m-%d')}\n"
            f"ID: {result.get_short_id()}\n"
            f"URL: {result.pdf_url}\n"
            f"Abstract: {result.summary.strip()}"
        )
    except StopIteration:
        return f"Error: No paper found with arXiv ID '{arxiv_id}'."
    except Exception as e:
        return f"Error fetching arXiv paper by ID: {str(e)}"

@tool
def search_web_tavily(query: str) -> str:
    """
    Search the web for general information, current events, developer docs, and news using Tavily.
    Use this for non-academic questions or general web searches.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "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")
    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(query=query, max_results=5)
        results = []
        for r in response.get("results", []):
            results.append(
                f"Title: {r.get('title', '')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Content: {r.get('content', '')}\n"
            )
        if not results:
            return "No web results found."
        return "\n---\n".join(results)
    except Exception as e:
        return f"Error searching the web: {str(e)}"

class ResearchAssistantAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=1024
        )
        tools_list = [search_arxiv, fetch_arxiv_by_id, search_web_tavily]
        self.llm_with_tools = self.llm.bind_tools(tools_list)
        self.tools_by_name = {tool.name: tool for tool in tools_list}

    def call_llm(self, state):
        return {
            "messages": [
                self.llm_with_tools.invoke(
                    [
                        SystemMessage(
                            content=(
                                "You are a professional Research Assistant Agent.\n"
                                "Your task is to search for academic papers or search the web to answer "
                                "the user's research query comprehensively and formulate a detailed summary.\n"
                                "Use the search_arxiv tool to search for scientific literature.\n"
                                "Use fetch_arxiv_by_id to get detail on a specific paper if needed.\n"
                                "Use search_web_tavily to find general information on the web.\n"
                                "Synthesize your findings and present a coherent, detailed, and well-structured research summary."
                            )
                        )
                    ]
                    + state['messages']
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }

    def tool_node(self, state):
        result = []
        for tool_call in state['messages'][-1].tool_calls:
            if 'self' in tool_call["args"]:
                del tool_call["args"]['self']
            tool_obj = self.tools_by_name[tool_call["name"]]
            observation = tool_obj.invoke(tool_call["args"])
            content_string = str(observation)
            result.append(
                ToolMessage(
                    content=content_string,
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": result}
