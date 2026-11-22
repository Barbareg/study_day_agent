"""Shared agent code for Study Day Assistant (used by app.py and main.py)."""
from collections import defaultdict
from typing import Annotated

import requests
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

MODEL = "llama-3.1-8b-instant"

# Tasks are stored per conversation (thread_id), so users never share a list.
_TASKS = defaultdict(list)


def get_tasks(thread_id: str) -> list:
    """Return the task list of one conversation (used by the UI sidebar)."""
    return _TASKS[thread_id]


def _thread_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("thread_id", "default")


@tool
def get_tbilisi_weather() -> dict:
    """Get current temperature in Tbilisi, in Celsius."""
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": 41.72, "longitude": 44.79, "current": "temperature_2m"},
        timeout=10,
    )
    r.raise_for_status()
    return {"temperature_c": r.json()["current"]["temperature_2m"]}


@tool
def add_task(title: str, config: RunnableConfig) -> str:
    """Add a study task to the to-do list."""
    _TASKS[_thread_id(config)].append({"title": title, "done": False})
    return f"Added task: {title}"


@tool
def list_tasks(config: RunnableConfig) -> list:
    """Show all tasks in the to-do list."""
    return _TASKS[_thread_id(config)]


@tool
def delete_task(index: int, config: RunnableConfig) -> str:
    """Delete a task by its number (starting from 0). Asks the human for approval first."""
    tasks = _TASKS[_thread_id(config)]
    if not (0 <= index < len(tasks)):
        return "Task not found."
    title = tasks[index]["title"]
    answer = interrupt(f"Delete task '{title}'? Type yes or no.")
    if answer is True or str(answer).strip().lower() in ("yes", "y", "true"):
        tasks.pop(index)
        return f"Deleted task: {title}"
    return "Deletion cancelled by the user."


@tool
def search_books(topic: str) -> list:
    """Search books by topic using Open Library. Returns a few titles and authors."""
    r = requests.get(
        "https://openlibrary.org/search.json",
        params={"q": topic, "limit": 3},
        timeout=20,
    )
    r.raise_for_status()
    return [
        {"title": d.get("title"), "author": (d.get("author_name") or ["Unknown"])[0]}
        for d in r.json().get("docs", [])
    ]


TOOLS = [get_tbilisi_weather, add_task, list_tasks, delete_task, search_books]

SYSTEM = SystemMessage(content=(
    "You are Study Day Assistant, a helpful assistant for students. "
    "Always reply in English. Use tools when needed: "
    "weather for Tbilisi, adding/listing/deleting tasks, and searching books. "
    "When the user asks to delete a task, you MUST call delete_task immediately. "
    "Never ask for confirmation yourself; the tool asks the human. "
    "If the tool says the deletion was cancelled, do NOT call delete_task again; "
    "just tell the user it was cancelled. "
    "When asked to plan a study day, check the Tbilisi weather first: "
    "suggest studying outdoors in a park if it is 18°C or warmer, "
    "otherwise suggest a library or café. "
    "Keep answers short and clear."
))


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph():
    """Build and compile the agent graph. GROQ_API_KEY must be set before calling."""
    llm_with_tools = ChatGroq(model=MODEL, temperature=0).bind_tools(TOOLS)

    def assistant(state: State):
        return {"messages": [llm_with_tools.invoke([SYSTEM] + state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile(checkpointer=MemorySaver())
