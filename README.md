# Study Day Assistant

An AI agent built with LangGraph and Groq that helps students plan their study day.
It checks the weather in Tbilisi, adds, lists and deletes study tasks (asking for human
approval before deleting), and searches for books using Open Library.
When asked to plan a study day, it suggests studying outdoors in a park if it is 18 °C or
warmer, and a library or café otherwise.

## Tools

| Tool | What it does |
|---|---|
| `get_tbilisi_weather` | Current temperature in Tbilisi (Open-Meteo API) |
| `add_task` / `list_tasks` | Add and show study tasks |
| `delete_task` | Deletes a task, but only after human approval (`interrupt()`) |
| `search_books` | Finds a few books by topic (Open Library API) |

## How it works

- **LangGraph StateGraph:** an `assistant` node and a `tools` node with conditional routing
- **Memory:** `MemorySaver` checkpointer, one `thread_id` per conversation
- **Approval flow:** the graph pauses before deleting; resume with `Command(resume=True/False)`
- **Per-user tasks:** tasks are stored per `thread_id`, so users do not share a list

## Files

| File | Purpose |
|---|---|
| `agent.ipynb` | Colab notebook: tools, graph, diagram, tests, evaluation table |
| `agent.py` | Shared agent code with `build_graph()` |
| `app.py` | Streamlit chat page |
| `main.py` | Terminal version |
| `requirements.txt` | Python dependencies |

## Run locally

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py                      # terminal chat (asks for the key, hidden)
export GROQ_API_KEY="your-key"
python -m streamlit run app.py      # web chat
```

Get a free key at https://console.groq.com. Never commit your key.
For Streamlit online, add `GROQ_API_KEY = "..."` in **Advanced settings → Secrets**.

## Limitations

- Tasks and memory live in RAM and are lost when the app restarts.
- The model sometimes answers in plain text instead of calling a tool; the system prompt reduces this.
- The Groq free tier has rate limits (the app shows a friendly message if they are hit).
