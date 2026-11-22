"""Streamlit chat UI for Study Day Assistant."""
import os
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

st.set_page_config(page_title="Study Day Assistant", page_icon="📚")

# API key: environment variable (local) or Streamlit secrets (online). Never hard-coded.
key = os.environ.get("GROQ_API_KEY")
if not key:
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = None
if not key:
    st.error("GROQ_API_KEY is not set. Add it to Streamlit Secrets or set it as an environment variable.")
    st.stop()
os.environ["GROQ_API_KEY"] = key

from agent import build_graph, get_tasks  # noqa: E402 (after the key is set)


@st.cache_resource
def get_graph():
    """Build the graph once, not on every message."""
    return build_graph()


graph = get_graph()

# Session state: separate conversation for every user
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat" not in st.session_state:
    st.session_state.chat = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def run(graph_input):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        result = graph.invoke(graph_input, config)
    except Exception:
        st.session_state.pending = None
        st.session_state.chat.append((
            "assistant",
            "Sorry, the AI service is not responding or the rate limit was reached. "
            "Please wait a minute and try again.",
        ))
        return
    if "__interrupt__" in result:
        st.session_state.pending = result["__interrupt__"][0].value
    else:
        st.session_state.pending = None
        st.session_state.chat.append(("assistant", result["messages"][-1].content))


st.title("📚 Study Day Assistant")
st.caption("Helps students plan their study day: Tbilisi weather, study tasks and book search.")

with st.sidebar:
    st.header("Your tasks")
    tasks = get_tasks(st.session_state.thread_id)
    if tasks:
        for i, t in enumerate(tasks):
            st.write(f"{i}. {t['title']}")
    else:
        st.write("No tasks yet.")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat = []
        st.session_state.pending = None
        st.rerun()

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

# Approval step: shows the exact action and two buttons
if st.session_state.pending:
    st.warning(f"⏸ Approval needed: {st.session_state.pending}")
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirm"):
        run(Command(resume=True))
        st.rerun()
    if col2.button("❌ Refuse"):
        run(Command(resume=False))
        st.rerun()

prompt = st.chat_input("Ask me something...", disabled=bool(st.session_state.pending))
if prompt:
    st.session_state.chat.append(("user", prompt))
    run({"messages": [HumanMessage(content=prompt)]})
    st.rerun()
