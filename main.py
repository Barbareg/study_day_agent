"""Terminal (CLI) version of Study Day Assistant."""
import getpass
import os
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import build_graph


def main():
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your GROQ_API_KEY (hidden): ")

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Study Day Assistant. Type 'exit' to quit.")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if not text:
            continue
        try:
            result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
            # Human approval loop for destructive actions
            while "__interrupt__" in result:
                question = result["__interrupt__"][0].value
                answer = input(f"⏸ {question} (yes/no): ").strip().lower()
                result = graph.invoke(Command(resume=answer in ("yes", "y")), config)
            print("Assistant:", result["messages"][-1].content)
        except Exception as e:
            print(f"Error: the AI service did not respond ({e}). Please try again in a minute.")


if __name__ == "__main__":
    main()
