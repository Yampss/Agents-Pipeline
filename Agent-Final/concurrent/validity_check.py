import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.agents.agent_types import AgentType

model = "llama3:70b"
llm = Ollama(model=model)
python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

tools = [repl_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

class CSVTaskState(TypedDict, total=False):
    folder_path: str
    corruption_output: str
    extension_output: str

# === Agents ===

def corruption_agent(state: CSVTaskState) -> dict:
    task_prompt = f"""You are given a folder with audios at this path: {state['folder_path']}.
Write a Python script to:
- Attempt to open and read each audio file.
- If a file fails to load or raises an error, mark it as corrupted and capture the error message.
Save a CSV listing all files and their status ("Corrupt" or "Valid") as audio_validity.csv in the same directory.

Finally, Respond with "Success" if all files are valid, otherwise "Invalid".
"""
    response = agent.invoke(task_prompt)
    return {"corruption_output": response}

def extension_agent(state: CSVTaskState) -> dict:
    if state.get("corruption_output") != "Success":
        print("Corruption check failed. Skipping extension check.")
        return state

    task_prompt = f"""You are given a folder with audios at this path: {state['folder_path']}.
Write a Python script to confirm that each file has a valid audio extension (only .wav or .mp3).
Skip any other file types (e.g., .txt, .json).

Respond with "Success" if all files are valid, otherwise "Invalid".
"""
    response = agent.invoke(task_prompt)
    return {"extension_output": response}


builder = StateGraph(CSVTaskState)
builder.add_node("corruption_agent", corruption_agent)
builder.add_node("extension_agent", extension_agent)

builder.set_entry_point("corruption_agent")
builder.add_conditional_edges(
    "corruption_agent",
    lambda state: "extension_agent" if state.get("corruption_output") == "Success" else END,
)
builder.add_edge("extension_agent", END)

graph = builder.compile()

initial_state = {
    "folder_path": "/root/abhinav/shrutilipi/newsonair_v5/hindi/Regional-Bhopal-Hindi-0705-201932960"
}

final_result = graph.invoke(initial_state)

# === Output ===
print("\n=== Final Outputs ===")
print("Corruption Check Output:\n", final_result.get("corruption_output", "N/A"))
print("Extension Check Output:\n", final_result.get("extension_output", "N/A"))
