import os
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
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
    file_path: str
    character_output: str
    vocab_output: str
    langs: str
    audio_length_output: str

def vocab_agent(state: CSVTaskState) -> dict:
    task_prompt = f"""You are given a CSV file at this path: {state['file_path']}.
    It has a column called 'Transcription'.

    Write a Python script to:
    1. Load the CSV.
    2. For each row, extract a list of unique words (vocabulary) from the transcription and store it in a new column called 'vocab_list'.
    3. Save the updated CSV with the new column to the same directory as vocab_list.csv.
    """
    response = agent.invoke(task_prompt)
    return {"vocab_output": response}

def character_agent(state: CSVTaskState) -> dict:
    task_prompt = f"""You are given a CSV file at this path: {state['file_path']}.
    It has a column called 'Transcription'.

    Write a Python script to:
    1. Load the CSV.
    2. For each row, extract a list of unique characters from the transcription and store it in a new column called 'character_list'.
    3. Save the updated CSV with the new column to the same directory as character_list.csv.
    """
    response = agent.invoke(task_prompt)
    return {"character_output": response}


def audio_length_agent(state: CSVTaskState) -> dict:
    task_prompt = f"""You are audios at this folder: {state['file_path']}.
    
    Write a Python script to:
    1. Make a csv with columns Filename and Audio_length.
    2. For each audio, calulcate total length of audio in 'Audio_length' column.
    3. Save the CSV as audio_length.csv.
    """
    response = agent.invoke(task_prompt)
    return {"audio_length_output": response}




builder = StateGraph(CSVTaskState)

# builder.add_node("get_languages", get_languages)
builder.add_node("character_agent", character_agent)
builder.add_node("vocab_agent", vocab_agent)
# builder.add_node("transcription_agent", transcription_agent)

builder.set_entry_point("character_agent")
# builder.add_edge("get_languages", "character_agent")
builder.add_edge("character_agent", "vocab_agent")
# builder.add_edge("vocab_agent", "transcription_agent")
# builder.add_edge("transcription_agent", END)

graph = builder.compile()

# --- Run Example ---

initial_state = {
    "file_path": "/root/abhinav/ai_agent/synthetic_data/Output/Hindi/IndianAgricultureAndFarming/extracted_transcriptions.csv"
}

final_result = graph.invoke(initial_state)

print("\n=== Final Outputs ===")
print("Character Output:\n", final_result.get("character_output", "N/A"))
print("Vocab Output:\n", final_result.get("vocab_output", "N/A"))
print("Languages Detected:\n", final_result.get("langs", "N/A"))
print("Transcription Output:\n", final_result.get("transcription_output", "N/A"))
