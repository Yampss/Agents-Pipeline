import json
import os
import pandas as pd
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.agents.agent_types import AgentType

model = "llama3.3:70b"  
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

folder_path = '/root/abhinav/shrutilipi/newsonair_v5/hindi/Regional-Bhopal-Hindi-0705-201932960'

task_prompt = f"""
You are given a folder of audio files located at: '{folder_path}'.

Your task is to write Python code and perform validation checks on all the **audio files only** in this directory. You must skip any non-audio files such as `.txt`, `.csv`, `.json`, etc.

Only include files with the following valid audio extensions: `.wav`, `.mp3`.

For the valid audio files, perform the following checks:

1. **Extension Check**: 
   - Confirm that the file has a valid audio extension (only `.wav` or `.mp3`).
   - Skip any other files (e.g., `.txt`, `.json`) and do not include them in the final report.

2. **Sample Rate Check**:
   - Ensure that the audio file has a sample rate of at least 16,000 Hz (16kHz).

3. **Corruption Check**:
   - Attempt to open and read each audio file.
   - If a file fails to load or raises an error, mark it as corrupted and capture the error message.

For each **valid audio file**, output the following:

File: "filename"
Extension Valid: True
Sample Rate ≥ 16kHz: True/False (If False, include actual sample rate)
Corrupted: True/False (If True, include error message)

After checking all audio files:
- Summarize the results into a CSV file named `'audio_validity_report.csv'` in the same directory.
- The CSV must contain these columns:
  - 'File'
  - 'Extension Valid'
  - 'Sample Rate ≥ 16kHz'
  - 'Corrupted'

Do NOT include skipped (non-audio) files in the report.

Finally:
- If all audio files pass every validation check (Extension Valid = True, Sample Rate ≥ 16kHz = True, Corrupted = False), respond with: "Validation checks: Successful".
- Otherwise, respond with: "Validation checks: Fail".

Do NOT generate or display any Python code after this.

The task is finished and stop when you have completed the task.
"""


response = agent.invoke(task_prompt)
print(response)
