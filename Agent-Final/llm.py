from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import time, ast, os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.agents.agent_types import AgentType
from utility_functions import transcribe_folder_to_csv, process_folder_vad, save_num_speakers, transcript_quality, force_alignment_and_ctc_score

llm = Ollama(model="llama3.3:70b")

folder_path = "/root/abhinav/shrutilipi/newsonair_v5/hindi/Regional-Bhopal-Hindi-0705-2018726820"
file_path = "/root/abhinav/shrutilipi/newsonair_v5/hindi/Regional-Bhopal-Hindi-0705-2018726820/transcriptions.txt"
csv_file_path = os.path.join(folder_path, "new_transcriptions.csv")

user_prompt = "I need hypothesis of these audios and then vocab list as well as character list"

prompt_1 = f"""You are given the following functions:
1. ASR Transcription
2. Number of Speaker calculation
3. Quality of Transcript
4. Graphene or character calculation
5. Vocab calculation
6. Language identification
7. Audio length calculation
8. Silence calculation (using VAD)
9. Sample rate check
10. CTC score calculation
11. Transcript quality check

Based on the prompt, reply with task numbers that have to be done without any explanation or reasoning.
Example: 1,3,5

Prompt: {user_prompt}
"""
resp_1 = llm.invoke(prompt_1)
print("Response:", resp_1)

prompt_2 = f"""You are given the following functions:
1. ASR Transcription using audio files
2. Number of Speaker calculation using audio files
3. Quality of Transcript using audio files and ground truth file
4. Graphene or character calculation using ground truth file
5. Vocab calculation using ground truth file
6. Language identification using ground truth file
7. Audio length calculation using audio files
8. Silence calculation (using VAD) audio files
9. Sample rate check using audio files
10. CTC score calculation using audio files and ground truth file
11. Transcript quality check using ground truth file

We have to do tasks: {resp_1}.
Make a Topological sorting for what is the best way to proceed with these tasks, sequentially and concurrently.
Example: [[1,3], [5], [8]] (this means do 1 and 3 concurrently, then do 5, and finally do 8)
We can do tasks concurrently if they are independent of each other. 

Finally, give me the topological sorting for the tasks: {resp_1} without any explanation or reasoning.
"""
resp_2 = llm.invoke(prompt_2)
print("Response:", resp_2)
resp_2 = ast.literal_eval(resp_2)

class CSVTaskState(TypedDict, total=False):
    folder_path: str
    corruption_output: str
    extension_output: str
    conversion_output: str
    sample_rate_output: str

def start_node(state: CSVTaskState) -> dict:
    return state

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

def corruption_agent(state: CSVTaskState) -> dict:
    task_prompt = f"""You are given a folder with audios at this path: {state['folder_path']}.
Write a Python script to:
- Attempt to open and read each audio file.
- If a file fails to load or raises an error, mark it as corrupted and capture the error message.
Save a CSV listing all files and their status ("Corrupt" or "Valid") as audio_validity.csv in the same directory.

Finally, Respond with "Success" if all files are valid, otherwise "Invalid".
"""
    print("Beginning corruption check at", time.time())
    response = agent.invoke(task_prompt)
    return {"corruption_output": response}

def extension_agent(state: CSVTaskState) -> dict:
    task_prompt = f"""You are given a folder with audios at this path: {state['folder_path']}.
Write a Python script to:
1. Confirm that each file except {file_path} has a valid audio extension (only .wav or .mp3).
2. For audio files, also check if they are in WAV format by attempting to read them using a library like wave or librosa.
3. Create a CSV with columns: Filename, Valid_Extension, Is_WAV_Format, Status
4. Status should be "Pass" only if both extension is valid and format is WAV.
5. Save the CSV as audio_format_check.csv in the same directory.

Respond with "Success" if all files pass, otherwise "Invalid".
"""
    print("Beginning extension and format check at", time.time())
    response = agent.invoke(task_prompt)
    return {"extension_output": response}

def sample_rate_agent(state: CSVTaskState) -> dict:
    task_prompt = f"""You are given a folder with audio files at this path: {state['folder_path']}.
Write a Python script to:
1. Check each audio file's sample rate
2. Create a CSV with columns: Filename, Sample_Rate, Status
3. Store "Pass" in Status if sample rate is 16000 Hz, otherwise "Fail"
4. Save the CSV as sample_rate_check.csv in the same directory

Use libraries like librosa, soundfile, or wave to check the sample rate.
"""
    print("Beginning sample rate check at", time.time())
    response = agent.invoke(task_prompt)
    return {"sample_rate_output": response}

def ground_file_conversion_agent(state: CSVTaskState) -> dict:
    task_prompt = f"""You are given a file of ground truths of audios {state['folder_path']} at {file_path}.
1. Get the structure of the txt, csv, json, xml file.
2. Identify the element/column that contains the filename and transcriptions(ground truth). If there is no such column, return "Invalid".
2. Convert the file to CSV with added columns of Filename and Transcription.
3. Save the updated CSV with the new column to the same directory as new_transcriptions.csv.
Finally, Respond with "Success" if all steps are done, otherwise "Invalid".
"""
    print("Beginning file conversion at", time.time())
    response = agent.invoke(task_prompt)
    return {"conversion_output": response}

qc_builder = StateGraph(CSVTaskState)

qc_builder.add_node("start", start_node)
qc_builder.add_node("corruption_agent", corruption_agent)
qc_builder.add_node("extension_agent", extension_agent)
qc_builder.add_node("sample_rate_agent", sample_rate_agent)
qc_builder.add_node("ground_file_conversion_agent", ground_file_conversion_agent)

qc_builder.set_entry_point("start")

qc_builder.add_edge("start", "corruption_agent")
qc_builder.add_edge("start", "extension_agent")
qc_builder.add_edge("start", "sample_rate_agent")
qc_builder.add_edge("start", "ground_file_conversion_agent")

qc_builder.add_edge("corruption_agent", END)
qc_builder.add_edge("extension_agent", END)
qc_builder.add_edge("sample_rate_agent", END)
qc_builder.add_edge("ground_file_conversion_agent", END)

qc_graph = qc_builder.compile()

initial_state = {
    "folder_path": folder_path
}

qc_result = qc_graph.invoke(initial_state)
print("QC Check Results:")
print("Corruption Check Output:", qc_result.get("corruption_output", "N/A"))
print("Extension Check Output:", qc_result.get("extension_output", "N/A"))
print("Sample Rate Check Output:", qc_result.get("sample_rate_output", "N/A"))
print("Conversion Check Output:", qc_result.get("conversion_output", "N/A"))

class CombinedStateDict(TypedDict, total=False):
    A: str
    D: str
    E: str
    character_output: str
    vocab_output: str
    audio_length_output: str
    folder_path: str
    ctc_score_output: str
    transcript_quality_output: str
    language_identification_output: str

def transcription_func(state: CombinedStateDict) -> CombinedStateDict:
    print("Running Transcription")
    result = transcribe_folder_to_csv(folder_path, source_language="Hindi")
    return {"A": result, "folder_path": folder_path}

def silence_vad_func(state: CombinedStateDict) -> CombinedStateDict:
    print("Running Silence Detection")
    result = process_folder_vad(folder_path)
    return {"D": result}

def num_speaker_func(state: CombinedStateDict) -> CombinedStateDict:
    print("Running Speaker Diarization")
    result = save_num_speakers(folder_path)
    return {"E": result}

def vocab_agent(state: CombinedStateDict) -> CombinedStateDict:
    print("Running vocab_agent")
    task_prompt = f"""You are given a CSV file at this path: {csv_file_path}.
It has a column called 'Transcription'.

Write a Python script to:
1. Load the CSV.
2. For each row, extract a list of unique words (vocabulary) from the transcription and store it in a new column called 'vocab_list'.
3. Save the updated CSV with the new column to the same directory as vocab_list.csv.
"""
    response = agent.invoke(task_prompt)
    return {"vocab_output": response}

def character_agent(state: CombinedStateDict) -> CombinedStateDict:
    print("Running character_agent")
    task_prompt = f"""You are given a CSV file at this path: {csv_file_path}.
It has a column called 'Transcription'.

Write a Python script to:
1. Load the CSV.
2. For each row, extract a list of unique characters from the transcription and store it in a new column called 'character_list'.
3. Save the updated CSV with the new column to the same directory as character_list.csv.
"""
    response = agent.invoke(task_prompt)
    return {"character_output": response}

def audio_length_agent(state: CombinedStateDict) -> CombinedStateDict:
    print("Running audio_length_agent")
    task_prompt = f"""You are given audio files in the folder: {state['folder_path']}.

Write a Python script to:
1. Create a CSV with columns Filename and Audio_length.
2. For each audio file, calculate its duration in seconds and store it in 'Audio_length' column.
3. Save the CSV as audio_length.csv in the same folder.
"""
    response = agent.invoke(task_prompt)
    return {"audio_length_output": response}

def language_identification_agent(state: CombinedStateDict) -> CombinedStateDict:
    print("Running language identification")
    task_prompt = f"""You are given a CSV file at this path: {csv_file_path}.
It has a column called 'Transcription'.

Write a Python script to:
1. Load the CSV
2. For each row's transcription, identify the language using a language detection library (like langdetect, fastText, or spacy)
3. Store the detected language code in a new column called 'Detected_Language'
4. Compare with expected language (Hindi) and add a 'Language_Match' column (True/False)
5. Save the updated CSV with the new columns to the same directory as language_identification.csv

Install any required packages if needed.
"""
    response = agent.invoke(task_prompt)
    return {"language_identification_output": response}

def ctc_score_agent(state: CombinedStateDict) -> CombinedStateDict:
    print("Running CTC score calculation")
    task_prompt = f"""You are given a folder with audio files at this path: {state['folder_path']} and a CSV file at {csv_file_path} containing transcriptions.

Write a Python script to:
1. Load the CSV file with transcriptions
2. For each audio file and its corresponding transcription, calculate the CTC score using the force_alignment_and_ctc_score function from utility_functions
3. Store the CTC score for each audio-transcript pair in a new column called 'CTC_Score'
4. Add a 'CTC_Status' column with "Good" if score > 0.7, "Medium" if score > 0.5, "Poor" otherwise
5. Save the updated CSV with the new columns to the same directory as ctc_scores.csv

Handle any errors gracefully, as some audio files or transcripts might cause issues.
"""
    response = agent.invoke(task_prompt)
    return {"ctc_score_output": response}

def transcript_quality_agent(state: CombinedStateDict) -> CombinedStateDict:
    print("Running transcript quality check")
    task_prompt = f"""You are given a CSV file at this path: {csv_file_path} containing transcriptions.

Write a Python script to check the quality of each transcript using the transcript_quality function from utility_functions.

Your script should:
1. Load the CSV file
2. Apply the transcript_quality function to each transcript
3. Store the result in a new column called 'Quality_Check'
4. Save the updated CSV with the new column to the same directory as transcript_quality.csv
"""
    response = agent.invoke(task_prompt)
    return {"transcript_quality_output": response}

node_map = {
    1: ("node_transcription", transcription_func, "A"),
    2: ("node_num_speaker", num_speaker_func, "E"),
    3: ("node_transcript_quality", transcript_quality_agent, "transcript_quality_output"),
    4: ("node_character", character_agent, "character_output"),
    5: ("node_vocab", vocab_agent, "vocab_output"),
    6: ("node_language_identification", language_identification_agent, "language_identification_output"),
    7: ("node_audio_length", audio_length_agent, "audio_length_output"),
    8: ("node_silence", silence_vad_func, "D"),
    9: ("node_sample_rate", sample_rate_agent, "sample_rate_output"),
    10: ("node_ctc_score", ctc_score_agent, "ctc_score_output"),
}

def build_graph_from_structure(structure: list[list[int]]):
    graph_builder = StateGraph(CombinedStateDict)

    for group in structure:
        for node_id in group:
            if node_id in node_map:
                node_name, func, _ = node_map[node_id]
                graph_builder.add_node(node_name, func)

    for i in range(len(structure) - 1):
        for src in structure[i]:
            if src in node_map:
                for dst in structure[i + 1]:
                    if dst in node_map:
                        graph_builder.add_edge(node_map[src][0], node_map[dst][0])

    first_valid_node = None
    for node_id in structure[0]:
        if node_id in node_map:
            first_valid_node = node_id
            break
    
    if first_valid_node is not None:
        graph_builder.set_entry_point(node_map[first_valid_node][0])
    else:
        raise ValueError("No valid nodes found in the first group")

    for node_id in structure[-1]:
        if node_id in node_map:
            graph_builder.add_edge(node_map[node_id][0], END)

    return graph_builder.compile()

structure = resp_2
print("Using structure:", structure)

main_graph = build_graph_from_structure(structure)
main_graph.get_graph().print_ascii()

initial_state = {"folder_path": folder_path}
final_result = main_graph.invoke(initial_state)

print("\nFinal Pipeline Results:")
for key, value in final_result.items():
    if key != "folder_path":
        print(f"{key}: {value[:100]}..." if isinstance(value, str) and len(value) > 100 else f"{key}: {value}")
