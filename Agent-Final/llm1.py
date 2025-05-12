import json
import logging
import os
import re
from typing import Dict, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor
import ast
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, END
from utility_functions1 import (
    transcribe_folder_to_csv,
    process_folder_vad,
    save_num_speakers,
    process_audio_directory,
    transcript_quality,
    force_alignment_and_ctc_score,
    check_upsampling_folder,
    language_identification_indiclid,
    transliterate_file,
    get_required_inputs
)
import pandas as pd


logging.basicConfig(filename="pipeline.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize LLM with updated ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key="sk-proj-J0eiEMCmbATmMs3jRYQtcZcgCgFYuLxGx4A"
)


def parse_prompt(prompt: str) -> Dict[str, Optional[str]]:
    result = {
        "audio_dir": None,
        "ground_truth_csv": None,
        "lang_code": None
    }
    
    path_pattern = r"['\"]?(/[^'\"]+/[^'\"]+/)['\"]?"
    csv_pattern = r"['\"]?(/[^'\"]+\.csv)['\"]?"
    lang_pattern = r"\b(bn|gu|hi|kn|ml|mr|pa|sd|si|ta|te|ur)\b"
    
    audio_dir_match = re.search(path_pattern, prompt)
    if audio_dir_match:
        result["audio_dir"] = audio_dir_match.group(1)
    
    csv_match = re.search(csv_pattern, prompt)
    if csv_match:
        result["ground_truth_csv"] = csv_match.group(1)
    
    lang_match = re.search(lang_pattern, prompt, re.IGNORECASE)
    if lang_match:
        result["lang_code"] = lang_match.group(1).lower()
    
    if not result["audio_dir"] and not result["ground_truth_csv"]:
        llm_prompt = f"""Extract the following from the prompt:
        1. Audio directory path (e.g., /path/to/audio/)
        2. Ground truth CSV path (e.g., /path/to/file.csv)
        3. Language code (e.g., hi, te)
        If any are unclear, return None for that field.
        Prompt: {prompt}
        Return a JSON object with keys 'audio_dir', 'ground_truth_csv', 'lang_code'.
        """
        try:
            response = llm.invoke(llm_prompt)
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            llm_result = json.loads(content)
            for key in result:
                if not result[key]:
                    result[key] = llm_result.get(key)
        except Exception as e:
            logging.error(f"LLM prompt parsing failed: {e}")
            # Fallback to regex if LLM fails
            if audio_dir_match:
                result["audio_dir"] = audio_dir_match.group(1)
            elif "Audio_dir" in prompt:
                # Extract path after Audio_dir
                audio_dir_match = re.search(r"Audio_dir\s*=\s*['\"]?(/[^'\"]+/)['\"]?", prompt)
                if audio_dir_match:
                    result["audio_dir"] = audio_dir_match.group(1)
    
    if result["audio_dir"] and not os.path.isdir(result["audio_dir"]):
        logging.error(f"Invalid audio directory: {result['audio_dir']}")
        result["audio_dir"] = None
    if result["ground_truth_csv"] and not os.path.isfile(result["ground_truth_csv"]):
        logging.error(f"Invalid ground truth CSV: {result['ground_truth_csv']}")
        result["ground_truth_csv"] = None
    
    return result


class CombinedStateDict(TypedDict, total=False):
    audio_dir: str
    ground_truth_csv: str
    lang_code: str
    user_prompt: str
    A: str
    D: str
    E: str
    character_output: str
    vocab_output: str
    audio_length_output: str
    ctc_score_output: str
    language_verification_output: str
    upsampling_output: str
    valid_speaker_output: str
    domain_checker_output: str
    audio_transcript_matching_output: str
    language_identification_indiclid_output: str
    normalization_remove_tags_output: str
    llm_score_output: str
    transliteration_output: str
    corruption_output: str
    extension_output: str
    sample_rate_output: str

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
def select_tasks(user_prompt: str) -> str:
    prompt_1 = f"""You are given the following functions:
    1. ASR Transcription
    2. Number of Speakers calculation and duration per speaker
    3. Quality of Transcript
    4. Graphene or character calculation
    5. Vocab calculation
    6. Language verification (verify if transcriptions match an expected language)
    7. Audio length calculation
    8. Silence calculation (using VAD)
    9. Sample rate check
    10. CTC score calculation
    11. Upsampling Check
    12. Check if speakers are new or old
    13. Check the domain of the speech dataset
    14. Map transcriptions to audio files using forced alignment
    15. Language identification using ASR transcriptions and IndicLID
    16. Normalization by removing HTML and other tags from transcriptions in JSON or XML files
    17. Evaluate transcript coherence and fluency using LLM-as-a-Judge and score out of 10
    18. Transliteration - Convert Roman script words to Native script using Transliteration
    19. Audio corruption check
    20. Audio extension and format check
    21. Audio sample rate check

    Based on the prompt, identify the task numbers that must be executed to fulfill the request.
    - If the prompt mentions 'Vocab calculation', include task 5.
    - If the prompt mentions 'Character calculation', include task 4.
    - If the prompt mentions 'verify' and 'language' or 'expected language', include task 6.
    - If the prompt mentions 'language identification' or 'IndicLID', include tasks 1 and 15 (task 15 requires task 1 to generate transcriptions).
    - Include QC tasks (19, 20, 21) only if explicitly mentioned (e.g., 'corruption', 'extension', 'sample rate')
    - Only include tasks explicitly relevant to the prompt.
    Return the task numbers as a comma-separated string (e.g., '4,5,6').
    Prompt: {user_prompt}
    """
    resp_1 = llm.invoke(prompt_1).content
    # Clean resp_1 to extract task numbers
    task_numbers = re.findall(r'\b\d+\b', resp_1)
    expected_tasks = set()
    if 'Vocab' in user_prompt:
        expected_tasks.add('5')
    if 'Character' in user_prompt:
        expected_tasks.add('4')
    if 'verify' in user_prompt.lower() and 'language' in user_prompt.lower():
        expected_tasks.add('6')
    if expected_tasks:
        task_numbers = list(expected_tasks)
    resp_1 = ','.join(task_numbers) if task_numbers else ''
    logging.info(f"Selected Tasks: {resp_1}")
    return resp_1

def topological_sort_tasks(resp_1: str) -> list[list[int]]:
    prompt_2 = f"""You are given the following functions:
    1. ASR Transcription using audio files
    2. Number of Speakers calculation and duration per speaker using audio files
    3. Quality of Transcript using audio files and ground truth file
    4. Graphene or character calculation using ground truth file
    5. Vocab calculation using ground truth file
    6. Language verification (verify if transcriptions match an expected language) using ground truth file
    7. Audio length calculation using audio files
    8. Silence calculation (using VAD) audio files
    9. Sample rate check using audio files
    10. CTC score calculation using audio files and ground truth file
    11. Upsampling Check using audio files
    12. Check if speakers are new or old using the results from number of speakers calculation
    13. Check the domain of the speech dataset using transcriptions from ASR
    14. Map transcriptions to audio files using forced alignment, using ground truth transcriptions
    15. Language identification using ASR transcriptions and IndicLID, using transcriptions from ASR
    16. Normalization by removing HTML and other tags from transcriptions in JSON or XML files
    17. Evaluate transcript coherence and fluency using LLM-as-a-Judge and score out of 10
    18. Transliteration - Convert Roman script words to Native script using Transliteration
    19. Audio corruption check using audio files
    20. Audio extension and format check using audio files
    21. Audio sample rate check using audio files

    We have to do tasks: {resp_1}.
    Make a topological sorting to execute these tasks efficiently:
    - Include ONLY the tasks listed in: {resp_1}. Do not include any other tasks.
    - Place independent tasks in the same group to run concurrently.
    - Respect dependencies: Task 12 depends on task 2; Task 13 depends on task 1; Task 15 depends on task 1; Task 17 depends on task 1; Task 14 depends on ground truth availability.
    - Use the provided audio directory when needed.
    Return the sorting as a list of lists, e.g., [[6]] for task 6 alone.
    Ensure the output is a valid Python list of lists, enclosed in square brackets.
    If no valid structure can be formed, return a list containing a single group with the tasks from the input, e.g., [[6]] for input '6'.
    """
    try:
        resp_2 = llm.invoke(prompt_2).content
        resp_2 = resp_2.strip()
        if not resp_2.startswith('[') or not resp_2.endswith(']'):
            resp_2 = resp_2.replace('```python', '').replace('```', '').strip()
            resp_2 = re.search(r'\[\[.*\]\]', resp_2, re.DOTALL)
            resp_2 = resp_2.group(0) if resp_2 else '[]'
        structure = ast.literal_eval(resp_2)
        if not isinstance(structure, list) or not all(isinstance(group, list) for group in structure):
            raise ValueError("Invalid structure: must be a list of lists")
        tasks = set(int(t) for t in resp_1.split(',') if t.strip().isdigit())
        structure = [[task for task in group if task in tasks] for group in structure]
        structure = [group for group in structure if group]
        if not structure:  # Fallback to tasks in resp_1
            structure = [list(tasks)] if tasks else []
        logging.info(f"Topological Structure: {structure}")
        return structure
    except Exception as e:
        logging.error(f"Failed to parse topological structure: {e}")
        tasks = [int(t) for t in resp_1.split(',') if t.strip().isdigit()]
        return [tasks] if tasks else []



def corruption_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for corruption check: {audio_dir}")
        return {"corruption_output": "Invalid: No audio directory"}
    task_prompt = f"""You are given a folder with audios at this path: {audio_dir}.
Write a Python script to:
- Attempt to open and read each audio file.
- If a file fails to load or raises an error, mark it as corrupted and capture the error message.
Save a CSV listing all files and their status ("Corrupt" or "Valid") as audio_validity.csv in the same directory.

Finally, Respond with "Success" if all files are valid, otherwise "Invalid".
"""
    try:
        response = agent.invoke(task_prompt)
        return {"corruption_output": response.get("output", "Invalid")}
    except Exception as e:
        logging.error(f"Corruption check failed: {e}")
        return {"corruption_output": f"Error: {e}"}

def extension_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for extension check: {audio_dir}")
        return {"extension_output": "Invalid: No audio directory"}
    task_prompt = f"""You are given a folder with audios at this path: {audio_dir}.
Write a Python script to:
1. Confirm that each file has a valid audio extension (only .wav or .mp3).
2. For audio files, check if they are in WAV format by attempting to read them using a library like wave or librosa.
3. Create a CSV with columns: Filename, Valid_Extension, Is_WAV_Format, Status
4. Status should be "Pass" only if both extension is valid and format is WAV.
5. Save the CSV as audio_format_check.csv in the same directory.

Respond with "Success" if all files pass, otherwise "Invalid".
"""
    try:
        response = agent.invoke(task_prompt)
        return {"extension_output": response.get("output", "Invalid")}
    except Exception as e:
        logging.error(f"Extension check failed: {e}")
        return {"extension_output": f"Error: {e}"}

def sample_rate_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for sample rate check: {audio_dir}")
        return {"sample_rate_output": "Invalid: No audio directory"}
    task_prompt = f"""You are given a folder with audio files at this path: {audio_dir}.
Write a Python script to:
1. Check each audio file's sample rate
2. Create a CSV with columns: Filename, Sample_Rate, Status
3. Store "Pass" in Status if sample rate is 16000 Hz, otherwise "Fail"
4. Save the CSV as sample_rate_check.csv in the same directory

Use libraries like librosa, soundfile, or wave to check the sample rate.
"""
    try:
        response = agent.invoke(task_prompt)
        return {"sample_rate_output": response.get("output", "Invalid")}
    except Exception as e:
        logging.error(f"Sample rate check failed: {e}")
        return {"sample_rate_output": f"Error: {e}"}

def transcription_func(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for transcription: {audio_dir}")
        return {"A": "Error: Invalid audio directory"}
    logging.info("Running Transcription")
    result = transcribe_folder_to_csv(audio_dir, source_language="Hindi")
    return {"A": result, "audio_dir": audio_dir}

def silence_vad_func(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for silence detection: {audio_dir}")
        return {"D": "Error: Invalid audio directory"}
    logging.info("Running Silence Detection")
    result = process_folder_vad(audio_dir)
    return {"D": result}

def num_speaker_func(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for speaker diarization: {audio_dir}")
        return {"E": "Error: Invalid audio directory"}
    logging.info("Running Speaker Diarization and Duration Calculation")
    result = save_num_speakers(audio_dir)
    return {"E": result}

def vocab_agent(state: CombinedStateDict) -> CombinedStateDict:
    csv_path = state.get('ground_truth_csv')
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for vocab calculation: {csv_path}")
        return {"vocab_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running vocab_agent")
    task_prompt = f"""You are given a CSV file at this path: {csv_path}.
It has a column called 'Transcription' or 'Ground_Truth',(search case insensitively).

Write a Python script to:
1. For each row, extract a list of unique words (vocabulary) from the transcription and store it in a new column called 'vocab_list'.
2. Save the updated CSV with the new column to the same directory as vocab_list.csv

Respond with "Success" if the script completes
"""
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(os.path.dirname(csv_path), "vocab_list.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"vocab_output": f"CSV saved at: {output_path}"}
        else:
            error_msg = f"Failed to generate {output_path}"
            logging.error(error_msg)
            return {"vocab_output": error_msg}
    except Exception as e:
        logging.error(f"Vocab calculation failed: {e}")
        return {"vocab_output": f"Error: {e}"}

def character_agent(state: CombinedStateDict) -> CombinedStateDict:
    csv_path = state.get('ground_truth_csv')
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for character calculation: {csv_path}")
        return {"character_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running character_agent")
    task_prompt = f"""You are given a CSV file at this path: {csv_path}.
It has a column called 'Transcription' or 'Ground_Truth',(search case insensitively), take it as transcription coloumn.

Write a Python script to:
1. For each row, extract a list of unique characters from the transcription coloumn and store it in a new column called 'character_list'.
2. Save the updated CSV with the new column to the same directory as character_list.csv

Respond with "Success" if the script completes
"""
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(os.path.dirname(csv_path), "character_list.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"character_output": f"CSV saved at: {output_path}"}
        else:
            error_msg = f"Failed to generate {output_path}"
            logging.error(error_msg)
            return {"character_output": error_msg}
    except Exception as e:
        logging.error(f"Character calculation failed: {e}")
        return {"character_output": f"Error: {e}"}

def audio_length_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for audio length calculation: {audio_dir}")
        return {"audio_length_output": "Error: Invalid audio directory"}
    logging.info("Running audio_length_agent")
    task_prompt = f"""You are given audio files in the folder: {audio_dir}.

Write a Python script to:
1. Create a CSV with columns Filename and Audio_length.
2. For each audio file, calculate its duration in seconds and store it in 'Audio_length' column.
3. Save the CSV as audio_length.csv in the same folder.
"""
    try:
        response = agent.invoke(task_prompt)
        return {"audio_length_output": response.get("output", "Invalid")}
    except Exception as e:
        logging.error(f"Audio length calculation failed: {e}")
        return {"audio_length_output": f"Error: {e}"}


# def language_verification_agent(state: CombinedStateDict) -> CombinedStateDict:
#     csv_path = state.get('ground_truth_csv')
#     expected_lang = state.get('lang_code')  
#     if not csv_path or not os.path.isfile(csv_path):
#         logging.error(f"Invalid ground truth CSV for language verification: {csv_path}")
#         return {"language_verification_output": f"Error: CSV file {csv_path} not found"}
#     if not expected_lang:
#         logging.error("No expected language provided for verification")
#         return {"language_verification_output": "Error: No expected language provided"}
#     logging.info(f"Running language verification for expected language: {expected_lang} with CSV: {csv_path}")
#     task_prompt = f"""You are given a CSV file at this path: {csv_path}.
# It has a column called 'Transcription' (case-insensitive).
# The expected language is '{expected_lang}' (e.g., 'hi' for Hindi).

# Use the `python_repl` tool to execute a Python script that:
# 1. Loads the CSV using pandas.
# 2. For each row's transcription, identifies the language using the `langdetect` library.
# 3. Adds a 'Detected_Language' column with the identified language code (e.g., 'hi', 'en').
# 4. Adds a 'Language_Match' column (True if the detected language matches '{expected_lang}', False otherwise).
# 5. Saves the updated CSV as 'language_verification.csv' in the same directory as the input CSV.
# 6. Handles errors gracefully (e.g., empty transcriptions, missing columns, file access issues).
# 7. Prints "Success" if the CSV is saved successfully, otherwise prints an error message with the specific exception.

# Here is the script to execute in `python_repl`:
# """
# import pandas as pd
# import os
# import logging
# from langdetect import detect, LangDetectException


# logging.basicConfig(filename='pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# csv_path = r'{csv_path}'
# output_path = os.path.join(os.path.dirname(csv_path), "language_verification.csv")
# expected_lang = '{expected_lang}'

# try:
#     logging.info(f"Attempting to read CSV: {{csv_path}}")
#     df = pd.read_csv(csv_path)
#     transcription_col = next((col for col in df.columns if col.lower() == 'transcription'), None)
#     if not transcription_col:
#         raise ValueError("No 'Transcription' column found in CSV")
#     logging.info(f"Found transcription column: {{transcription_col}}")
    
#     def detect_language(text):
#         if pd.isna(text) or str(text).strip() == "":
#             logging.warning(f"Empty or null transcription encountered")
#             return None
#         try:
#             lang = detect(str(text))
#             logging.info(f"Detected language for text '{{text[:50]}}...': {{lang}}")
#             return lang
#         except LangDetectException as e:
#             logging.warning(f"LangDetectException for text '{{text[:50]}}...': {{e}}")
#             return None
    
#     df['Detected_Language'] = df[transcription_col].apply(detect_language)
#     df['Language_Match'] = df['Detected_Language'] == expected_lang
#     logging.info(f"Saving output CSV to: {{output_path}}")
#     df.to_csv(output_path, index=False)
#     print("Success")
# except Exception as e:
#     error_msg = f"Error in language verification: {str(e)}"
#     logging.error(error_msg)
#     print(error_msg)
def language_verification_agent(state: CombinedStateDict) -> CombinedStateDict:
    csv_path = state.get('ground_truth_csv')
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for language verification: {csv_path}")
        return {"language_verification_output": f"Error: CSV file {csv_path} not found"}
    logging.info(f"Running Devanagari script verification with CSV: {csv_path}")
    
    task_prompt = f"""You are a script recognition expert.

Your task is to determine whether the text in the 'ground_truth' column of a CSV file is written in the Devanagari script.

Please follow these steps:

Unicode Range for Devanagari Script:
The Unicode range for the Devanagari script is from U+0900 to U+097F.
This includes characters used in languages like Hindi, Sanskrit, Marathi, and others that use Devanagari as their writing system.

Steps:
1. Load the CSV file at this path: {csv_path}.
2. Identify the 'ground_truth' column (case-insensitive, e.g., 'Ground_Truth', 'transcription').
3. For each row in the 'ground_truth' column, check if all characters (excluding whitespace and punctuation) fall within the Unicode range U+0900 to U+097F.
4. Add a new column 'Is_Devanagari' with True if all relevant characters are in the Devanagari range, False otherwise.
5. If the transcription is empty or contains only whitespace/punctuation, set 'Is_Devanagari' to False.
6. Save the updated CSV as 'language_verification.csv' in the same directory as the input CSV.
7. Ensure the output CSV includes columns: 'Filename', 'Transcription' (the ground_truth text), and 'Is_Devanagari'.
8. Handle errors gracefully (e.g., missing columns, file access issues).

Example Text:
"नमस्ते, आप कैसे हैं?" -> Is_Devanagari: True (all characters are in U+0900 to U+097F)
"Hello" -> Is_Devanagari: False (characters are not in U+0900 to U+097F)

Use the `python_repl` tool to execute a Python script that performs these steps.

Respond with "Success" if the CSV is saved successfully, otherwise return an error message.
"""
    
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(os.path.dirname(csv_path), "language_verification.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            logging.info(f"Devanagari verification CSV saved to: {output_path}")
            return {"language_verification_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"Devanagari verification failed to generate {output_path}: {response.get('output', 'No output')}")
            return {"language_verification_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Devanagari verification failed: {e}")
        return {"language_verification_output": f"Error: {e}"}

def ctc_score_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = "/raid/ganesh/pdadiga/chriss/test_data"
    csv_path = "/raid/ganesh/pdadiga/chriss/test_data/new_transcriptions.csv"
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for CTC score: {audio_dir}")
        return {"ctc_score_output": "Error: Invalid audio directory"}
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for CTC score: {csv_path}")
        return {"ctc_score_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running CTC score calculation")
    try:
        output_path = os.path.join(os.path.dirname(csv_path), "ctc_scores.csv")
        # Call the batch processor directly
        results = process_audio_directory(audio_dir, csv_path, output_path)
        if results:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            if df.empty:
                logging.error("No valid results generated from process_audio_directory")
                return {"ctc_score_output": "Error: No valid results generated"}
            
            # Group by filename to create the required CSV format
            grouped = df.groupby('filename').agg({
                'label': lambda x: ' '.join(x),  # Aligned_Transcript
                'average_ctc_score': 'first',    # CTC_Score
            }).reset_index()
            
            # Add Aligned_Segments as a list of dictionaries
            grouped['Aligned_Segments'] = grouped['filename'].apply(
                lambda x: json.dumps([
                    {'label': row['label'], 'start': row['start_time'], 'end': row['end_time'], 'score': row['score']}
                    for _, row in df[df['filename'] == x].iterrows()
                ])
            )
            
            # Rename columns
            grouped.columns = ['Filename', 'Aligned_Transcript', 'CTC_Score', 'Aligned_Segments']
            
            # Add CTC_Status based on CTC_Score
            grouped['CTC_Status'] = grouped['CTC_Score'].apply(
                lambda x: "Good" if float(x) > 0.7 else "Medium" if float(x) > 0.5 else "Poor"
            )
            
            # Reorder columns
            grouped = grouped[['Filename', 'Aligned_Segments', 'Aligned_Transcript', 'CTC_Score', 'CTC_Status']]
            
            # Save to CSV
            grouped.to_csv(output_path, index=False)
            logging.info(f"CTC scores saved to: {output_path}")
            return {"ctc_score_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"No results generated for CTC score calculation")
            return {"ctc_score_output": f"Error: No results generated"}
    except Exception as e:
        logging.error(f"CTC score calculation failed: {e}")
        return {"ctc_score_output": f"Error: {e}"}

def transcript_quality_agent(state: CombinedStateDict) -> CombinedStateDict:
    csv_path = state.get('ground_truth_csv')
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for transcript quality: {csv_path}")
        return {"transcript_quality_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running transcript quality check")
    task_prompt = f"""You are given a CSV file at this path: {csv_path} containing transcriptions.

Write a Python script to check the quality of each transcript using the transcript_quality function from utility_functions.

Your script should:
1. Load the CSV file
2. Apply the transcript_quality function to each transcript
3. Store the result in a new column called 'Quality_Check'
4. Save the updated CSV with the new column to the same directory as transcript_quality.csv
"""
    try:
        response = agent.invoke(task_prompt)
        return {"transcript_quality_output": response.get("output", "Invalid")}
    except Exception as e:
        logging.error(f"Transcript quality check failed: {e}")
        return {"transcript_quality_output": f"Error: {e}"}

def upsampling_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    # audio_dir = "/raid/ganesh/pdadiga/chriss/test_data"
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for upsampling check: {audio_dir}")
        return {"upsampling_output": "Error: Invalid audio directory"}
    logging.info("Running upsampling check")
    result = check_upsampling_folder(audio_dir)
    return {"upsampling_output": result}

def valid_speaker_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for valid speaker check: {audio_dir}")
        return {"valid_speaker_output": "Error: Invalid audio directory"}
    logging.info("Running valid speaker check")
    task_prompt = f"""You are given a folder at this path: {audio_dir} containing a CSV file named 'num_speakers.csv'. 
The CSV has columns 'File Name', 'Number of Speakers', and 'Speaker Durations', where 'Speaker Durations' is a JSON string mapping speaker IDs to their speaking durations in hours.

Write a Python script to:
1. Load the 'num_speakers.csv' file.
2. Create a dictionary to count how many files each speaker appears in.
3. For each file in the CSV:
   - Parse the 'Number of Speakers' and 'Speaker Durations' columns.
   - If 'Number of Speakers' is 1 and 'SPEAKER_00' appears in more than one file, set Speaker_Status to 'Old' and Common_File to the current file name.
   - If 'Number of Speakers' > 1 and any speaker appears in more than one file, set Speaker_Status to 'Old' and Common_File to the current file name.
   - Otherwise, set Speaker_Status to 'New' and Common_File to an empty string.
4. Create a new CSV with columns: 'Filename', 'Speaker_Status', 'Common_File'.
5. Save the CSV as 'valid_speaker.csv' in the same directory.
6. Handle errors gracefully.

Respond with "Success" if the script completes and the CSV is saved, otherwise "Invalid".
"""
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(audio_dir, "valid_speaker.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"valid_speaker_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"Valid speaker check failed to generate {output_path}")
            return {"valid_speaker_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Valid speaker check failed: {e}")
        return {"valid_speaker_output": f"Error: {e}"}

def domain_checker_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for domain checker: {audio_dir}")
        return {"domain_checker_output": "Error: Invalid audio directory"}
    logging.info("Running domain checker")
    task_prompt = f"""You are given a folder at this path: {audio_dir} containing a CSV file named 'indicconf_hypothesis.csv'. 
The CSV has columns 'Filename' and 'Indiconformer_Hypothesis', where 'Indiconformer_Hypothesis' contains transcriptions of audio files.

Write a Python script to:
1. Load the 'indicconf_hypothesis.csv' file.
2. Analyze the content of the 'Indiconformer_Hypothesis' column to determine the general domain of the speech dataset.
3. Return the domain as a one- or two-word phrase (e.g., 'News', 'Call Center').
4. Create a CSV with columns 'Folder Path' and 'Domain', containing a single row with the folder path and the inferred domain.
5. Save the CSV as 'domain_check.csv' in the same directory.
6. Handle errors gracefully.

Respond with "Success" if the script completes and the CSV is saved, otherwise "Invalid".
"""
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(audio_dir, "domain_check.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"domain_checker_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"Domain checker failed to generate {output_path}")
            return {"domain_checker_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Domain checker failed: {e}")
        return {"domain_checker_output": f"Error: {e}"}

def audio_transcript_matching_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    # csv_path = state.get('ground_truth_csv')
    # audio_dir = "/raid/ganesh/pdadiga/chriss/test_data"
    csv_path = "/raid/ganesh/pdadiga/chriss/test_data/indicconf_hypothesis.csv"
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for transcript matching: {audio_dir}")
        return {"audio_transcript_matching_output": "Error: Invalid audio directory"}
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for transcript matching: {csv_path}")
        return {"audio_transcript_matching_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running audio and transcript matching")
    task_prompt = f"""You are given a folder at this path: {audio_dir} containing audio files (.wav or .mp3) and a CSV file at {csv_path}. 
The CSV has columns 'Filename' and 'Indiconformer_Hypothesis', where 'Indiconformer_Hypothesis' contains ground truth transcriptions.

Write a Python script to:
1. Load the CSV file.
2. For each row in the CSV:
   - Get the audio file path by joining the folder path with the 'Filename'.
   - Use the force_alignment_and_ctc_score function from utility_functions to perform forced alignment.
   - Create an aligned transcript by joining tokens from aligned_segments, excluding special tokens.
3. Create a CSV with columns 'Filename', 'Aligned_Segments' (JSON string), and 'Aligned_Transcript'.
4. Save the CSV as 'audio_transcript_matching.csv' in the same directory.
5. Handle errors gracefully.

Respond with "Success" if the script completes and the CSV is saved, otherwise "Invalid".
"""
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(audio_dir, "audio_transcript_matching.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"audio_transcript_matching_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"Transcript matching failed to generate {output_path}")
            return {"audio_transcript_matching_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Transcript matching failed: {e}")
        return {"audio_transcript_matching_output": f"Error: {e}"}

# def language_identification_indiclid_agent(state: CombinedStateDict) -> CombinedStateDict:
#     audio_dir = state.get('audio_dir')
#     if not audio_dir or not os.path.isdir(audio_dir):
#         logging.error(f"Invalid audio directory for IndicLID: {audio_dir}")
#         return {"language_identification_indiclid_output": "Error: Invalid audio directory"}
#     logging.info("Running language identification with IndicLID")
#     task_prompt = f"""You are given a folder at this path: {audio_dir} containing a CSV file named 'indicconf_hypothesis.csv'. 
# The CSV has columns 'Filename' and 'Indiconformer_Hypothesis', where 'Indiconformer_Hypothesis' contains ASR transcriptions.

# Write a Python script to:
# 1. Load the 'indicconf_hypothesis.csv' file.
# 2. Use the language_identification_indiclid function from utility_functions to perform language identification.
# 3. Create a CSV with columns 'Filename', 'Transcription', 'Detected_Language', 'Confidence', and 'Model_Used'.
# 4. Save the CSV as 'indiclid_language_identification.csv' in the same directory.
# 5. Handle errors gracefully.

# Respond with "Success" if the script completes and the CSV is saved, otherwise "Invalid".
# """
#     try:
#         response = agent.invoke(task_prompt)
#         output_path = os.path.join(audio_dir, "indiclid_language_identification.csv")
#         if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
#             return {"language_identification_indiclid_output": f"CSV saved at: {output_path}"}
#         else:
#             logging.error(f"IndicLID failed to generate {output_path}")
#             return {"language_identification_indiclid_output": f"Error: Failed to generate {output_path}"}
#     except Exception as e:
#         logging.error(f"IndicLID failed: {e}")
#         return {"language_identification_indiclid_output": f"Error: {e}"}


# def language_identification_indiclid_agent(state: CombinedStateDict) -> CombinedStateDict:
#     audio_dir = state.get('audio_dir')
#     if not audio_dir or not os.path.isdir(audio_dir):
#         logging.error(f"Invalid audio directory for IndicLID: {audio_dir}")
#         return {"language_identification_indiclid_output": "Error: Invalid audio directory"}
#     logging.info("Running language identification with IndicLID")
#     try:
#         input_csv = os.path.join(audio_dir, "indicconf_hypothesis.csv")
#         if not os.path.isfile(input_csv):
#             logging.error(f"Indicconf hypothesis CSV not found: {input_csv}")
#             return {"language_identification_indiclid_output": f"Error: CSV file {input_csv} not found"}
        
#         output_path = os.path.join(audio_dir, "indiclid_language_identification.csv")
#         df = pd.read_csv(input_csv)
#         if 'Filename' not in df.columns or 'Indiconformer_Hypothesis' not in df.columns:
#             logging.error(f"Invalid columns in {input_csv}. Expected 'Filename' and 'Indiconformer_Hypothesis'")
#             return {"language_identification_indiclid_output": f"Error: Invalid columns in CSV"}
        
#         results = []
#         for _, row in df.iterrows():
#             filename = row['Filename']
#             transcription = str(row['Indiconformer_Hypothesis'])
#             if pd.isna(transcription) or transcription.strip() == "":
#                 logging.warning(f"Empty transcription for {filename}")
#                 results.append({
#                     "Filename": filename,
#                     "Transcription": transcription,
#                     "Detected_Language": "Unknown",
#                     "Confidence": 0.0,
#                     "Model_Used": "IndicLID"
#                 })
#                 continue
            
#             try:
#                 lid_result = language_identification_indiclid(transcription)
#                 results.append({
#                     "Filename": filename,
#                     "Transcription": transcription,
#                     "Detected_Language": lid_result.get("detected_language", "Unknown"),
#                     "Confidence": lid_result.get("confidence", 0.0),
#                     "Model_Used": lid_result.get("model_used", "IndicLID")
#                 })
#             except Exception as e:
#                 logging.warning(f"IndicLID failed for {filename}: {e}")
#                 results.append({
#                     "Filename": filename,
#                     "Transcription": transcription,
#                     "Detected_Language": "Error",
#                     "Confidence": 0.0,
#                     "Model_Used": "IndicLID"
#                 })

#         if results:
#             output_df = pd.DataFrame(results, columns=["Filename", "Transcription", "Detected_Language", "Confidence", "Model_Used"])
#             output_df.to_csv(output_path, index=False)
#             logging.info(f"IndicLID results saved to: {output_path}")
#             return {"language_identification_indiclid_output": f"CSV saved at: {output_path}"}
#         else:
#             logging.error(f"No results generated for IndicLID")
#             return {"language_identification_indiclid_output": f"Error: No results generated"}
#     except Exception as e:
#         logging.error(f"IndicLID failed: {e}")
#         return {"language_identification_indiclid_output": f"Error: {e}"}

# def language_identification_indiclid_agent(state: CombinedStateDict) -> CombinedStateDict:
#     audio_dir = state.get('audio_dir')
#     if not audio_dir or not os.path.isdir(audio_dir):
#         logging.error(f"Invalid audio directory for IndicLID: {audio_dir}")
#         return {"language_identification_indiclid_output": "Error: Invalid audio directory"}
#     logging.info("Running language identification with IndicLID")
#     try:
#         input_csv="/raid/ganesh/pdadiga/chriss/test_data/indicconf_hypothesis.csv"
#         # input_csv = os.path.join(audio_dir, "indicconf_hypothesis.csv")
#         if not os.path.isfile(input_csv):
#             logging.error(f"Indicconf hypothesis CSV not found: {input_csv}. Ensure task 1 (ASR transcription) is run first.")
#             return {"language_identification_indiclid_output": f"Error: CSV file {input_csv} not found"}
        
#         output_path="/raid/ganesh/pdadiga/chriss/test_data/indiclid_language_identification.csv"
#         # output_path = os.path.join(audio_dir, "indiclid_language_identification.csv")
#         df = pd.read_csv(input_csv)
#         df.columns = df.columns.str.lower()
#         if 'filename' not in df.columns or 'indiconformer_hypothesis' not in df.columns:
#             if 'filename' not in df.columns and 'Filename' in df.columns:
#                 df['filename'] = df['Filename']
#             if 'indiconformer_hypothesis' not in df.columns and 'Transcription' in df.columns:
#                 df['indiconformer_hypothesis'] = df['Transcription']
#             if 'filename' not in df.columns or 'indiconformer_hypothesis' not in df.columns:
#                 logging.error(f"Invalid columns in {input_csv}. Expected 'Filename' and 'Indiconformer_Hypothesis'")
#                 return {"language_identification_indiclid_output": f"Error: Invalid columns in CSV"}
#         df = df.rename(columns={'filename': 'Filename', 'indiconformer_hypothesis': 'Indiconformer_Hypothesis'})
#         results = []
#         for _, row in df.iterrows():
#             filename = row['Filename']
#             transcription = str(row['Indiconformer_Hypothesis'])
#             if pd.isna(transcription) or transcription.strip() == "":
#                 logging.warning(f"Empty transcription for {filename}")
#                 results.append({
#                     "Filename": filename,
#                     "Transcription": transcription,
#                     "Detected_Language": "Unknown",
#                     "Confidence": 0.0,
#                     "Model_Used": "IndicLID"
#                 })
#                 continue
            
#             logging.info(f"Processing transcription for {filename}: {transcription[:50]}...")
#             try:
#                 lid_result = language_identification_indiclid(transcription)
#                 if not isinstance(lid_result, dict):
#                     raise ValueError(f"Invalid result format from language_identification_indiclid: {lid_result}")
#                 results.append({
#                     "Filename": filename,
#                     "Transcription": transcription,
#                     "Detected_Language": lid_result.get("detected_language", "Unknown"),
#                     "Confidence": float(lid_result.get("confidence", 0.0)),
#                     "Model_Used": lid_result.get("model_used", "IndicLID")
#                 })
#             except Exception as e:
#                 logging.warning(f"IndicLID failed for {filename}: {e}")
#                 results.append({
#                     "Filename": filename,
#                     "Transcription": transcription,
#                     "Detected_Language": "Error",
#                     "Confidence": 0.0,
#                     "Model_Used": "IndicLID"
#                 })
        
#         if results:
#             output_df = pd.DataFrame(results, columns=["Filename", "Transcription", "Detected_Language", "Confidence", "Model_Used"])
#             output_df.to_csv(output_path, index=False)
#             logging.info(f"IndicLID results saved to: {output_path}")
#             return {"language_identification_indiclid_output": f"CSV saved at: {output_path}"}
#         else:
#             logging.error(f"No valid results generated for IndicLID")
#             return {"language_identification_indiclid_output": f"Error: No valid results generated"}
#     except Exception as e:
#         logging.error(f"IndicLID processing failed: {e}")
#         return {"language_identification_indiclid_output": f"Error: {e}"}

# def language_identification_indiclid_agent(state: CombinedStateDict) -> CombinedStateDict:
#     import os
#     import logging
#     import pandas as pd

#     audio_dir = state.get('audio_dir')
#     if not audio_dir or not os.path.isdir(audio_dir):
#         logging.error(f"Invalid audio directory for IndicLID: {audio_dir}")
#         return {"language_identification_indiclid_output": "Error: Invalid audio directory"}

#     logging.info("Running language identification with IndicLID")

#     try:
#         input_csv = "/raid/ganesh/pdadiga/chriss/test_data/indicconf_hypothesis.csv"
#         output_path = "/raid/ganesh/pdadiga/chriss/test_data/indiclid_language_identification.csv"

#         if not os.path.isfile(input_csv):
#             logging.error(f"CSV file not found: {input_csv}")
#             return {"language_identification_indiclid_output": f"Error: CSV file {input_csv} not found"}

#         df = pd.read_csv(input_csv)
#         df.columns = df.columns.str.lower()

#         # Handle different column casing
#         if 'filename' not in df.columns or 'indiconformer_hypothesis' not in df.columns:
#             if 'filename' not in df.columns and 'Filename' in df.columns:
#                 df['filename'] = df['Filename']
#             if 'indiconformer_hypothesis' not in df.columns and 'Transcription' in df.columns:
#                 df['indiconformer_hypothesis'] = df['Transcription']
#             if 'filename' not in df.columns or 'indiconformer_hypothesis' not in df.columns:
#                 logging.error("Expected columns 'Filename' and 'Indiconformer_Hypothesis' not found.")
#                 return {"language_identification_indiclid_output": "Error: Invalid columns in CSV"}

#         df = df.rename(columns={'filename': 'Filename', 'indiconformer_hypothesis': 'Indiconformer_Hypothesis'})

#         results = []

#         for _, row in df.iterrows():
#             filename = row['Filename']
#             transcription = str(row['Indiconformer_Hypothesis'])

#             if pd.isna(transcription) or transcription.strip() == "":
#                 logging.warning(f"Empty transcription for {filename}")
#                 results.append((filename, transcription, "Unknown", 0.0, "IndicLID"))
#                 continue

#             try:
#                 # Expected output: list of tuples (text, lang_code, confidence, model)
#                 lid_outputs = language_identification_indiclid(transcription)

#                 if not isinstance(lid_outputs, list):
#                     raise ValueError(f"Invalid result format from language_identification_indiclid: {lid_outputs}")

#                 for text, lang_code, confidence, model in lid_outputs:
#                     results.append((filename, text, lang_code, confidence, model))

#             except Exception as e:
#                 logging.warning(f"IndicLID failed for {filename}: {e}")
#                 results.append((filename, transcription, "Error", 0.0, "IndicLID"))

#         if results:
#             output_df = pd.DataFrame(results, columns=["Filename", "Transcription", "Detected_Language", "Confidence", "Model_Used"])
#             output_df.to_csv(output_path, index=False)
#             logging.info(f"IndicLID results saved to: {output_path}")
#             return {"language_identification_indiclid_output": f"CSV saved at: {output_path}"}
#         else:
#             logging.error("No valid results generated for IndicLID")
#             return {"language_identification_indiclid_output": "Error: No valid results generated"}

#     except Exception as e:
#         logging.error(f"IndicLID processing failed: {e}")
#         return {"language_identification_indiclid_output": f"Error: {e}"}
def language_identification_indiclid_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for IndicLID: {audio_dir}")
        return {"language_identification_indiclid_output": "Error: Invalid audio directory"}
    logging.info("Running language identification with IndicLID")
    
    try:
        # Use task 1 output if available, else fall back to default path
        input_csv = state.get('A', os.path.join(audio_dir, "indicconf_hypothesis.csv"))
        output_path = os.path.join(audio_dir, "indiclid_language_identification.csv")
        
        if not os.path.isfile(input_csv):
            logging.error(f"Indicconf hypothesis CSV not found: {input_csv}. Ensure task 1 (ASR transcription) ran successfully.")
            return {"language_identification_indiclid_output": f"Error: CSV file {input_csv} not found"}
        
        df = pd.read_csv(input_csv)
        df.columns = df.columns.str.lower()
        # Handle column name variations
        if 'filename' not in df.columns or 'indiconformer_hypothesis' not in df.columns:
            if 'filename' not in df.columns and 'Filename' in df.columns:
                df['filename'] = df['Filename']
            if 'indiconformer_hypothesis' not in df.columns and 'Transcription' in df.columns:
                df['indiconformer_hypothesis'] = df['Transcription']
            if 'filename' not in df.columns or 'indiconformer_hypothesis' not in df.columns:
                logging.error(f"Invalid columns in {input_csv}. Expected 'Filename' and 'Indiconformer_Hypothesis'")
                return {"language_identification_indiclid_output": "Error: Invalid columns in CSV"}
        df = df.rename(columns={'filename': 'Filename', 'indiconformer_hypothesis': 'Indiconformer_Hypothesis'})
        
        results = []
        for _, row in df.iterrows():
            filename = row['Filename']
            transcription = str(row['Indiconformer_Hypothesis'])
            if pd.isna(transcription) or transcription.strip() == "":
                logging.warning(f"Empty transcription for {filename}")
                results.append((filename, transcription, "Unknown", 0.0, "IndicLID"))
                continue
            
            logging.info(f"Processing transcription for {filename}: {transcription[:50]}...")
            try:
                # Expected output: list of tuples (text, lang_code, confidence, model)
                lid_outputs = language_identification_indiclid(transcription)
                if not isinstance(lid_outputs, list):
                    raise ValueError(f"Invalid result format from language_identification_indiclid: {lid_outputs}")
                for text, lang_code, confidence, model in lid_outputs:
                    results.append((filename, text, lang_code, float(confidence), model))
            except Exception as e:
                logging.warning(f"IndicLID failed for {filename}: {e}")
                results.append((filename, transcription, "Error", 0.0, "IndicLID"))
        
        if results:
            output_df = pd.DataFrame(results, columns=["Filename", "Transcription", "Detected_Language", "Confidence", "Model_Used"])
            output_df.to_csv(output_path, index=False)
            logging.info(f"IndicLID results saved to: {output_path}")
            return {"language_identification_indiclid_output": f"CSV saved at: {output_path}"}
        else:
            logging.error("No valid results generated for IndicLID")
            return {"language_identification_indiclid_output": "Error: No valid results generated"}
    
    except Exception as e:
        logging.error(f"IndicLID processing failed: {e}")
        return {"language_identification_indiclid_output": f"Error: {e}"}


def normalization_remove_tags_agent(state: CombinedStateDict) -> CombinedStateDict:
    csv_path = state.get('ground_truth_csv')
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for character calculation: {csv_path}")
        return {"normalization_remove_tags_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running normalizing_agent")
    task_prompt = f"""You are given a CSV file at this path: {csv_path}.
It has a column called 'Transcription' or 'Ground_Truth',(search case insensitively), take it as transcription coloumn.

Write a Python script to:
1. For each row, remove HTML tags from those columns from the transcription coloumn and store it in a new column called 'normalized_transcripts'.
2. Save the updated CSV with the new column to the same directory as normalized_list.csv

Respond with "Success" if the script completes
"""
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(os.path.dirname(csv_path), "normalized_list.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"normalization_remove_tags_output": f"CSV saved at: {output_path}"}
        else:
            error_msg = f"Failed to generate {output_path}"
            logging.error(error_msg)
            return {"normalization_remove_tags_output": error_msg}
    except Exception as e:
        logging.error(f"Character calculation failed: {e}")
        return {"normalization_remove_tags_output": f"Error: {e}"}



def llm_score_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for LLM score: {audio_dir}")
        return {"llm_score_output": "Error: Invalid audio directory"}
    logging.info("Running LLM score evaluation")
    task_prompt = f"""You are given a folder at this path: {audio_dir} containing a CSV file named 'indicconf_hypothesis.csv'. 
The CSV has columns 'Filename' and 'Indiconformer_Hypothesis', where 'Indiconformer_Hypothesis' contains ASR transcriptions.

Write a Python script to:
1. Load the 'indicconf_hypothesis.csv' file.
2. For each transcription:
   - Evaluate its coherence and fluency using the LLM as a judge.
   - Assign a score from 0 to 10.
   - Provide a brief comment explaining the score.
3. Create a CSV with columns 'Filename', 'Transcription', 'LLM_Score', and 'Evaluation_Comment'.
4. Save the CSV as 'llm_scores.csv' in the same directory.
5. Handle errors gracefully.

Respond with "Success" if the script completes and the CSV is saved, otherwise "Invalid".
"""
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(audio_dir, "llm_scores.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"llm_score_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"LLM score evaluation failed to generate {output_path}")
            return {"llm_score_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"LLM score evaluation failed: {e}")
        return {"llm_score_output": f"Error: {e}"}

def transliteration_agent(state: CombinedStateDict) -> CombinedStateDict:
    csv_path = state.get('ground_truth_csv')
    lang_code = state.get('lang_code')
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for transliteration: {csv_path}")
        return {"transliteration_output": f"Error: CSV file {csv_path} not found"}
    if not lang_code:
        logging.error("No language code provided for transliteration")
        return {"transliteration_output": "Error: No language code provided"}
    logging.info("Running transliteration")
    try:
        result = transliterate_file(csv_path, lang_code)
        return {"transliteration_output": result}
    except Exception as e:
        logging.error(f"Transliteration failed: {e}")
        return {"transliteration_output": f"Error: {e}"}

# Node mapping for tasks
node_map = {
    1: ("node_transcription", transcription_func, "A"),
    2: ("node_num_speaker", num_speaker_func, "E"),
    3: ("node_transcript_quality", transcript_quality_agent, "transcript_quality_output"),
    4: ("node_character", character_agent, "character_output"),
    5: ("node_vocab", vocab_agent, "vocab_output"),
    6: ("node_language_verification", language_verification_agent, "language_verification_output"),
    7: ("node_audio_length", audio_length_agent, "audio_length_output"),
    8: ("node_silence", silence_vad_func, "D"),
    9: ("node_sample_rate", sample_rate_agent, "sample_rate_output"),
    10: ("node_ctc_score", ctc_score_agent, "ctc_score_output"),
    11: ("node_upsampling", upsampling_agent, "upsampling_output"),
    12: ("node_valid_speaker", valid_speaker_agent, "valid_speaker_output"),
    13: ("node_domain_checker", domain_checker_agent, "domain_checker_output"),
    14: ("node_audio_transcript_matching", audio_transcript_matching_agent, "audio_transcript_matching_output"),
    15: ("node_language_identification_indiclid", language_identification_indiclid_agent, "language_identification_indiclid_output"),
    16: ("node_normalization_remove_tags", normalization_remove_tags_agent, "normalization_remove_tags_output"),
    17: ("node_llm_score", llm_score_agent, "llm_score_output"),
    18: ("node_transliteration", transliteration_agent, "transliteration_output"),
    19: ("node_corruption", corruption_agent, "corruption_output"),
    20: ("node_extension", extension_agent, "extension_output"),
    21: ("node_sample_rate", sample_rate_agent, "sample_rate_output"),
}


def build_graph_from_structure(structure: list[list[int]], valid_tasks: set) -> StateGraph:
    graph_builder = StateGraph(CombinedStateDict)
    
    # Add valid task nodes
    added_nodes = set()
    valid_structure = [[task for task in group if str(task) in valid_tasks] for group in structure]
    valid_structure = [group for group in valid_structure if group]  # Remove empty groups
    
    # Fallback: if structure is empty but valid_tasks exist, use tasks from valid_tasks
    if not valid_structure and valid_tasks:
        valid_structure = [[int(task) for task in valid_tasks if task.isdigit()]]
    
    for group in valid_structure:
        for task_id in group:
            if task_id in node_map and task_id not in added_nodes:
                node_name, func, _ = node_map[task_id]
                graph_builder.add_node(node_name, func)
                added_nodes.add(task_id)
    
    # Add start node
    graph_builder.add_node("start", lambda state: state)
    
    # Add edges based on topological order
    for i in range(len(valid_structure)):
        current_group = valid_structure[i]
        if i == 0:
            for task_id in current_group:
                node_name, _, _ = node_map[task_id]
                graph_builder.add_edge("start", node_name)
        if i < len(valid_structure) - 1:
            next_group = valid_structure[i + 1]
            for curr_task in current_group:
                curr_node_name, _, _ = node_map[curr_task]
                for next_task in next_group:
                    next_node_name, _, _ = node_map[next_task]
                    graph_builder.add_edge(curr_node_name, next_node_name)
        if i == len(valid_structure) - 1:
            for task_id in current_group:
                node_name, _, _ = node_map[task_id]
                graph_builder.add_edge(node_name, END)
    
    # Set entry point if valid tasks exist
    if valid_structure:
        graph_builder.set_entry_point("start")
    else:
        raise ValueError("No valid tasks provided in structure")
    
    return graph_builder.compile()

# Main execution
def main(user_prompt: str):
    # Parse prompt
    parsed_inputs = parse_prompt(user_prompt)
    if not parsed_inputs["audio_dir"] and not parsed_inputs["ground_truth_csv"]:
        logging.error("No valid audio directory or ground truth CSV provided in prompt")
        print("Error: Please provide audio directory and/or ground truth CSV in the prompt")
        return

    # Select tasks and build main graph
    resp_1 = select_tasks(user_prompt)
    valid_tasks = set(resp_1.split(',')) if resp_1 else set()
    structure = topological_sort_tasks(resp_1)
    print("Using structure:", structure)

    main_graph = build_graph_from_structure(structure, valid_tasks)
    main_graph.get_graph().print_ascii()

    # Run main pipeline
    initial_state = {
        "audio_dir": parsed_inputs["audio_dir"],
        "ground_truth_csv": parsed_inputs["ground_truth_csv"],
        "lang_code": parsed_inputs["lang_code"],
        "user_prompt": user_prompt
    }
    final_result = main_graph.invoke(initial_state)

    # Print results
    print("\nFinal Pipeline Results:")
    for key, value in final_result.items():
        if key not in ["audio_dir", "ground_truth_csv", "lang_code", "user_prompt"]:
            print(f"{key}: {value[:100]}..." if isinstance(value, str) and len(value) > 100 else f"{key}: {value}")

if __name__ == "__main__":
    user_prompt= "Identify Language using INDICLID in the file :'/raid/ganesh/pdadiga/chriss/test_data/' "
    # user_prompt="i need you to check if speakers are new or old in folder : '/raid/ganesh/pdadiga/chriss/test_data/' "
    # user_prompt = "I need you to find silence in the audios  in '/raid/ganesh/pdadiga/chriss/test_data/' "
    # user_prompt="Transliterate the contents in file = '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/krishivaani_known.csv' with lang code ' Hi '"
    # user_prompt="verify if the file matches the expected language of 'Hi' in file ='/raid/ganesh/pdadiga/chriss/test_data/new_transcriptions.csv/'"
    main(user_prompt)
