# Audio Processing and Transcription Analysis Pipeline


## Overview
####### This repository contains llm.py, a sophisticated Python-based pipeline designed for processing audio datasets and performing transcription analysis tasks. The pipeline leverages large language models (LLMs), agentic workflows, and graph-based task orchestration to execute a variety of audio and text processing tasks, such as transcription, vocabulary extraction, character analysis, language verification, and quality control (QC) checks. It is built to handle diverse inputs, including audio directories and ground truth CSV files, and supports concurrent task execution for efficiency.The pipeline is modular, extensible, and integrates with external libraries like pandas, langchain, and jiwer for data manipulation, LLM interactions, and word error rate (WER) computation. It uses a directed acyclic graph (DAG) to manage task dependencies and execution order, ensuring tasks like WER computation depend on transcription outputs. The system is designed for robustness, with detailed logging and error handling to facilitate debugging and monitoring.
Key features include:
Dynamic Task Selection: Automatically selects relevant tasks based on user prompts using LLM-driven prompt analysis.
Concurrent Task Execution: Utilizes ThreadPoolExecutor for parallel processing of independent tasks.
Flexible Input Handling: Supports audio directories and CSV files with dynamic column detection (e.g., Transcription, ground_truth, Ground_Truth).
Comprehensive QC Checks: Includes audio corruption, extension, and sample rate checks, with configurable execution based on input availability.
Extensive Logging: Logs all operations and errors to pipeline.log for traceability.

## Technical Architecture

### Core Components:

LLM Integration:
Primary LLM: For task selection, script generation, and prompt verification.
Secondary LLM: For prompt checker iterations.
Agent Framework: Uses langchain to initialize a zero-shot-react-description agent with a Python REPL tool for dynamic script execution.


### Task Orchestration:
State Management: Uses a CombinedStateDict (TypedDict) to maintain pipeline state, including inputs (audio_dir, ground_truth_csv, lang_code) and task outputs (e.g., character_output, vocab_output).
Graph Construction: Employs langgraph to build a DAG based on a topological sort of tasks, ensuring dependencies are respected (e.g., task 1 must complete before tasks 3, 4, 5, 6, 10, 13, 14, 15, 16, 17).
Concurrent Execution: Groups independent tasks into concurrent execution groups using ThreadPoolExecutor.


### Input Parsing:
Prompt Parser: Extracts audio_dir, ground_truth_csv, and lang_code from user prompts using regex patterns and fallback LLM-based parsing.
Validation: Checks the existence and validity of input paths, logging errors for invalid directories or files.


### Task Definitions:

The pipeline supports 24 tasks, each mapped to a specific function in node_map. Examples include:
Task 1: ASR transcription using transcribe_folder_to_csv.
Task 4: Character calculation, extracting unique characters from transcriptions.
Task 5: Vocabulary calculation, extracting unique words.
Task 24: WER computation using jiwer.
Tasks are dynamically selected based on the prompt and verified by a prompt checker agent.


### Output Handling:
Outputs are saved as CSV files in the input directory (e.g., character_list.csv, vocab_list.csv, wer.csv).
Results are stored in the state dictionary and printed at the end of execution.

### Key Functions

#### parse_prompt:
Extracts input paths and language codes using regex and LLM fallback.
Validates inputs, ensuring directories and files exist.
Returns a dictionary with audio_dir, ground_truth_csv, lang_code, and user_prompt.


#### select_tasks:
Uses LLM to identify relevant tasks based on the prompt.
Applies rules to include tasks like 4 (character calculation) and 5 (vocab calculation) for prompts mentioning "character" or "vocab".
Excludes QC tasks (19, 20, 21) unless explicitly requested or audio_dir is provided.


#### prompt_checker_agent:

Iteratively verifies task selection using a secondary LLM.
Ensures compliance with predefined rules, adding missing tasks (e.g., task 16 before tasks 9, 23, 24).
Returns a comma-separated string of verified task numbers.


#### topological_sort_tasks:

Generates a topological sort of tasks as a list of lists, grouping independent tasks for concurrent execution.
Respects strict dependencies, such as task 1 (transcription) preceding tasks 3, 4, 5, 6, 10, 13, 14, 15, 16, 17.
Falls back to a single group if parsing fails.


#### build_graph_from_structure:

Constructs a langgraph DAG from the topological structure.
Maps tasks to nodes using node_map, adding edges based on group dependencies.
Supports concurrent execution within groups.


#### Agent Functions:

character_agent: Extracts unique characters from transcriptions, saving results to character_list.csv.
vocab_agent: Extracts unique words, saving results to vocab_list.csv.
wer_computation_agent: Computes WER between normalized transcripts and hypotheses using jiwer.
Other agents handle tasks like language verification, CTC scoring, and domain checking.



### Dependencies and Libraries

Python: 3.8+
Core Libraries:
pandas: Data manipulation and CSV handling.
langchain: LLM integration and agent framework.
langchain-groq: Groq LLM interface.
langgraph: Graph-based task orchestration.
jiwer: Word error rate computation.
os, re, json, logging: File handling, regex, and logging.


### External Utilities:
utility_functions2: Custom functions for transcription, VAD, speaker diarization, and more (assumed to be in the same directory).



### Setup Instructions
Prerequisites

Python Version: 3.8 or higher.
Hardware: Multi-core CPU for concurrent task execution; sufficient storage for audio and CSV files.

Install Dependencies:Create a virtual environment and install required packages:
python -m venv venv
source venv/bin/activate
Set Up API Keys.
Set the environment variables.


### Ensure Utility Functions:
Verify that utility_functions.py is present in the same directory and contains functions like transcribe_folder_to_csv, process_folder_vad, etc.
If missing, implement or obtain these functions from the original source.

### Prepare Input Data:
Audio Directory: Place audio files (e.g., .wav) in a directory.
Ground Truth CSV: Create a CSV and place it in the audio_dir with columns like Filename and ground_truth.

### Enter a Prompt and watch as the magic happens:
Example prompt:verify language as Hindi ('hi'),CTC SCORE, WER computation, normalization, calculate LLM score, check domain, english word counter, and chcek for utterance duplicate in :'/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC2/audios/'
The prompt specifies tasks and the audio directory. For tasks requiring a ground truth CSV, include a path like:I want character vocab and give it a path ='/raid/ganesh/pdadiga/chriss/test_data/new_transcriptions.csv'


Expected Output:

The pipeline prints the task structure, an ASCII DAG, and final results:Using structure: [[1], [6, 10, 13, 16, 17], [9, 23, 24]]
+-----------+
|   start   |
+-----------+
      |
      v
+--------------------+
| node_transcription |
+--------------------+
      |
      v
+------------------+
|   group_1        |
+------------------+
      |
      v
+------------------+
|   group_2        |
+------------------+
      |
      v
+-----------+
|    END    |
+-----------+




Task Dependencies
The pipeline enforces strict dependencies and are necessary for program flow order and efficiency:

Task 1 (Transcription): Required before tasks 3, 4, 5, 6, 10, 13, 14, 15, 16, 17.
Task 2 (Speaker Calculation): Required before tasks 12 and 22.
Task 16 (Normalization): Required before tasks 9, 23, 24.
Independent tasks (e.g., 4, 5, 7, 8, 11, 19, 20, 21) can run concurrently if no dependencies exist.


Performance Considerations
Concurrency: Independent tasks within a topological group run concurrently using ThreadPoolExecutor, reducing execution time on multi-core systems.
LLM Efficiency: Uses temperature=0.1 and max_tokens=4096 to balance response quality and speed.
File I/O: CSV operations are optimized using pandas, but large datasets may require sufficient disk space and memory.


Please ensure code adheres to PEP 8 standards and includes appropriate logging and error handling.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or support, contact the repository maintainer at [chrissattasseril16@gmail.com] or open an issue on the repository.

