# [ACL'25] SPEECH-QC AGENGTS


## Overview
###### This repository contains llm.py, a sophisticated Python-based pipeline designed for processing audio datasets and performing transcription analysis tasks. The pipeline leverages large language models (LLMs), agentic workflows, and graph-based task orchestration to execute a variety of audio and text processing tasks, such as transcription, vocabulary extraction, character analysis, language verification, and quality control (QC) checks. It is built to handle diverse inputs, including audio directories and ground truth CSV files, and supports concurrent task execution for efficiency.The pipeline is modular, extensible, and integrates with external libraries like pandas, langchain, and jiwer for data manipulation, LLM interactions, and word error rate (WER) computation. It uses a directed acyclic graph (DAG) to manage task dependencies and execution order, ensuring tasks like WER computation depend on transcription outputs. The system is designed for robustness, with detailed logging and error handling to facilitate debugging and monitoring.
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


### External Utilities:
utility_functions2: Custom functions for transcription, VAD, speaker diarization, and more (assumed to be in the same directory).


## 🚀 Quick Start 
--------------------------------------------------------------------------------------
### 📦 Install packages

```bash
conda create -n envqc python=3.10
conda activate envqc
pip install -r requirements.txt
```

### 🔑 Add API keys in the top section of the llm.py fie.










