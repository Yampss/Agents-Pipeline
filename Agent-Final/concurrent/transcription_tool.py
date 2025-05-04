import os
import torch
import librosa
import nemo.collections.asr as nemo_asr
import csv
from langgraph.graph import StateGraph, END
from typing import TypedDict

hi_indicconf_model_path = '/root/abhinav/models/indicconformer_stt_hi_hybrid_rnnt_large.nemo'
hi_indicconf_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=hi_indicconf_model_path)
hi_indicconf_model.freeze()  # inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hi_indicconf_model = hi_indicconf_model.to(device)  # transfer model to device
hi_indicconf_model.cur_decoder = "ctc"

class TranscriptionTaskState(TypedDict):
    folder_path: str
    source_lang: str
    transcription_output: str

def transcribe_audio(audio_path, source_lang):
    try:
        audio_array, sr = librosa.load(audio_path, sr=16_000)

        if source_lang == "Hindi":
            transcription = hi_indicconf_model.transcribe([audio_path], batch_size=1, logprobs=False, language_id="hi")[0]

        print(transcription)
        TRANSCRIPT = transcription[0].strip().split()
        return " ".join(TRANSCRIPT)

    except Exception as e:
        print("Error in transcribing:", e)
        return f"Error: {e}"

def process_audio_folder(folder_path: str, source_lang: str) -> str:
    try:
        csv_path = os.path.join(folder_path, "indicconformer_transcripts.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "Transcription"])

            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith(".wav") or filename.endswith(".mp3"):  # Check for audio files
                    audio_path = os.path.join(folder_path, filename)
                    transcription = transcribe_audio(audio_path, source_lang)
                    writer.writerow([filename, transcription])

        return f"Transcription completed. CSV saved at {csv_path}"

    except Exception as e:
        return f"Error processing the folder: {e}"

def transcription_node(state: TranscriptionTaskState) -> TranscriptionTaskState:
    folder_path = state["folder_path"]
    source_lang = state["source_lang"]
    output = process_audio_folder(folder_path, source_lang)
    return {"transcription_output": output}

builder = StateGraph(TranscriptionTaskState)
builder.add_node("transcription_node", transcription_node)
builder.set_entry_point("transcription_node")
builder.add_edge("transcription_node", END)

graph = builder.compile()

folder_path = "/root/abhinav/shrutilipi/newsonair_v5/hindi/Regional-Bhopal-Hindi-0705-201932960"
source_lang = "Hindi"
initial_state = {"folder_path": folder_path, "source_lang": source_lang}

final_result = graph.invoke(initial_state)
print("Transcription Output:", final_result["transcription_output"])
print()