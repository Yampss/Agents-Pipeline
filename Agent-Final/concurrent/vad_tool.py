import os
import csv
import librosa
from typing import TypedDict
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
from langgraph.graph import StateGraph, END


token = "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"  # replace with your actual token
model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
pipeline = VoiceActivityDetection(segmentation=model)

HYPER_PARAMETERS = {
    "min_duration_on": 0.0,
    "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)


def get_total_silence_time(audio_file_path):
    vad_result = pipeline(audio_file_path)
    audio, sr = librosa.load(audio_file_path, sr=16000)

    total_audio_duration = len(audio) / sr
    total_silence = 0.0
    last_end = 0.0

    for segment in vad_result.itersegments():
        start = segment.start
        end = segment.end
        if start > last_end:
            total_silence += start - last_end
        last_end = end

    if last_end < total_audio_duration:
        total_silence += total_audio_duration - last_end

    return total_silence


def perform_vad_and_save_csv(folder_path: str) -> str:
    try:
        csv_path = os.path.join(folder_path, "silence_results.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "Total Silence Time (s)"])

            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    audio_path = os.path.join(folder_path, filename)
                    total_silence = get_total_silence_time(audio_path)
                    writer.writerow([filename, total_silence])

        return f"Silence results saved at {csv_path}"
    except Exception as e:
        return f"Error: {e}"


class SilenceTaskState(TypedDict):
    folder_path: str
    vad_output: str


def vad_node(state: SilenceTaskState) -> SilenceTaskState:
    folder_path = state["folder_path"]
    output = perform_vad_and_save_csv(folder_path)
    return {"vad_output": output}

builder = StateGraph(SilenceTaskState)
builder.add_node("vad_node", vad_node)
builder.set_entry_point("vad_node")
builder.add_edge("vad_node", END)

graph = builder.compile()

folder_path = "/root/abhinav/shrutilipi/newsonair_v5/hindi/Regional-Bhopal-Hindi-0705-201932960"
initial_state = {"folder_path": folder_path}
final_result = graph.invoke(initial_state)

print("VAD Output:", final_result["vad_output"])
