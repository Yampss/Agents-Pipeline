import os
import torch
import csv
from typing import TypedDict
from pyannote.audio import Pipeline
from langgraph.graph import StateGraph, END

class DiarizationTaskState(TypedDict):
    folder_path: str
    diarization_output: str

def diarize_and_output_csv(folder_path: str, model_token="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") -> str:
    # Set the device (e.g., GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pre-trained speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=model_token)
    pipeline.to(torch.device(device))

    output_csv = os.path.join(folder_path, "num_speakers.csv")
    try:
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "Number of Speakers"])

            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith('.wav') or filename.endswith('.mp3'):
                    audio_path = os.path.join(folder_path, filename)

                    try:
                        diarization = pipeline(audio_path)
                        unique_speakers = set()

                        # Iterate through the diarization results to count unique speakers
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            unique_speakers.add(speaker)

                        # Write the results to the CSV
                        writer.writerow([filename, len(unique_speakers)])
                        print(f"Processed {filename}: {len(unique_speakers)} unique speakers")

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        return f"Diarization results saved in {output_csv}"
    except Exception as e:
        return f"Error saving CSV: {e}"

# LangGraph Node for Diarization
def diarization_node(state: DiarizationTaskState) -> DiarizationTaskState:
    folder_path = state["folder_path"]
    output = diarize_and_output_csv(folder_path)
    return {"diarization_output": output}


builder = StateGraph(DiarizationTaskState)
builder.add_node("diarization_node", diarization_node)
builder.set_entry_point("diarization_node")
builder.add_edge("diarization_node", END)

graph = builder.compile()

folder_path = "/root/abhinav/shrutilipi/newsonair_v5/hindi/Regional-Bhopal-Hindi-0705-201932960"
initial_state = {"folder_path": folder_path}
final_result = graph.invoke(initial_state)

print("Diarization Output:", final_result["diarization_output"])
