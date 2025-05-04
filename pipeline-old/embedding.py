from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist
import torch
import os
import pandas as pd
import pickle 

model = Model.from_pretrained("pyannote/embedding", use_auth_token="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")
inference = Inference(model, window="whole")


def save_embeddings(audio_directory, output_path):
    file_paths = [os.path.join(audio_directory, f) for f in os.listdir(audio_directory) if f.endswith('.wav')]
    File_Names = []
    D_Results = []
    for filename in file_paths:
        try:
            embedding = inference(filename)
            File_Names.append(os.path.basename(filename))
            D_Results.append(embedding)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            D_Results.append(f"ERROR : {e}")


    data = {
        "File": File_Names, 
        "Embedding": D_Results
    }

    fpath = output_path
    with open(fpath, "wb") as file:
        pickle.dump(data, file)
        
    # print(f"Data saved successfully to {fpath}.")
