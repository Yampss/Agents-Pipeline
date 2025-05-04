import torch
import librosa
import torchaudio.functional as F
import torch
import nemo.collections.asr as nemo_asr

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

hi_indicconf_model_path = '/root/abhinav/models/indicconformer_stt_hi_hybrid_rnnt_large.nemo'
hi_indicconf_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=hi_indicconf_model_path)
hi_indicconf_model.freeze()  # inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hi_indicconf_model = hi_indicconf_model.to(device)  # transfer model to device
hi_indicconf_model.cur_decoder = "ctc"

def transcribe_audio(audio_path, source_lang):
    try:
        audio_array, sr = librosa.load(audio_path, sr=16_000)

        if source_lang == "Marathi":
            print(f"Transcribing {audio_path}")
            transcription = mr_indicconf_model.transcribe([audio_path], batch_size=1, logprobs=False, language_id="mr")[0]

        elif source_lang == "Hindi":
            print(f"Transcribing {audio_path}")
            transcription = hi_indicconf_model.transcribe([audio_path], batch_size=1, logprobs=False, language_id="hi")[0]

        print(transcription)
        TRANSCRIPT = transcription[0].strip().split()
        print("Transcript: ", TRANSCRIPT)
        return " ".join(TRANSCRIPT)

    except Exception as e:
        print("Facing error in transcribing:",e)
        return f"Error: {e}"

import os
import pandas as pd

def transcribe_folder_to_csv(folder_path: str, source_language: str):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            audio_path = os.path.join(folder_path, filename)
            hypothesis = transcribe_audio(audio_path, source_language)
            results.append({
                "Filename": filename,
                "Indiconformer_Hypothesis": hypothesis
            })

    df = pd.DataFrame(results)
    output_path = os.path.join(folder_path, "indicconf_hypothesis.csv")
    df.to_csv(output_path, index=False)
    print(f"CSV saved to {output_path}")