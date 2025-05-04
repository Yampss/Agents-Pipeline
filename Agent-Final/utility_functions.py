
import torch
import librosa
import torchaudio.functional as F
import torch
import nemo.collections.asr as nemo_asr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from dataclasses import dataclass

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
    return f"CSV saved at: {output_path}"

import os
import json
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import numpy as np

token = "hf_CUqhMlqnNPXrNRDzJcDFjkKslhJPqDkDDDA"
model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
    "min_duration_on": 0.0,  # Ignore speech shorter than 0.0 seconds
    "min_duration_off": 0.0  # Ignore silence shorter than 0.0 seconds
}
pipeline.instantiate(HYPER_PARAMETERS)

silent_list = []

def perform_vad(audio_file):
    vad_result = pipeline(audio_file)
    return vad_result

def get_total_silence_time(audio_file_path):
    vad_result = perform_vad(audio_file_path)
    audio, sr = librosa.load(audio_file_path, sr=16000)

    total_audio_duration = len(audio) / sr
    total_silence = 0.0
    last_end = 0.0

    for segment in vad_result.itersegments():
        start = segment.start
        end = segment.end
        if start > last_end:
            silence_duration = start - last_end
            total_silence += silence_duration
        last_end = end

    if last_end < total_audio_duration:
        total_silence += total_audio_duration - last_end

    return total_silence

import pandas as pd

def process_folder_vad(audio_folder: str):
    results = []

    for filename in os.listdir(audio_folder):
        if filename.lower().endswith((".wav", ".flac", ".mp3")):
            audio_path = os.path.join(audio_folder, filename)
            try:
                silence = get_total_silence_time(audio_path)
                results.append({
                    "Filename": filename,
                    "Total Silence (s)": round(silence, 2)
                })
                print(f"Processed: {filename}, Silence: {round(silence, 2)}s")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(results)
    output_path = os.path.join(audio_folder, "vad_silence_stats.csv")
    df.to_csv(output_path, index=False)
    return f"CSV saved at: {output_path}"

import os
import torch
import csv
from pyannote.audio import Pipeline

def save_num_speakers(folder_path: str, model_token="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") -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=model_token)
        pipeline.to(device)
    except Exception as e:
        return f"Failed to load diarization pipeline: {e}"

    output_csv = os.path.join(folder_path, "num_speakers.csv")

    try:
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "Number of Speakers"])

            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith((".wav", ".mp3")):
                    audio_path = os.path.join(folder_path, filename)

                    try:
                        diarization = pipeline(audio_path)
                        unique_speakers = {
                            speaker for _, _, speaker in diarization.itertracks(yield_label=True)
                        }

                        writer.writerow([filename, len(unique_speakers)])
                        print(f"Processed {filename}: {len(unique_speakers)} unique speakers")

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        return f"CSV saved at: {output_csv}"
    except Exception as e:
        return f"Error writing to CSV: {e}"

def transcript_quality(transcript):
    words = transcript.strip().split()
    repeated = len(set(words)) < len(words) * 0.7
    chars = set(transcript)
    if len(words) > 3 and not repeated:
        return "passed"
    else:
        return "failed"

def force_alignment_and_ctc_score(audio_path, given_transcript):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"  # Hindi ASR model
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float

    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

    def get_trellis(emission, token_ids, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(token_ids)

        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, token_ids[1:]],
            )
        return trellis

    def backtrack(trellis, emission, token_ids, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1
        path = [Point(j, t, emission[t, blank_id].exp().item())]
        
        while j > 0 and t > 0:
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, token_ids[j]]
            
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change
            
            t -= 1
            if changed > stayed:
                j -= 1

            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))

        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1
        
        return path[::-1]

    def merge_repeats(path):
        i1, i2 = 0, 0
        segments = []
        
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            
            avg_score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(Segment(
                processor.tokenizer.convert_ids_to_tokens(path[i1].token_index),
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                avg_score
            ))
            
            i1 = i2
        return segments


    wav, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(wav, return_tensors="pt", sampling_rate=16000).input_values.to(device)


    with torch.no_grad():
        logits = model(input_values).logits
        emissions = torch.log_softmax(logits, dim=-1).cpu()

    token_ids = processor.tokenizer.encode(given_transcript, add_special_tokens=False)

    trellis = get_trellis(emissions[0], token_ids)
    path = backtrack(trellis, emissions[0], token_ids)

    segments = merge_repeats(path)
    
    filtered_segments = [seg for seg in segments if seg.label not in ["|","<s>","<pad>","<unk>","</s>"]]
    total_score = sum(seg.score for seg in filtered_segments)
    average_ctc_score = total_score / len(filtered_segments) if filtered_segments else 0  # Avoid division by zero
    
    return average_ctc_score
