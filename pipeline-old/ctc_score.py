import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"  # Change to your ASR model
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

def force_alignment_and_ctc_score(audio_path, given_transcript):
    # Load the audio file
    wav, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(wav, return_tensors="pt", sampling_rate=16000).input_values.to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
        emissions = torch.log_softmax(logits, dim=-1).cpu()

    # Tokenize the given transcript
    token_ids = processor.tokenizer.encode(given_transcript, add_special_tokens=False)

    # Compute the trellis and backtrack for forced alignment
    trellis = get_trellis(emissions[0], token_ids)
    path = backtrack(trellis, emissions[0], token_ids)

    # Merge repeated tokens and create aligned segments
    segments = merge_repeats(path)

    # Calculate the CTC score and segment times
    waveform = torch.tensor(wav).unsqueeze(0)
    sample_rate = sr
    ratio = waveform.size(1) / trellis.size(0)

    # Print aligned segments with their CTC scores
    aligned_transcript = ""
    for i, seg in enumerate(segments):
        x0 = int(ratio * seg.start)
        x1 = int(ratio * seg.end)
        start_time = x0 / sample_rate
        end_time = x1 / sample_rate
        aligned_transcript += f"[{i}] {seg.label} ({seg.score:.2f}): {start_time:.2f} - {end_time:.2f} sec\n"
    
    filtered_segments = [seg for seg in segments if seg.label not in  ["|","<s>","<pad>","<unk>","</s>"]]
    total_score = sum(seg.score for seg in filtered_segments)
    average_ctc_score = total_score / len(filtered_segments) if filtered_segments else 0  # Avoid division by zero
    
    return aligned_transcript, average_ctc_score

# Example usage
audio_path = "/root/abhinav/pipeline/IndicSpeechFlow/test/5348024557502746_f2026_chunk_0_subchunk_1.wav"
given_transcript = " मीडिया|प्लेयर||खोलें||और||मेरी||जिम|प््लेलिस्ट||चलाए|"
aligned_transcript, avg_ctc_score = force_alignment_and_ctc_score(audio_path, given_transcript)

print("Aligned Transcript:\n", aligned_transcript)
print(f"Average CTC Score: {avg_ctc_score:.2f}")
