#environment: nemo-abhinav
import string, re
import pandas as pd
import numpy as np
import torch
import os
from utility import normalize_audio_folder, transcript_quality
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transcription import transcribe_audio
from embedding import save_embeddings
from vad import get_total_silence_time
from total_speaker_count import get_merged_master_audio, find_number_of_speakers
from ctc_score_given_transcript import forced_allignemnt_given_transcript
import sys

##### USER INPUTS #####
print("\n\nWelcome to IndicSpeechFlow pipeline.\n\n")
source_lang = "Hindi"
input_folder = input("Give path to the input folder containing the audios: ")
if os.path.exists(input_folder):
    pass
else:
    print(f"Invalid folder path: {input_folder}. Please check the path and try again.")
    sys.exit(1)
    
#######################

output_folder = os.path.join(input_folder, "converted_audios")
normalize_audio_folder(input_folder, output_folder)
os.makedirs(output_folder, exist_ok=True)
idle_path = os.path.join(output_folder, "idle_files")
os.makedirs(idle_path, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
pipeline.to(torch.device(device))    
        
wav_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".wav")])
print(f"Found {len(wav_files)} wav files")   

def clean_sentence(input_string):
    if not isinstance(input_string, str):
        return ""    
    remove_chars = string.punctuation + '|'
    for char in remove_chars:
        input_string = input_string.replace(char, " ")
    input_string = input_string.replace("-", " ").replace("–", " ").replace("—", " ")
    result = input_string.replace("।", " ").strip()
    result = result.replace(':', ' ')
    result = result.strip()
    return re.sub(r'\s+', ' ', result)
         
def get_vocab(sentence):
    # sentence = clean_sentence(sentence)
    words = sentence.split()
    return words

def get_characters(sentence):
    # sentence = clean_sentence(sentence)
    seen = set()
    unique_chars = []
    for char in sentence:
        if char not in seen:
            if char != " ":
                seen.add(char)
                unique_chars.append(char)
    return sorted(unique_chars)


transcripts = []
num_speakers = []
dia = []
vocab_list = []
num_words = []
character_list = []
silence_durations = []
total_durations = []
forced_allignemnts = []
ctc_scores = []
speaker_status = []
common_files = []
quality = []

merged_audio_chunks = get_merged_master_audio(output_folder)
for i in range(len(wav_files)):
    
    ### number of speakers 
    wav_file = os.path.join(output_folder, wav_files[i])
    diarization = pipeline(wav_file)
    unique_speakers = set()
    dia.append(wav_files[i])
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dia.append(f"{turn.start:.2f} {turn.end:.2f} {speaker}")
        unique_speakers.add(speaker)
    dia.append("")
    num_speakers.append(len(unique_speakers))
        
    ### transcript indicconf  hindi
    transcript = transcribe_audio(wav_file, source_lang)
    transcript = clean_sentence(transcript)
    transcripts.append(transcript)
    
    ### vocab_list 
    voc = get_vocab(transcript)
    vocab_list.append(voc)
    num_words.append(len(voc))
    
    ### character_list
    characters = get_characters(transcript)
    character_list.append(characters)

    ### silence duration
    silence_time = get_total_silence_time(wav_file)
    silence_durations.append(round(silence_time, 2))

    ### total duration
    audio = AudioSegment.from_wav(wav_file)
    duration_secs = audio.duration_seconds
    total_durations.append(round(duration_secs, 2))
    
    ### forced allignment and CTC score
    aligned_transcript, average_ctc_score = forced_allignemnt_given_transcript(wav_file, transcript)
    ctc_scores.append(average_ctc_score)       
    forced_allignemnts.append(aligned_transcript)      
        
    ### Repeated Speaker status
    status, common_file = find_number_of_speakers(wav_file, merged_audio_chunks)
    speaker_status.append(status)
    common_files.append(common_file)
    
    ### Transcript Quality
    transcript_qual = transcript_quality(transcript)
    quality.append(transcript_qual)
    
df = pd.DataFrame({
    "File": wav_files,
    "Num Speakers": num_speakers,
    "Indicconformer Hypothesis": transcripts,
    "Vocab List": vocab_list,
    "Character List": character_list,
    "Number of Words": num_words,
    "Silence Duration": silence_durations,
    "Total Audio Duration": total_durations,
    "Forced Alignment (Label, Start_time, End_time, CTC_Score)": forced_allignemnts,
    "Average CTC Scores": ctc_scores,
    "Speaker Status" : speaker_status,
    "Common Speaker File Sample" : common_files,
    "Transcript Quality": quality, 
})

output_csv = os.path.join(idle_path, "audio_indicspeech_results.csv")
df.to_csv(output_csv, index=False)

### speaker embedding 
output_path = os.path.join(idle_path, "audios_pyannote_speaker_embeddings.pkl")
save_embeddings(output_folder, output_path)

### diarization results
output_diarization = os.path.join(idle_path, "audio_indicspeech_diarization_results.txt")
with open(output_diarization, "w") as f:
    for line in dia:
        f.write(line + "\n")

print("Pipeline completed successfully.")
print(f"Output CSV saved to: {output_csv}")