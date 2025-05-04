import os, re
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from utility import convert_mp3_to_wav, merge_folder_with_silence
from collections import Counter
import tempfile

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
pipeline.to(torch.device(device))    

def get_merged_master_audio(input_folder):
    idle_path = os.path.join(input_folder, "num_files")
    os.makedirs(idle_path, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".mp3"):
            convert_mp3_to_wav(os.path.join(input_folder, file))
            
    wav_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".wav")])
    print(f"Foumd {len(wav_files)} wav files")   
    
    for i in range(len(wav_files)):
        wav_file = os.path.join(input_folder, wav_files[i])
        audio = AudioSegment.from_wav(wav_file)
        output_txt = os.path.join(idle_path, f"{os.path.splitext(os.path.basename(wav_file))[0]}.txt")
        diarization = pipeline(wav_file)
        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            duration = end - start
            if speaker not in speaker_segments or duration > (speaker_segments[speaker][1] - speaker_segments[speaker][0]):
                speaker_segments[speaker] = (start, end)
        with open(output_txt, "w") as f:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                f.write(f"{turn.start:.2f} {turn.end:.2f} {speaker}\n")
        for idx, (speaker, (start, end)) in enumerate(speaker_segments.items()):
            fname = os.path.splitext(wav_files[i])[0]
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            chunk = audio[start_ms:end_ms]
            out_path = os.path.join(idle_path, f"{fname}_speaker_{idx}.wav")
            chunk.export(out_path, format="wav")

    # output_wav = os.path.join(idle_path, "merged_unique_speakers.wav")
    # merge_folder_with_silence(idle_path, output_wav)
    # return output_wav
    return idle_path

def find_number_of_speakers(audio_path, output_wav_folder):
    try:
        for audio_file in os.listdir(output_wav_folder):
            if audio_file.endswith(".wav") and re.sub(r'_speaker_.*', '', os.path.basename(audio_file)) != os.path.basename(audio_path).replace(".wav", ""):
                audio1 = AudioSegment.from_wav(audio_path)
                audio2 = AudioSegment.from_wav(os.path.join(output_wav_folder, audio_file))
                
                silence = AudioSegment.silent(duration=1000)
                merged_audio = audio1 + silence + audio2
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    merged_audio.export(tmp_file.name, format="wav")
                    tmp_path = tmp_file.name
                    
                    diarization = pipeline(tmp_path)
                    print(diarization)

                    speaker_labels = [label for _, _, label in diarization.itertracks(yield_label=True)]
                    speaker_counts = Counter(speaker_labels)

                    os.remove(tmp_path)
                    num_speakers = len(speaker_counts)   
                    common_file = audio_file     
                    if num_speakers == 1:
                        if speaker_counts.get("SPEAKER_00", 0) > 1:
                            return "Old", common_file  # Return "Old" and stop further processing
                    else:
                        # If any speaker is repeated more than once, return "Old"
                        if any(count > 1 for count in speaker_counts.values()):
                            return "Old", common_file

        # If no repeated speakers are found, return "New"
        return "New", ""
        
    except Exception as e:
        print("Error in find_number_of_speakers:", e)
        return e, ""
