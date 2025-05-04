import os
from pydub import AudioSegment
import shutil

def normalize_audio_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    supported_formats = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if file.lower().endswith(supported_formats):
            try:
                audio = AudioSegment.from_file(file_path)
                # Convert to mono channel and set frame rate to 16000
                audio = audio.set_channels(1).set_frame_rate(16000)
                # Define output path in the specified output folder
                base_name = os.path.splitext(file)[0]
                output_path = os.path.join(output_folder, f"{base_name}.wav")
                # Export the audio as .wav to the output folder
                audio.export(output_path, format="wav")
                print(f"Converted: {file} → {output_path}")

            except Exception as e:
                print(f"Failed to convert {file}: {e}")

def convert_mp3_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    base = os.path.splitext(mp3_path)[0]
    wav_path = base + ".wav"
    audio.export(wav_path, format="wav")
    print(f"Converted: {mp3_path} → {wav_path}")


def merge_folder_with_silence(input_folder, output_file, silence_duration=2000):
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".wav")])
    files = [f for f in files if os.path.basename(f) != "merged_unique_speakers.wav"]
    # print(f"Found {len(files)} audio files")
    if not files:
        print("No audio files found in folder.")
        return
    combined_audio = AudioSegment.silent(duration=0)
    silence = AudioSegment.silent(duration=silence_duration)
    for f in files:
        path = os.path.join(input_folder, f)
        audio = AudioSegment.from_file(path)
        combined_audio += audio + silence
    combined_audio.export(output_file, format="wav")


 
def transcript_quality(transcript):
    words = transcript.strip().split()
    repeated = len(set(words)) < len(words) * 0.7
    chars = set(transcript)
    if len(words) > 3  and not repeated:
        return "passed"
    else:
        return "failed"