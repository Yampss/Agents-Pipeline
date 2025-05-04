import os
import json
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import numpy as np

token = "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"
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
