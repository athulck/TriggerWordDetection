import pyaudio
import wave

import numpy as np
from keras.models import Model, load_model, Sequential
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
from model import GRUModel
from td_utils import *
import time


def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')

def detect_triggerword(filename):

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    return predictions

# Audio input from microphone
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "soundfile.wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)


# Prediction model
model = GRUModel(input_shape = (5511, 101))         #(5511, 101)
model = load_model('model_01.h5')


chime_threshold = 0.5

while True:
	s = time.time()
	print ("recording...")
	frames = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
	print ("finished recording")

	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(frames))
	waveFile.close()

	preprocess_audio(WAVE_OUTPUT_FILENAME)
	prediction = detect_triggerword(WAVE_OUTPUT_FILENAME)

	consecutive_timesteps = 0
	for i in range(prediction.shape[1]):
		consecutive_timesteps += 1
		if prediction[0,i,0] > chime_threshold and consecutive_timesteps > 75:
			print("Trigger Word Detected !")
			consecutive_timesteps = 0

	#print("Time : ", time.time() - s)

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
