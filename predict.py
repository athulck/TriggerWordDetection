
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, Sequential
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
# import IPython
from model import GRUModel
from td_utils import *


def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    print(type(segment))
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


chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')

import time


model = GRUModel(input_shape = (5511, 101))         #(5511, 101)
import tensorflow as tf
tf.compat.v1.disable_v2_behavior() # model trained in tf1
model = tf.compat.v1.keras.models.load_model('./models/tr_model.h5')
# model = load_model('./models/tr_model.h5')


your_filename = "./soundfile.wav"
chime_threshold = 0.5

s = time.time()
preprocess_audio(your_filename)
prediction = detect_triggerword(your_filename)
chime_on_activate(your_filename, prediction, chime_threshold)
print("Time : ",time.time() - s)
# Output in 'chime_output.wav'