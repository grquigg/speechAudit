# Feature extraction example
import math
import random

import numpy as np
import librosa
import wave
import audiomentations
from audiomentations import AddGaussianSNR, TimeStretch, PitchShift, Shift
from PIL import Image
import simpleaudio as sa
from scipy.io import wavfile
from librosa import display
import matplotlib.pyplot as plt
import os
# Load the example clip
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import glob

def add_noise(audio, params={}, method='none'):
    #convert audio into float for adding noise
    audio = audio.astype(float)
    X = librosa.stft(audio)
    if (method=='gaussian'):
        X_noise = add_gaussian_noise(X, params)
    elif(method=='min_suppression'):
        print("Min suppression")
        X_noise = minimum_suppression(X, params)
    elif(method=='superimpose'):
        X_noise = superimpose(X, params)
    elif(method=='lengthen'):
        X_noise = lengthen(audio).astype(np.int16)
        # sa.play_buffer(X_noise, 1, 2, 16000).wait_done()
        return X_noise
    elif(method=='cutout'):
        X_noise = add_cutout_noise(X, params)
    elif(method=='real_world_noise'):
        X_noise = add_real_world_noise(X, params)
    else:
        X_noise = X
    # #convert audio back into signal
    Xn = librosa.istft(X_noise)
    # sa.play_buffer(Xn.astype(np.int16), 1, 2, 16000).wait_done()
    #return the noisy audio signal as a short int for deepspeech
    return Xn.astype(np.int16)

def add_gaussian_noise(X, params):
    mean = params['mean'] if 'mean' in params else 0
    std = params['std'] if 'std' in params else 1
    gaussian = np.random.normal(loc=mean, scale=std, size=(X.shape))
    X += gaussian
    return X

def minimum_suppression(X, params):
    threshold = params['threshold'] if 'threshold' in params else 5000
    mask = np.greater(X,threshold)
    X = np.multiply(X, mask)
    return X

# Helper class to pre-load the noise stfts
class SuperImposeNoise:
    def __init__(self):
        self.mag = 0
        self.noi = []
        self.__loaded = False

    def is_loaded(self):
        return self.__loaded

    def load_noise(self, noise_dir, noise_mag):
        for nf in os.listdir(noise_dir):
            fin = wave.open(os.path.join(noise_dir, nf))
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
            noise = librosa.stft(audio.astype(float))
            self.noi.append(noise)
        self.mag = noise_mag
        self.__loaded = True
    
    def add_noise(self, X):
        for noise in self.noi:
            assert(X.shape[0] == noise.shape[0]) # n_fft should be same
            if noise.shape[1] > X.shape[1]:
                start_idx = np.random.randint(0, noise.shape[1] - X.shape[1] + 1)
                noise_sam = noise[:, start_idx:start_idx + X.shape[1]]
            else:
                num_stack = 1 + X.shape[1] // noise.shape[1]
                noise_sam = np.tile(noise, num_stack)[:, :X.shape[1]]
            X += noise_sam * self.mag
        return X

SINObj = SuperImposeNoise()
def superimpose(X, params):
    # TODO
    # - get sample_rate from client.py to make sure both audio & noise
    # files are sampled in the same sample_rate
    # - add multiple noise_src files & change noise_mag to be random weights
    # to have weighted noise added to X (maybe even make sure that the sum
    # of all the random weights equal to a max_mag)

    noise_dir = params['noise_dir'] if 'noise_dir' in params else 'noise_src'
    noise_mag = params['noise_mag'] if 'noise_mag' in params else 1

    if not SINObj.is_loaded():
        SINObj.load_noise(noise_dir, noise_mag)
    return SINObj.add_noise(X)

def lengthen(audio):

    audio_transforms = audiomentations.Compose([
        TimeStretch(min_rate=0.5, max_rate=0.5, leave_length_unchanged=False, p=1)
    ])

    y = audio_transforms(samples=audio, sample_rate=8000)
    # play the augmented audio signal

    return y

def add_real_world_noise(audio, params):
    # params is the SNR, range from 90 to 110.
    # read_dir = 'realworld/output{}.wav'.format()
    soundfiles = glob.glob("realworld/*.wav")
    samplerate, data = wavfile.read(random.choice(soundfiles))
    noise_spec = librosa.stft(data[:,0])
    if(noise_spec.shape[1] > audio.shape[1]):
        noise_spec = noise_spec[:,:audio.shape[1]]
    noise = np.zeros(audio.shape, dtype=complex)
    noise[:,:noise_spec.shape[1]] = noise_spec[:audio.shape[0],:]
    if(len(noise_spec) < len(audio)): #in the case that the noise audio is shorter than the actual audio, just replay
        for i in range(0, len(data), len(audio)):
            noise[i:i+len(data),:] = noise_spec.copy()
        mod = len(audio) % len(data)
        noise[-mod:,:] = noise_spec[:mod,:].copy()
    # #this is E[N^2], so we want the squared mean of the noise to match this value
    RMS_noise = np.mean(audio**2)/params['SNR']
    a = np.sqrt(np.divide(RMS_noise,np.mean(noise**2)))
    result_noise = np.multiply(a,noise)
    result = audio+np.multiply(a,noise)

    return result

def add_cutout_noise(X, params):
    h, w = X.shape[0], X.shape[1]
    mask = np.ones((h, w), np.int16)
    for n in range(params["cutout"]):
        y = np.random.randint(h-10)
        x = np.random.randint(w-10)

        mask[y: y+10, x: x+10] = 0.

    X = X * mask

    return X

def apply_audio_transforms(audio):
    # audio augmentation:
    #   Gaussian noise is added with prob 0.5,
    #   TimeStretch with prob 0.5
    #   PitchShift with prob 0.5
    #   Shift with prob 0.5
    audio_transforms = audiomentations.Compose([
        AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=40.0, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])

    y = audio_transforms(samples=audio, sample_rate=16000)

    # play the augmented audio signal

    return y