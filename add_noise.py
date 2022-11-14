# Feature extraction example
import numpy as np
import librosa
from librosa import display
import matplotlib.pyplot as plt
# Load the example clip

def add_noise(audio, method='gaussian', mu=0, sigma=8):
    #convert audio into float for adding noise
    audio = audio.astype(float)
    X = librosa.stft(audio)
    gaussian = np.random.normal(loc=mu, scale=sigma, size=(X.shape))
    X += gaussian
    # #convert audio back into signal
    Xn = librosa.istft(X)
    #return the noisy audio signal as a short int for deepspeech
    return Xn.astype(np.short)

