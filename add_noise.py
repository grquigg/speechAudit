# Feature extraction example
import numpy as np
import librosa
from librosa import display
import matplotlib.pyplot as plt
# Load the example clip

def add_noise(audio, method='gaussian', mu=0, sigma=8):
    audio = audio.astype(float)
    X = librosa.stft(audio)
    gaussian = np.random.normal(loc=mu, scale=sigma, size=(X.shape))
    X += gaussian
    # #convert audio back
    Xn = librosa.istft(X)
    return Xn.astype(np.short)

