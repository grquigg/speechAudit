# Feature extraction example
import numpy as np
import librosa
from librosa import display
import matplotlib.pyplot as plt
# Load the example clip

def add_noise(audio, params, method='none'):
    #convert audio into float for adding noise
    audio = audio.astype(float)
    X = librosa.stft(audio)
    if (method=='gaussian'):
        X_noise = add_gaussian_noise(X, params)
    elif(method=='min_suppression'):
        X_noise = minimum_suppression(X, params)
    else:
        X_noise = X
    # #convert audio back into signal
    Xn = librosa.istft(X)
    #return the noisy audio signal as a short int for deepspeech
    return Xn.astype(np.int16)

def add_gaussian_noise(X, params):
    gaussian = np.random.normal(loc=params[0], scale=params[1], size=(X.shape))
    X += gaussian
    return X

def minimum_suppression(X, params):
    threshold = params[0]
    mask = np.greater(X,threshold)
    X = np.multiply(X, mask)
    return X
