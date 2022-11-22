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
    Xn = librosa.istft(X_noise)
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
