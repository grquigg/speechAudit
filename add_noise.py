# Feature extraction example
import numpy as np
import librosa
import os
import wave

def add_noise(audio, params={}, method='none'):
    #convert audio into float for adding noise
    audio = audio.astype(float)
    X = librosa.stft(audio)
    if (method=='gaussian'):
        X_noise = add_gaussian_noise(X, params)
    elif(method=='min_suppression'):
        X_noise = minimum_suppression(X, params)
    elif(method=='superimpose'):
        X_noise = superimpose(X, params)
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

def superimpose(X, params):
    # TODO
    # - (IMPORTANT) come up with a way to pre-load or store the noise
    # instead of reading it every time for every input !!
    # - get sample_rate from client.py to make sure both audio & noise
    # files are sampled in the same sample_rate
    # - add multiple noise_src files & change noise_mag to be random weights
    # to have weighted noise added to X (maybe even make sure that the sum
    # of all the random weights equal to a max_mag)

    no_dir = params['noise_dir'] if 'noise_dir' in params else 'noise_src'
    noise_mag = params['noise_mag'] if 'noise_mag' in params else 1
    
    for no_file in os.listdir(no_dir):
        fin = wave.open(os.path.join(no_dir, no_file))
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        noise = librosa.stft(audio.astype(float))
        assert(X.shape[0] == noise.shape[0]) # n_fft should be same

        if noise.shape[1] > X.shape[1]:
            start_idx = np.random.randint(0, noise.shape[1] - X.shape[1] + 1)
            noise_sam = noise[:, start_idx:start_idx + X.shape[1]]
        else:
            num_stack = 1 + X.shape[1] // noise.shape[1]
            noise_sam = np.tile(noise, num_stack)[:, :X.shape[1]]
        
        X += noise_sam * noise_mag

    return X
