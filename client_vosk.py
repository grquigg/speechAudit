from vosk import Model, SetLogLevel
import os, sys
import wave

import numpy as np
import scipy.io.wavfile

from vosk_api import vosk_asr
from add_noise import add_noise

# no debug pls
SetLogLevel(-1)

def main():
    # can use a better model - https://alphacephei.com/vosk/models
    model = Model("model/vosk-model-small-en-us-0.15")

    i_dir = "test_audio"
    o_dir = "output_test_vosk"
    NOISE_PARAMS = {}
    NOISE_METHOD = "none"

    # stroing noised inputs in dir 
    # TODO:
    # CHANGE 
    interim_dir = "vosk_noise_audio_dir"
    if not os.path.isdir(interim_dir):
        print(f"--- Creating {interim_dir}")
        os.mkdir(interim_dir)

    if not os.path.isdir(o_dir):
        print(f"--- Creating {o_dir}")
        os.mkdir(o_dir)

    for file in os.listdir(i_dir):
        flin = os.path.join(i_dir, file)
        fout = os.path.join(o_dir, f"{file[:-4]}.txt")
        
        fin = wave.open(flin, "rb")
        if fin.getnchannels() != 1 or fin.getsampwidth() != 2 or fin.getcomptype() != "NONE":
            print(f"Audio file {fin} must be WAV format mono PCM.")
            sys.exit(1)
        
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        audio = add_noise(audio, NOISE_PARAMS, method=NOISE_METHOD)

        inter_fname = os.path.join(interim_dir, f"noised_{file}")
        
        scipy.io.wavfile.write(inter_fname, fin.getframerate(), audio)
        wf = wave.open(inter_fname, "rb")

        with open(fout, "w") as file:
            file.write(vosk_asr(model, wf))
            file.close()


if __name__ == "__main__":
    main()

    