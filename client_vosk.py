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
    model = Model("vosk-model-small-en-us-0.15")

    noise_levels = [1]
    for level in noise_levels:
        i_dir = "archive/data/TEST/"
        o_dir = "output_vosk_long"
        NOISE_PARAMS = {}
        NOISE_METHOD = "lengthen"

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
        dir_list = os.listdir(i_dir)
        for dir in dir_list:
            i_dir2 = os.path.join(i_dir, dir)
            subout_dir = os.path.join(o_dir, dir)
            #create corresponding directory in output
            if(not os.path.isdir(subout_dir)):
                os.mkdir(subout_dir)
            subsub_dir = os.listdir(i_dir2)
            for sub_dir in subsub_dir:
                read_dir = os.path.join(i_dir2, sub_dir)

                outdir = os.path.join(subout_dir, sub_dir)
                if(not os.path.isdir(outdir)):
                    os.mkdir(outdir)
                files = os.listdir(read_dir)
                print(files)
                for fname in files:
                    if(fname[-4:] == '.wav'):
                        flin = os.path.join(read_dir, fname)
                        fout = os.path.join(outdir, f"{fname[:-4]}.json")
                        
                        fin = wave.open(flin, "rb")
                        if fin.getnchannels() != 1 or fin.getsampwidth() != 2 or fin.getcomptype() != "NONE":
                            print(f"Audio file {fin} must be WAV format mono PCM.")
                            sys.exit(1)
                        
                        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
                        audio = add_noise(audio, NOISE_PARAMS, method=NOISE_METHOD)

                        inter_fname = os.path.join(interim_dir, f"noised_{fname}")
                        
                        scipy.io.wavfile.write(inter_fname, fin.getframerate(), audio)
                        wf = wave.open(inter_fname, "rb")

                        with open(fout, "w") as file:
                            file.write(vosk_asr(model, wf))

if __name__ == "__main__":
    main()

    