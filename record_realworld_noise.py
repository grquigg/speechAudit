import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # Sample rate
seconds = 3  # Duration of recording
for i in range(100):
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished

    write('realworld/output{}.wav'.format(i), fs, myrecording)  # Save as WAV file
