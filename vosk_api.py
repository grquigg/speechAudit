import json
from vosk import KaldiRecognizer

def vosk_asr(model, wf):
    """
    Params:
    model: initialized vosk model
    wf: wave file

    Returns:
    response: string
    """    
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    rec.SetPartialWords(True)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            rec.Result()
        else:
            rec.PartialResult()

    res = json.loads(rec.FinalResult())
    return res["text"]