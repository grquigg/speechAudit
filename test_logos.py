import requests
import re

def get_out_of_vocab(word):
    response = requests.post("http://www.speech.cs.cmu.edu/cgi-bin/tools/logios/lextool2.pl", files={'wordfile': word})
    html = response.text
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,response.text)
    r2 = requests.get(url[0][0])
    transcription = r2.text
    entry = transcription.replace('\n', '').split('\t')
    return entry