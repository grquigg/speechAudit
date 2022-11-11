import requests
import os
#retrieve CMU pronunciation dict
response = requests.get("http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/sphinxdict/cmudict.0.7a_SPHINX_40")
open("cmudict.0.7a_SPHINX_40", "wb").write(response.content)
#manually read in csv; the formatting of the file breaks pd.read_csv
dict = {}
phones = []
with open("cmudict.0.7a_SPHINX_40", "r") as file:
    for line in file:
        entry = line.replace('\n', '').split('\t')
        dict[entry[0]] = entry[1]

with open("Sphinx_phones_40", 'r') as phone_file:
    for line in phone_file:
        phones.append(line.replace('\n', ''))

def transcribe_audio(file_path, from_file=False):
    transcript = []
    with open(file_path, "r") as f:
        for line in f:
            transcript = line.upper()
            transcript = transcript.replace('\n', '')
            transcript = transcript.replace('.', '')
            transcript = transcript.split(' ')
    if(from_file):
        transcript = transcript[2:]
    phonetic = []
    for word in transcript:
        transcribe = dict[word]
        transcribe = transcribe.split(' ')
        phonetic += transcribe
    return phonetic

def main():

    print(transcribe_audio('archive/data/TEST/DR1/FAKS0/SA1.TXT'))
if __name__ == '__main__':
    actual_transcriptions = []
    predicted_transcriptions = []
    in_dir = "output"
    text_dir = "archive/data/TEST"
    s_list = os.listdir(text_dir)

    dir_list = os.listdir(text_dir)
    for dir in dir_list:
        i_dir2 = os.path.join(text_dir, dir)

        subsub_dir = os.listdir(i_dir2)
        for sub_dir in subsub_dir:
            read_dir = os.path.join(i_dir2, sub_dir)

            files = os.listdir(read_dir)
            for fname in files:
                if(fname[-4:] == '.wav'):
                    name_base = fname[:-8]
                    true_trans_path = os.path.join(read_dir, name_base+".TXT")
                    true_transcription = transcribe_audio(true_trans_path, from_file=True)
                    actual_transcriptions.append(true_transcription)
                    pred_trans_path= os.path.join(in_dir, dir+"/"+sub_dir+"/"+name_base+".WAV.txt")
                    pred_transcription = transcribe_audio(pred_trans_path)