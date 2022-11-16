import requests
import os
import copy
import jiwer
import numpy as np
import jellyfish
import pandas as pd
from collections import Counter
#retrieve CMU pronunciation dict
# response = requests.get("http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/sphinxdict/cmudict.0.7a_SPHINX_40")
# open("cmudict.0.7a_SPHINX_40", "wb").write(response.content)
#manually read in csv; the formatting of the file breaks pd.read_csv
dict = {}
phones = []
global OUT_OF_VOCAB
global ALIGNS
OUT_OF_VOCAB = 0
ALIGNS = 0
VOWELS = ['AA', 'AE', 'AH', 'AW', 'AY', 'EH', 'EY', 'OW', 'OY', 'UH', 'UW']
with open("cmudict.0.7a_SPHINX_40", "r") as file:
    for line in file:
        entry = line.replace('\n', '').split('\t')
        # print(entry)
        dict[entry[0]] = entry[1]

with open("Sphinx_phones_40", 'r') as phone_file:
    for line in phone_file:
        phones.append(line.replace('\n', ''))

def transcribe_text(file_path, from_file=False):
    transcript = []
    with open(file_path, "r") as f:
        for line in f:
            transcript = line.upper()
            transcript = transcript.replace('\n', '')
            transcript = transcript.replace('.', '')
            transcript = transcript.replace(',', '')
            transcript = transcript.replace(';', '')
            transcript = transcript.replace(':', '')
            transcript = transcript.replace('?', '')
            transcript = transcript.replace('!', '')
            transcript = transcript.replace('-','')
            # transcript = transcript.replace("'", "")
            transcript = transcript.split(' ')
    if(from_file):
        transcript = transcript[2:]
    return transcript
def transcribe_audio(file_path, from_file=False):
    transcript = []
    with open(file_path, "r") as f:
        for line in f:
            transcript = line.upper()
            transcript = transcript.replace('\n', '')
            transcript = transcript.replace('.', '')
            transcript = transcript.replace(',', '')
            transcript = transcript.replace(';', '')
            transcript = transcript.replace(':', '')
            transcript = transcript.replace('?', '')
            transcript = transcript.replace('!', '')
            transcript = transcript.replace('-', '')
            # transcript = transcript.replace("'", "")
            transcript = transcript.split(' ')
    if(from_file):
        transcript = transcript[2:]
    phonetic = []
    for word in transcript:
        if word in dict:
            transcribe = dict[word].split(' ')
            phonetic += transcribe
        else:
            global OUT_OF_VOCAB
            print("OUT OF VOCAB: {}".format(word))
            transcribe_out_of_vocab(word)
            OUT_OF_VOCAB += 1
    return phonetic

def transcribe(actual_words, pred_words, actual_phones, pred_phones, matrix, phone_dict):
    #assume that the phonemes in the intersection between the words recognized and the actual words is correct
    #CORE ASSUMPTION #1: Words only appear once in each phrase
    #This is a bad assumption but one that can be remediated
    distance = jellyfish.levenshtein_distance(' '.join(pred_phones),' '.join(actual_phones))
    print(actual_phones)
    print(pred_phones)
    print("Prediction is off by {} phones".format(distance))
    correct_words = set(word_actual).intersection(word_pred)
    print(correct_words)
    #set correct predictions in the confusion matrix
    if(len(actual_phones) == len(pred_phones)): #then we can map the phones 1-to-1
        global ALIGNS
        ALIGNS += 1
        for i in range(len(actual_phones)):
            confusion_matrix[phone_dict[actual_phones[i]]][phone_dict[pred_phones[i]]] += 1
    else:
        for word in correct_words:
            t = dict[word]
            t = t.split(' ')
            for phone in t:
                confusion_matrix[phone_dict[phone]][phone_dict[phone]] += 1
    #now for incorrect predictions
    #correct predictions allow us to establish basis for comparing incorrect predictions
    # diff_pred = set(word_pred) - set(word_actual).intersection(word_pred)
    # diff_actual = set(word_actual)- set(word_actual).intersection(word_pred)
    # print(diff_pred)
    # print(diff_actual)
    # for i in range(len(actual_words)):
    #     word = actual_words[i]
    #     if(word in diff_actual):
    #         print(word)
    #         if(i != 0):
    #             counter_left = 1
    #             context_left = actual_words[i-counter_left]
    #             while(context_left not in pred_words or i-counter_left > 0):
    #                 counter_left+=1
    #                 context_left = actual_words[i-counter_left]
    #             print("Context left")
    #             print(context_left)
    #         else:
    #             counter_left = 1
    #         counter_right = 1
    #         context_right = actual_words[i+counter_right]
    #         while(context_right not in pred_words):
    #             counter_right+=1
    #             context_right = actual_words[i+counter_right]
    #         print("Context right")
    #         print(context_right)
    #         #find left context in predicted string
    #         if(i != 0):
    #             left_index = pred_words.index(context_left)
    #             print(left_index)
    #         else:
    #             left_index = -1
    #         #find right context in predicted string
    #         right_index = pred_words.index(context_right)
    #         print(right_index)
    #         comparison = pred_words[left_index+1:right_index]
    #         phones_true = []
    #         phones_err = []
    #         true_context = actual_words[i-counter_left+1:i+counter_right]
    #         print("True context")
    #         print(true_context)
    #         for w in comparison:
    #             phones_err += dict[w].split(' ')
    #         for x in true_context:
    #             phones_true = dict[x].split(' ')
    #             print(phones_true)
    #             print(phones_err)
    #             #align consonants first
    #             #check to see whether or not there are any phones in common
    #             common_phones = set(phones_true).intersection(phones_err)
    #             print(common_phones)
    #             raise NotImplementedError()

def compute_accuracy(actual_phones, predicted_phones, matrix, phones_dict):
    for i in range(len(actual_phones)):
        temp = np.zeros(matrix.shape, dtype=int)
        for j in range(len(actual_phones[i])):
            ph = actual_phones[i][j]
            temp[1, phones_dict[ph]] += 1
        for k in range(len(predicted_phones[i])):
            ph2 = predicted_phones[i][k]
            temp[0, phones_dict[ph2]] += 1
            if(temp[0, phones_dict[ph2]] > temp[1, phones_dict[ph2]]): #handle overpredictions
                temp[0, phones_dict[ph2]] = temp[1, phones_dict[ph2]].copy()
                temp[2, phones_dict[ph2]] += 1
        matrix += temp
    pass

def transcribe_out_of_vocab(word):
    pass
def main():

    print(transcribe_audio('archive/data/TEST/DR1/FAKS0/SA1.TXT'))
if __name__ == '__main__':
    actual_transcriptions = []
    predicted_transcriptions = []
    actual_trans = []
    predicted_trans = []
    in_dir = "output"
    text_dir = "archive/data/TEST"
    s_list = os.listdir(text_dir)
    phones_dict = {}
    for i in range(len(phones)):
        phones_dict[phones[i]] = i
    confusion_matrix = np.zeros((len(phones), len(phones)), dtype=int)
    decision_matrix = np.zeros((3, len(phones)))
    dir_list = os.listdir(text_dir)
    num_audio = 0
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
                    pred_trans_path= os.path.join(in_dir, dir+"/"+sub_dir+"/"+name_base+".WAV.txt")
                    word_actual = transcribe_text(true_trans_path, from_file=True)
                    word_pred = transcribe_text(pred_trans_path)
                    print(word_actual)
                    print(word_pred)
                    true_transcription = transcribe_audio(true_trans_path, from_file=True)
                    print(true_transcription)
                    actual_transcriptions.append(' '.join(true_transcription))
                    actual_trans.append(true_transcription)
                    pred_transcription = transcribe_audio(pred_trans_path)
                    print(pred_transcription)
                    if(len(pred_transcription) == 0):
                        raise NotImplementedError()
                    predicted_transcriptions.append(' '.join(pred_transcription))
                    predicted_trans.append(pred_transcription)
                    transcribe(word_actual, word_pred, true_transcription, pred_transcription, confusion_matrix, phones_dict)
                    index = 0
                    max_align_dist = 1 #assume that transcriptions are off by only one or two phonemes
                    previous_error_rate = 0
                    num_audio += 1
    print("TOTAL NUMBER OF OUT OF CONTEXT WORDS: {}".format(OUT_OF_VOCAB))
    error = jiwer.compute_measures(predicted_transcriptions, actual_transcriptions)
    print("Phone error rate: {}".format(error['wer']))
    compute_accuracy(actual_trans, predicted_trans, decision_matrix, phones_dict)
    print("Accuracy per phone")
    accuracy_per_phone = decision_matrix[0] / decision_matrix[1]
    print(accuracy_per_phone)
    poorest_identify = np.argmin(accuracy_per_phone)
    print("Phone with lowest accuracy is {} with accuracy of {}".format(phones[poorest_identify], np.min(accuracy_per_phone)))
    print("Phone with lowest number of appearances in data: {}".format(phones[np.argmin(decision_matrix[1])]))
    print("Overpredictions:")
    print(decision_matrix[2])
    print(confusion_matrix)
    percent = ALIGNS / num_audio
    print("Total percentage of transcriptions where predicted number of phones equals actual number of phones: {}".format(percent))
