import requests
import os
import copy
import jiwer
import numpy as np
import jellyfish
from test_logos import get_out_of_vocab
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
#retrieve CMU pronunciation dict
# response = requests.get("http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/sphinxdict/cmudict.0.7a_SPHINX_40")
# open("cmudict.0.7a_SPHINX_40", "wb").write(response.content)
#manually read in csv; the formatting of the file breaks pd.read_csv
dict = {}
phones = []
VOWELS = ['AA', 'AE', 'AH', 'AW', 'AY', 'EH', 'EY', 'OW', 'OY', 'UH', 'UW']
ORTH_VOWELS = ['A', 'E', 'I', 'O', 'U']
CONSONANTS = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'T', 'TH', 'V', 'W', 'Z', 'ZH']

with open("cmudict.0.7a_SPHINX_40", "r") as file:
    for line in file:
        entry = line.replace('\n', '').split('\t')
        # print(entry)
        dict[entry[0]] = entry[1]

with open("Sphinx_phones_40", 'r') as phone_file:
    for line in phone_file:
        phones.append(line.replace('\n', ''))

phones.append("NONE")
def transcribe_text(file_path, from_file=False):
    transcript = []
    with open(file_path, "r") as f:
        for line in f:
            transcript = line.upper()
            transcript = transcript.replace("\n", "")
            transcript = transcript.replace(". ", " ")
            transcript = transcript.replace(".", "")
            transcript = transcript.replace(",", "")
            transcript = transcript.replace(";", "")
            transcript = transcript.replace(":", "")
            transcript = transcript.replace("?", "")
            transcript = transcript.replace("!", "")
            transcript = transcript.replace("-- ","")
            transcript = transcript.replace("-", " ")
            transcript = transcript.replace('"', '')
            transcript = transcript.split(' ')
            if '' in transcript:
                transcript.remove('')
    if(from_file):
        transcript = transcript[2:]
    return transcript

def transcribe_audio(file_path, from_file=False):
    transcript = []
    with open(file_path, "r") as f:
        for line in f:
            transcript = line.upper()
            transcript = transcript.replace("\n", "")
            transcript = transcript.replace(". ", " ")
            transcript = transcript.replace(".", "")
            transcript = transcript.replace(",", "")
            transcript = transcript.replace(";", "")
            transcript = transcript.replace(":", "")
            transcript = transcript.replace("?", '')
            transcript = transcript.replace("!", '')
            transcript = transcript.replace("-- ",'')
            transcript = transcript.replace("-", " ")
            transcript = transcript.replace('"', '')
            transcript = transcript.split(' ')
            if '' in transcript:
                transcript.remove('')
    if(from_file):
        transcript = transcript[2:]
    phonetic = []
    for word in transcript:
        if word in dict:
            transcribe = dict[word].split(' ')
            phonetic += transcribe
        else:
            response = get_out_of_vocab(word)
            dict[response[0]] = response[1]
            transcribe = dict[word].split(' ')
            phonetic += transcribe
    return phonetic

def align_phones(actual_phones, pred_phones):
    pass
    # print(actual_phones)
    # print(pred_phones)
    if(len(actual_phones) == 0 and len(pred_phones) == 0):
        # print("BAD")
        return []
    alignments = []
    search_range = np.abs(len(actual_phones) - len(pred_phones)) + 2
    best_cluster = []
    best_cluster_length = 0
    align_border_left = 0
    align_true_border_left = 0
    for i in range(len(actual_phones)):
        for j in range(max(i-search_range, 0), min(i+search_range, len(pred_phones))):
            #special case
            if(actual_phones[i] == pred_phones[j] or 'R' in actual_phones[i] and 'R' in pred_phones[j]):
                index = 0
                current_cluster = []
                while(actual_phones[i+index] == pred_phones[j+index] or 'R' in actual_phones[i+index] and 'R' in pred_phones[j+index]):
                    current_cluster.append(copy.deepcopy(actual_phones[i+index]))
                    index += 1
                    if(len(current_cluster) > best_cluster_length):
                        best_cluster = current_cluster
                        best_cluster_length = len(current_cluster)
                        align_border_left = j
                        align_true_border_left = i
                    # print(current_cluster)
                    if(i + index >= len(actual_phones) or j + index >= len(pred_phones)):
                        break
                i += index
                if i >= len(actual_phones):
                    break
    if(best_cluster_length == 0):
        # if(len(actual_phones) != 0 and len(pred_phones) > len(actual_phones)):
        #     raise NotImplementedError
        for i in range(len(actual_phones)):
            if(i >= len(pred_phones)):
                alignments.append((actual_phones[i], "NONE"))
            else:
                alignments.append((actual_phones[i], pred_phones[i]))
        if(len(pred_phones) > len(actual_phones)):
            for i in range(len(actual_phones), len(pred_phones)):
                alignments.append(("NONE", pred_phones[i]))
        return alignments
    # print(align_border_left)
    # print(align_true_border_left)
    # print("Align cluster")
    for k in range(best_cluster_length):
        alignments.append((actual_phones[align_true_border_left+k], pred_phones[align_border_left+k]))
    left_phones = align_phones(actual_phones[0:align_true_border_left], pred_phones[0:align_border_left])
    for i in range(len(left_phones)):
        alignments.insert(i, left_phones[i])
    # print(alignments)
    # print("Align right")
    right_phones = align_phones(actual_phones[align_true_border_left+best_cluster_length:], pred_phones[align_border_left+best_cluster_length:])
    for i in range(len(right_phones)):
        alignments.append(right_phones[i])
    # print(alignments)
    return alignments

def transcribe(actual_words, pred_words, actual_phones, pred_phones, matrix, phone_dict):
    #assume that the phonemes in the intersection between the words recognized and the actual words is correct
    #CORE ASSUMPTION #1: Words only appear once in each phrase
    #This is a bad assumption but one that can be remediated
    distance = jellyfish.levenshtein_distance(' '.join(pred_phones),' '.join(actual_phones))
    print(actual_phones)
    print(pred_phones)
    print("Prediction is off by {} phones".format(distance))
    correct_words = set(word_actual).intersection(word_pred)
    # print(correct_words)
    #set correct predictions in the confusion matrix
    if(len(actual_phones) == len(pred_phones)): #then we can map the phones 1-to-1
        for i in range(len(actual_phones)):
            confusion_matrix[phone_dict[actual_phones[i]]][phone_dict[pred_phones[i]]] += 1
    else:
        alignment = align_phones(actual_phones, pred_phones)
        print(alignment)
        for i in range(len(alignment)):
            confusion_matrix[phone_dict[alignment[i][0]]][phone_dict[alignment[i][1]]] += 1

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
    #check if the word is simply a subpart of another word
    sub_matches = []
    super_matches = []
    if(word[-1:] == "S" and word[:-1] in dict):
        print(word)
        dict[word] = dict[word[:-1]] + " Z"
        print(dict[word])
        return
    for key in dict.keys():
        if word in key:
            #very specific edge case
            match = key.find(word)
            if(match+len(word) != len(key)):
                if(word[-1:] == "D" and key[match+len(word)] == "G"):
                    continue
            print("Subword whole")
            print(key)
            print(match)
            print(dict[key])
            super_matches.append(key)
            print(dict[key].split(' ')[match:])

        if key in word: #look for better context
            sub_matches.append(key)
    # if(len(super_matches) > 0):
    #     raise NotImplementedError("Should implement this")
    # print("Subword matches for {}".format(word))
    if(len(sub_matches) > 0):
    #then use the matches to construct transcriptions of these OOV words
        sub_matches = sorted(sub_matches, key=lambda x: len(x), reverse=True)
        # print(sub_matches)
        # print(word)
        # mod_word = copy.deepcopy(word)
        # for i in sub_matches:
        #     print(i)
        #     print(dict[i])
        #     candidate = 
    #     raise NotImplementedError("Should implement this")
    # if(len(sub_matches) == 0 and len(super_matches) == 0):
    #     raise ValueError("No idea what to do with this")

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
                    print(' '.join(true_transcription))
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
    error = jiwer.compute_measures(predicted_transcriptions, actual_transcriptions)
    print("Phone error rate: {}".format(error['wer']))
    compute_accuracy(actual_trans, predicted_trans, decision_matrix, phones_dict)
    print("Accuracy per phone")
    accuracy_per_phone = decision_matrix[0][:-1] / decision_matrix[1][:-1]
    print(accuracy_per_phone)
    poorest_identify = np.argmin(accuracy_per_phone)
    print("Phone with highest accuracy is {} with accuracy of {}".format(phones[np.argmax(accuracy_per_phone)], np.max(accuracy_per_phone)))
    print("Phone with lowest accuracy is {} with accuracy of {}".format(phones[poorest_identify], np.min(accuracy_per_phone)))
    print("Phone with lowest number of appearances in data: {}".format(phones[np.argmin(decision_matrix[1])]))
    errors = decision_matrix[1] - decision_matrix[0]
    print("Phone with highest number of errors {}".format(phones[np.argmax(errors)]))
    print("Overpredictions:")
    print(decision_matrix[2])
    print("Phone with highest number of overpredictions: {}".format(phones[np.argmax(decision_matrix[2])]))
    print(confusion_matrix)
    print(confusion_matrix[:,0])
    print(np.sum(confusion_matrix[:,0]))
    total = np.sum(confusion_matrix, axis=0)
    print(total)
    accuracies = confusion_matrix.copy().astype(float)
    total_correct = 0
    for i in range(accuracies.shape[0]):
        accuracies[:,i] = accuracies[:,i] / total[i]
        print(phones[i])
        print(accuracies[i,i])
        total_correct += confusion_matrix[i,i]
    print("Accuracy: {}".format(total_correct / np.sum(total)))
    disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=phones)
    disp.text_ = None
    disp.plot(include_values=False)
    plt.show()
