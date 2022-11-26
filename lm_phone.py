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
            if 'IX' in transcribe:
                transcribe.remove('IX')
            phonetic += transcribe
    return phonetic

def align_phones(actual_phones, pred_phones):
    if(len(actual_phones) == 0 and len(pred_phones) == 0):
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

def transcribe(actual_phones, pred_phones, matrix, phone_dict, verbose=False):
    #CORE ASSUMPTION #1: Words only appear once in each phrase
    #This is a bad assumption but one that can be remediated
    alignment = align_phones(actual_phones, pred_phones)
    if(verbose):
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

if __name__ == '__main__':
    phones_dict = {}
    for i in range(len(phones)):
        phones_dict[phones[i]] = i
    confusion_matrix = np.zeros((len(phones), len(phones)), dtype=int) #table for each respective phone
    #input settings are "word" and "corpus"
    input_setting = "corpus"
    if(input_setting == "corpus"):
        actual_transcriptions = []
        predicted_transcriptions = []
        actual_trans = []
        predicted_trans = []

        in_dir = "output_gaussian_0_5000"
        text_dir = "archive/data/TEST"
        s_list = os.listdir(text_dir)

        decision_matrix = np.zeros((3, len(phones)))
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
                        pred_trans_path= os.path.join(in_dir, dir+"/"+sub_dir+"/"+name_base+".WAV.txt")
                        word_actual = transcribe_text(true_trans_path, from_file=True)
                        word_pred = transcribe_text(pred_trans_path)
                        true_transcription = transcribe_audio(true_trans_path, from_file=True)
                        pred_transcription = transcribe_audio(pred_trans_path)
                        if(len(pred_transcription) != 0):
                            predicted_transcriptions.append(' '.join(pred_transcription))
                            predicted_trans.append(pred_transcription)
                            actual_transcriptions.append(' '.join(true_transcription))
                            actual_trans.append(true_transcription)
                        else:
                            print(true_trans_path)
                        transcribe(true_transcription, pred_transcription, confusion_matrix, phones_dict)
                        index = 0
                        max_align_dist = 1 #assume that transcriptions are off by only one or two phonemes
                        previous_error_rate = 0

        error = jiwer.compute_measures(predicted_transcriptions, actual_transcriptions)
        print("Phone error rate: {}".format(error['wer'])) #treat each phone in the transcription as a word
        total = np.sum(confusion_matrix, axis=0)
        accuracy_per_phone = np.zeros(len(phones)-1)
        accuracies = confusion_matrix.copy().astype(float)
        total_correct = 0
        for i in range(accuracies.shape[0]-1):
            accuracies[:,i] = accuracies[:,i] / total[i]
            total_correct += confusion_matrix[i,i]
            decision_matrix[0,i] = confusion_matrix[i,i]
            decision_matrix[1,i] = total[i].copy()
            decision_matrix[2,i] = confusion_matrix[i,-1]
            accuracy_per_phone[i] = accuracies[i,i].copy()

        print("Phone with highest accuracy is {} with accuracy of {}".format(phones[np.argmax(accuracy_per_phone)], np.max(accuracy_per_phone)))
        print("Phone with lowest accuracy is {} with accuracy of {}".format(phones[np.argmin(accuracy_per_phone)], np.min(accuracy_per_phone)))
        print("Phone with lowest number of appearances in data: {}".format(phones[np.argmin(total)]))
        errors = decision_matrix[1,:] - decision_matrix[0,:]
        print("Phone with highest number of errors: {}".format(phones[np.argmax(errors)]))
        print("Phone that was added the most often: {}".format(phones[np.argmax(confusion_matrix[-1,:])]))
        print("Phone that was deleted the most often: {}".format(phones[np.argmax(decision_matrix[2,:])]))
        print(phones)
        print(accuracy_per_phone)
        print("Overall phone accuracy: {}".format(total_correct / np.sum(total)))
        disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=phones)
        disp.text_ = None
        disp.plot(include_values=False)
        plt.xticks(rotation=90)
        plt.show()

    elif(input_setting == "word"):

        TRUE_TEXT_PATH = "archive/data/TEST/DR1/FAKS0/SA1.TXT" #path to true transcription
        MODEL_TEXT_PATH = "output_min_suppression_2000\DR1\FAKS0\SA1.WAV.txt" #path to the model's outputted transcription
        word_actual = transcribe_text(TRUE_TEXT_PATH, from_file=True)
        word_pred = transcribe_text(MODEL_TEXT_PATH)
        print(word_actual)
        print(word_pred)
        true_transcription = transcribe_audio(TRUE_TEXT_PATH, from_file=True)
        pred_transcription = transcribe_audio(MODEL_TEXT_PATH)
        print(true_transcription)
        print(pred_transcription)
        transcribe(true_transcription, pred_transcription, confusion_matrix, phones_dict, verbose=True)
