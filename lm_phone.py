import requests
import os
import copy
import jiwer
import numpy as np
import jellyfish
import sys
import json
from test_logos import get_out_of_vocab
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
np.set_printoptions(threshold=sys.maxsize)
#retrieve CMU pronunciation dict
# response = requests.get("http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/sphinxdict/cmudict.0.7a_SPHINX_40")
# open("cmudict.0.7a_SPHINX_40", "wb").write(response.content)
#manually read in csv; the formatting of the file breaks pd.read_csv
dict = {}
phones = []
VOWELS = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
CONSONANTS = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'ER', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

with open("cmudict.0.7a_SPHINX_40", "r") as file:
    for line in file:
        entry = line.replace('\n', '').split('\t')
        # print(entry)
        dict[entry[0]] = entry[1]

with open("SphinxPhones_40", 'r') as phone_file:
    for line in phone_file:
        phones.append(line.replace('\n', ''))

phones.append("-")
def transcribe_text(file_path, from_file=False):
    transcript = []
    with open(file_path, "r") as f:
        for line in f:
            transcript = line.upper()
            transcript = transcript.replace("\n", "")
            transcript = transcript.replace(". ", " ")
            transcript = transcript.replace(".", "")
            transcript = transcript.replace(",", "")
            transcript = transcript.replace("'", "")
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

def transcribe_audio(file_path, from_file=False, file_type="txt"):
    transcript = []
    with open(file_path, "r") as f:
        if(file_type == "json"):
            fjson = json.load(f)
            transcript = fjson["text"]
            transcript = transcript.upper()
        else:
            for line in f:
                if(file_type != "json"):
                    transcript = line.upper()
        transcript = transcript.replace("\n", "")
        transcript = transcript.replace(". ", " ")
        transcript = transcript.replace(".", "")
        transcript = transcript.replace(",", "")
        transcript = transcript.replace(";", "")
        transcript = transcript.replace("'", "")
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
            if(response[0] == "WIFI"):
                dict[response[0]] = response[2]
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

def needleman_wunsch(actual_phones, pred_phones, matrix, phone_dict, verbose=False):
    MATCH_SCORE = 10
    MISMATCH_SCORE = -1
    GAP_SCORE= -2
    MISMATCH_VOWEL_CONS_SCORE = -100 #avoid confusing consonants for vowels
    print(actual_phones)
    print(pred_phones)
    grid = np.zeros((len(pred_phones)+1, len(actual_phones)+1), dtype=int)
    op_grid = np.zeros((len(pred_phones)+1, len(actual_phones)+1))
    for i in range(1, grid.shape[1]):
        grid[0,i] = grid[0,i-1] + GAP_SCORE
    for j in range(1, grid.shape[0]):
        grid[j,0] = grid[j-1,0] + GAP_SCORE
        pass
    for i in range(1, grid.shape[0]):
        for j in range(1, grid.shape[1]):
            top_left = grid[i-1,j-1].copy()
            top = grid[i-1,j].copy()+GAP_SCORE
            left = grid[i,j-1].copy()+GAP_SCORE
            if(pred_phones[i-1] == actual_phones[j-1]):
                top_left+=MATCH_SCORE
            else:
                if(pred_phones[i-1] in CONSONANTS and actual_phones[j-1] in VOWELS):
                    top_left+= MISMATCH_VOWEL_CONS_SCORE
                elif(pred_phones[i-1] in VOWELS and actual_phones[j-1] in CONSONANTS):
                    top_left+=MISMATCH_VOWEL_CONS_SCORE
                else:
                    top_left+=MISMATCH_SCORE
            array = [top_left, top, left]
            grid[i,j] = array[np.argmax(array)]
    #alignment 
    alignment_a = []
    alignment_b = []
    i = len(pred_phones)
    j = len(actual_phones)
    while(i > 0 or j > 0):
        if(i > 0 and j > 0 and grid[i,j] == grid[i-1,j-1] + MATCH_SCORE and pred_phones[i-1] == actual_phones[j-1]):
            alignment_a.insert(0, pred_phones[i-1])
            alignment_b.insert(0, actual_phones[j-1])
            i = i-1
            j = j-1
        elif(j > 0 and grid[i,j] == grid[i,j-1] + GAP_SCORE):
            alignment_a.insert(0, "-")
            alignment_b.insert(0, actual_phones[j-1])
            j = j-1
        elif(i > 0 and j > 0 and grid[i,j] == grid[i-1,j-1] + MISMATCH_SCORE and
            ((pred_phones[i-1] in CONSONANTS and actual_phones[j-1] in CONSONANTS) or 
            (pred_phones[i-1] in VOWELS and actual_phones[j-1] in VOWELS))):
                alignment_a.insert(0, pred_phones[i-1])
                alignment_b.insert(0, actual_phones[j-1])
                i = i-1
                j = j-1
        else:
            alignment_a.insert(0, pred_phones[i-1])
            alignment_b.insert(0, "-")
            i = i-1
    print("ALIGNMENT")
    print(alignment_a)
    print(alignment_b)
    #verify vowels are not aligned with things
    for i in range(len(alignment_a)):
        if(alignment_a[i] != alignment_b[i]):
            if(alignment_a[i] in CONSONANTS and alignment_b[i] in VOWELS):
                np.savetxt('grid.csv', grid, fmt='%d', delimiter=',')
                raise NotImplementedError
            elif(alignment_b[i] in CONSONANTS and alignment_a[i] in VOWELS):
                np.savetxt('grid.csv', grid, fmt='%d', delimiter=',')
                raise NotImplementedError
    #find most common sources of error
    errors_a = []
    errors_b = []
    for i in range(len(alignment_a)):
        if(alignment_a[i] != alignment_b[i]):
            error_a = ""
            error_b = ""
            if(i == 0 or alignment_a[i-1] == alignment_b[i-1]):
                if(i!=0):
                    error_a += alignment_a[i-1]
                    error_b += alignment_b[i-1]
                if(alignment_a[i] != '-'):
                    error_a += " " + alignment_a[i]
                if(alignment_b[i] != '-'):
                    error_b += " " + alignment_b[i]
                while(alignment_a[i] != alignment_b[i] and i < len(alignment_a)-1):
                    i+=1
                    if(alignment_a[i] != '-'):
                        error_a += " " + alignment_a[i]
                    if(alignment_b[i] != '-'):
                        error_b += " " + alignment_b[i]
            errors_a.append(error_a)
            errors_b.append(error_b)
    print(errors_a)
    print(errors_b)
    for i in range(len(alignment_a)):
        predicted = alignment_a[i]
        actual = alignment_b[i]
        confusion_matrix[phone_dict[actual]][phone_dict[predicted]] += 1
    return errors_a, errors_b
if __name__ == '__main__':
    phones_dict = {}
    for i in range(len(phones)):
        phones_dict[phones[i]] = i
    confusion_matrix = np.zeros((len(phones), len(phones)), dtype=int) #table for each respective phone
    #input settings are "word" and "corpus"
    input_setting = "needleman_wunsch"
    if(input_setting == "corpus"):
        actual_transcriptions = []
        predicted_transcriptions = []
        actual_trans = []
        predicted_trans = []

        in_dir = "output_gaussian_0_100"
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
        print(error)
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
        plt.title("Non-noisy Deepspeech output")
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
    elif(input_setting == "needleman_wunsch"):
        error_dict = {}
        actual_transcriptions = []
        predicted_transcriptions = []
        actual_trans = []
        predicted_trans = []
        dir_type = "deepspeech"
        in_dir = "output_vosk_superimpose_1"
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
                        print(true_trans_path)
                        if(dir_type == "vosk"):
                            pred_trans_path= os.path.join(in_dir, dir+"/"+sub_dir+"/"+name_base+".WAV.json")
                        else:
                            pred_trans_path= os.path.join(in_dir, dir+"/"+sub_dir+"/"+name_base+".WAV.txt")
                        # word_actual = transcribe_text(true_trans_path, from_file=True)
                        # word_pred = transcribe_text(pred_trans_path)
                        # print(word_actual)
                        # print(word_pred)
                        true_transcription = transcribe_audio(true_trans_path, from_file=True)
                        pred_transcription = transcribe_audio(pred_trans_path, file_type="json")
                        if(len(pred_transcription) != 0):
                            predicted_transcriptions.append(' '.join(pred_transcription))
                            predicted_trans.append(pred_transcription)
                            actual_transcriptions.append(' '.join(true_transcription))
                            actual_trans.append(true_transcription)
                        else:
                            print(true_trans_path)
                        error_a, error_b = needleman_wunsch(true_transcription, pred_transcription, confusion_matrix, phones_dict)
                        for i in range(len(error_b)):
                            if error_b[i] != '':
                                if(error_b[i] not in error_dict):
                                    error_dict[error_b[i]] = {"total": 0}
                                if(error_a[i]) not in error_dict[error_b[i]]:
                                    error_dict[error_b[i]][error_a[i]] = 0
                                error_dict[error_b[i]][error_a[i]] += 1
                                error_dict[error_b[i]]["total"] += 1
        #reshape the data structure
        mistakes = []
        for k, v in error_dict.items():
            for key, value in v.items():
                if(key == "total"):
                    continue
                mistake = [k, key, value]
                mistakes.append(mistake)
        print(mistakes)
        np.savetxt("mistakes_{}.csv".format(in_dir), np.array(mistakes), delimiter=',', fmt='%s')
        print(sorted(mistakes, key=lambda x:x[2], reverse=True))
        total = np.sum(confusion_matrix, axis=1)
        print(total)
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
        print("Overall phone accuracy: {}".format(total_correct / np.sum(total)))
        print("Phone that was added the most often: {}".format(phones[np.argmax(confusion_matrix[-1,:])]))
        print(np.max(confusion_matrix[-1,:]))
        print("Phone that was deleted the most often: {}".format(phones[np.argmax(decision_matrix[2,:])]))
        np.savetxt('phones.csv', accuracy_per_phone, delimiter=',')
        disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=phones)
        disp.text_ = None
        disp.plot(include_values=False)
        plt.xticks(rotation=90)
        plt.title("Non-noisy Deepspeech output")
        plt.show()
        np.savetxt('{}_confusion.csv'.format(in_dir), confusion_matrix, delimiter=',', fmt='%d')



