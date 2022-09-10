import scipy.io
import numpy as np
from threading import Thread
from random import randint


labels = []

def readFromMatlab(fileName: str):
    return scipy.io.loadmat(fileName)


def readLabels():
    global labels
    if len(labels) == 0:
        labels = readFromMatlab('Preprocessed_EEG/label.mat')['label'][0]

    return labels

def readEEGSignals(person_number, trial_number, result, index):
    print(f"Extracting data for person number {person_number}, trial number {trial_number} ...")
    x_train = readFromMatlab(f'Preprocessed_EEG/{person_number}_{trial_number}.mat')
    y_train = readLabels()

    result_x = []
    result_y = []
    for i in range(1, 16):
        x = tryExtractObjectFromXTrain(x_train, i)
        x_length = len(x)
        new_x = []
        for j in range(0, x_length):
            after_apply_window = applySlidingWindow(x[j], 100)
            after_take_sub_array = takeSubArray(after_apply_window, 40)
            new_x.append(after_take_sub_array)
        result_x.append(np.asarray(new_x).flatten())
        result_y.append(y_train[i-1])
    result_y = changeLabelValues(result_y)

    print(f"Finished extracting data for person {person_number}, trial number {trial_number}.")
    result[index] = (np.asarray(result_x), np.asarray(result_y))


def readAllEEGSignals():
    result_x = []
    result_y = []
    threads = [None] * 45
    results = [None] * 45
    thread_counter = 0

    for i in range(1, 16):  # for each person
        for j in range(1, 4):  # for each trial
            threads[thread_counter] = Thread(
                target=readEEGSignals, args=(i, j, results, thread_counter))
            threads[thread_counter].start()
            thread_counter += 1

    for i in range(len(threads)):
        threads[i].join()

    for i in range(0, len(results)):
        (current_result_x, current_result_y) = results[i] # (result_x, result_y)
        result_x.append(current_result_x)
        result_y.append(current_result_y)

    final_result_x = np.asarray(result_x)
    final_result_y = np.asarray(result_y)

    return (np.concatenate(final_result_x, axis=0), np.concatenate(final_result_y, axis=0))


def applySlidingWindow(arr, samples):
    array_length = len(arr)
    step = int(array_length / samples)
    if step == 0:
        return arr
    result = []
    for i in range(0, array_length - 1, step):
        last_index = i + step
        if len(arr[i:]) < step:
            last_index = len(arr) - 1

        max_value = np.average(arr[i:last_index])
        result.append(max_value)

    return np.asarray(result)


def changeLabelValues(labels_arr):
    # positive: 2
    # neutral: 1
    # negative: 0
    return [x+1 for x in labels_arr]


def takeSubArray(array, percentage):
    array_length = len(array)
    samples = int((percentage * array_length) / 100)
    last_index = array_length - 1
    last_sub_array_index = last_index - samples
    left_pointer = randint(0, last_sub_array_index)
    right_pointer = left_pointer + samples
    return array[left_pointer:right_pointer]


def tryExtractObjectFromXTrain(x_train, index):
    try:
        return x_train[f"djc_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"jl_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"jj_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"lqj_eeg{index}"]
    except: 
        pass
    try:
        return x_train[f"ly_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"phl_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"zjy_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"wk_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"ys_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"wsf_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"xyl_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"ww_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"mhw_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"wyw_eeg{index}"]
    except:
        pass
    try:
        return x_train[f"sxy_eeg{index}"]
    except:
        pass

# result = [None] * 15
# index = 0
# readEEGSignals(1, 1, result, index)