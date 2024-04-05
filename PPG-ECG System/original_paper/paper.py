import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICE"] = "1"
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import preprocessing
from scipy import signal
from ecgdetectors import Detectors
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
from tensorflow.keras.regularizers import L1L2
from sklearn.metrics import mean_squared_error
import time
if __name__ == "__main__":
    import two_average_detector
else:
    import original_paper.two_average_detector as two_average_detector


detectors = Detectors(125)

def correctPeaks(peak, ECG, window):
    peak_c = peak
    for index in range(len(peak)):
        start_loc = max(1, peak[index]-window)
        end_loc = min(len(ECG), peak[index]+window)
        segment = ECG[start_loc:end_loc]
        loc = np.argmax(segment)
        peak_c[index] = loc + start_loc
    return peak_c


def alignment(ECG, r_peak, ppg, speak, Fs):
    # go through the 3rd sys-peak onwards
    for index in range(2, len(speak)):
        flag = 0
        previouspeak = [x for x in r_peak if x < speak[index]] # get the r_peaks which are before this s_peak
        for i2 in range(len(previouspeak)-1, 0, -1): # go through previous peaks in reverse order
            rrinterval = r_peak[i2 + 1] - r_peak[i2]
            ppinterval = speak[index + 1] - speak[index]
            if abs(ppinterval - rrinterval) <= 0.05 * Fs: # weird criterion
                #print("no beer i be {} is j is {}".format(index, i2))
                n = i2
                flag = 1
                break
        if flag == 1:
            break

    shiftpoint = speak[index] - r_peak[i2]
    ecg_algined = ECG[1:len(ECG)-shiftpoint]
    ppg_aligned = ppg[shiftpoint+1:len(ppg)]
    return ecg_algined, ppg_aligned


def segment(ecg, ppg, fs, segment_len):
    ecg = np.reshape(ecg, (-1, segment_len*fs))
    ppg = np.reshape(ppg, (-1, segment_len*fs))
    return ecg, ppg


def fit_model(x, y, n_batch, n_epoch, n_neurons, regularizer):
    # model: Bilstm + dense layer
    tf.random.set_seed(1234)
    model = tf.keras.Sequential()
    x = x.reshape(x.shape[0], x.shape[1], 1)
    model.add(layers.Bidirectional(layers.LSTM(n_neurons,
                                               return_sequences=True,
                                               input_shape=(x.shape[1], x.shape[2]),
                                               kernel_regularizer=regularizer)))
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    # model.summary()
    history = model.fit(x, y, epochs=n_epoch, batch_size=n_batch, verbose=0, shuffle=False)
    return model, history


def plot_result(raw, predict, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    t = np.linspace(0,(len(raw)-1)/125,num=len(raw))
    ax1.plot(t, raw, 'k')
    ax1.plot(t, predict, 'r')
    ax1.set_title(title)


def signal_preprocessing(ecg, ppg, Fs, ecg_fh, ppg_fh, segment_len):
    Fs = 125
    signal_len = 62
    segment_len = 4
    ecg_fh = 20
    ppg_fh = 10
    index = 2
    fl = 0.5
    fn = Fs / 2

    b, a = signal.cheby2(4, 20, [fl / fn, ecg_fh / fn], 'bandpass')
    ecg_filtered = signal.filtfilt(b, a, ecg)

    r_peak = detectors.pan_tompkins_detector(ecg)
    r_peak = correctPeaks(r_peak, ecg_filtered, 15)
    b, a = signal.cheby2(4, 20, [fl / fn, ppg_fh / fn], 'bandpass')
    ppg_filtered = signal.filtfilt(b, a, ppg)

    systolicPeak = two_average_detector.extract(ppg, Fs)
    systolicPeak = correctPeaks(systolicPeak, ppg_filtered, 15)
    # align the ECG and PPG based on the third systolic peak and the corresponding R peak
    ecg_algined_1, ppg_aligned_1 = alignment(ecg_filtered, r_peak, ppg_filtered, systolicPeak, Fs)
    # PPG min-max scaling
    minmax_scale = preprocessing.MinMaxScaler()
    ppg_normalized = minmax_scale.fit_transform(ppg_aligned_1.reshape(-1, 1))
    ecg_algined = ecg_algined_1[0:Fs * 288]
    ppg_aligned = ppg_normalized[0:Fs * 288]
    ecg1, ppg1 = segment(ecg_algined, ppg_aligned, Fs, segment_len)

    trainSet_index = range(round(48 / segment_len))
    validation_index = range(round(48 / segment_len), round(60 / segment_len))
    test_index = range(round(60 / segment_len), ppg1.shape[0])
    train_ppg = fit_format(ppg1[trainSet_index, :])
    train_ecg = fit_format(ecg1[trainSet_index, :])
    validation_ppg = fit_format(ppg1[validation_index, :])
    validation_ecg = fit_format(ecg1[validation_index, :])
    test_ppg = fit_format(ppg1[test_index, :])
    test_ecg = fit_format(ecg1[test_index, :])

    return ecg_filtered, ppg_filtered, r_peak, systolicPeak, ecg_algined_1, ppg_aligned_1, ppg_normalized, ecg1, ppg1, train_ppg, train_ecg, validation_ppg, validation_ecg, test_ppg, test_ecg


def run_model(ppg, ecg, regularizer):
    print(', start traing>> ', end='')
    # Trainset: 80%  Test: 20%
    trainSet_num = round(ppg.shape[0] * 0.8)
    testSet_num = ppg.shape[0] - trainSet_num
    lstm_model, history = fit_model(ppg[0:ppg.shape[0]-testSet_num, :], ecg[0:ppg.shape[0]-testSet_num, :], 1, 1000, 25, regularizer)
    test_ppg = ppg
    test_ppg = test_ppg.reshape(test_ppg.shape[0], test_ppg.shape[1], 1)
    output = lstm_model.predict(test_ppg, batch_size=1)
    # plot_result(ecg1, output, 0)
    train_result, test_result = cal_score(ecg, output, range(trainSet_num), range(trainSet_num, ppg.shape[0]))
    return train_result, test_result


def fit_format(data):
    data1 = np.array(data)
    result = data1.reshape(data1.shape[0], data1.shape[1], 1)
    return result

def get_test_ecg(model, test_ppg):
    test_result = model.predict(test_ppg, batch_size=1)
    return test_result

if __name__ == '__main__':
    # section: configuring tensorflow options
    # subsection a: run on 12 core machine
    num_threads = 12
    os.environ["OMP_NUM_THREADS"] = "12"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "12"
    os.environ["TF_NUM_INTEROP_THREADS"] = "12"
    tf.config.threading.set_inter_op_parallelism_threads(
        num_threads
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        num_threads
    )
    tf.config.set_soft_device_placement(True)

    # subsection b: ensure gpu is enabled
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print()

    plt.close('all')
    data = scio.loadmat('original_paper/Records.mat')
    records = data['records']

    Fs = 125
    signal_len = 62
    segment_len = 4
    ecg_fh = 20
    ppg_fh = 10
    index = 2
    detectors = Detectors(Fs)
    for segment_len in range(1, 5):
        result = {'test_ecg': [], 'test_ppg': [], 'test_result': [], 'validation_ecg': [], 'validation_ppg': [],
                  'validation_result': []}
        result2 = {'train_ecg': [], 'train_ppg': [], 'train_result': []}
        for index in range(records.size):
            time_start = time.time()
            ecg = records[index, 0]['ecg_II'][:, 0]
            ppg = records[index, 0]['ppg'][:, 0]
            _, _, _, _, _, _, _, ecg_fixed, ppg_fixed, train_ppg, train_ecg, validation_ppg, validation_ecg, test_ppg, test_ecg = signal_preprocessing(ecg, ppg, Fs, ecg_fh, ppg_fh, segment_len)

            regularizer = L1L2(l1=0.0001, l2=0.0001)
            lstm_model, history = fit_model(train_ppg, train_ecg, 1, 1000, 25, regularizer)

            validation_result = lstm_model.predict(validation_ppg, batch_size=1)
            test_result = lstm_model.predict(test_ppg, batch_size=1)
            result['test_ecg'].append(test_ecg)
            result['test_ppg'].append(test_ppg)
            result['test_result'].append(test_result)
            result['validation_ecg'].append(validation_ecg)
            result['validation_ppg'].append(validation_ppg)
            result['validation_result'].append(validation_result)
            lstm_model.save('Results/Models/' + str(segment_len) + 's/Model' + str(index) + '.h5')
            scio.savemat('Results/result_' + str(segment_len) + 's_intra_L1_0001_L2_0001.mat', {'result': result})

            train_result = lstm_model.predict(train_ppg, batch_size=1)
            result2['train_ecg'].append(train_ecg)
            result2['train_ppg'].append(train_ppg)
            result2['train_result'].append(train_result)
            scio.savemat('Results/result_train' + str(segment_len) + 's_intra_L1_0001_L2_0001.mat', {'result': result2})