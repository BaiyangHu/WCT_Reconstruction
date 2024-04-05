import ppg_ecg_system as preprocessing
import original_paper.paper as paper
import scipy.io
from scipy import signal
import sys
import numpy as np
from tensorflow.keras.regularizers import L1L2

# Checks that output from filtering ECG and PPG signal same as that in paper
def test_high_low_pass(data):
    for index in range(data.size):
        ecg = data[index, 0]['ecg_II'][:, 0]
        ppg = data[index, 0]['ppg'][:, 0]

        ecg_expected, ppg_expected, _, _, _, _, _, _, _, _, _, _, _, _, _ = paper.signal_preprocessing(ecg, ppg, 125, 20, 10, 4)
        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 10 / (125 / 2)], "bandpass")
        ppg_actual = signal.filtfilt(b, a, ppg)
        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 20 / (125 / 2)], "bandpass")
        ecg_actual = signal.filtfilt(b, a, ecg)

        np.testing.assert_array_equal(ecg_expected, ecg_actual, "test_high_low_pass failed!")
        np.testing.assert_array_equal(ppg_expected, ppg_actual, "test_high_low_pass failed!")
    print("test_high_low_pass passed!")

# Checks that output from finding ECG and PPG peaks same as that in paper
def test_peaks(data):
    for index in range(data.size):
        ecg = data[index, 0]['ecg_II'][:, 0]
        ppg = data[index, 0]['ppg'][:, 0]

        _, _, r_peaks_expected, s_peaks_expected, _, _, _, _, _, _, _, _, _, _, _ = paper.signal_preprocessing(ecg, ppg, 125, 20, 10, 4)

        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 10 / (125 / 2)], "bandpass")
        ppg_actual = signal.filtfilt(b, a, ppg)
        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 20 / (125 / 2)], "bandpass")
        ecg_actual = signal.filtfilt(b, a, ecg)
        r_peaks_actual = preprocessing.get_r_peaks(ecg, ecg_actual)
        s_peaks_actual = preprocessing.get_systolic_peaks(ppg, ppg_actual)

        np.testing.assert_array_equal(r_peaks_actual, r_peaks_expected, "test_peaks failed!")
        np.testing.assert_array_equal(s_peaks_actual, s_peaks_expected, "test_peaks failed!")
    print("test_peaks passed!")

# Checks that output from aligned ECG and PPG peaks same as that in paper
def test_alignment(data):
    for index in range(data.size):
        ecg = data[index, 0]['ecg_II'][:, 0]
        ppg = data[index, 0]['ppg'][:, 0]

        _, _, _, _, expected_ecg_aligned, expected_ppg_aligned, _, _, _, _, _, _, _, _, _ = paper.signal_preprocessing(ecg, ppg, 125, 20, 10, 4)

        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 10 / (125 / 2)], "bandpass")
        ppg_actual = signal.filtfilt(b, a, ppg)
        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 20 / (125 / 2)], "bandpass")
        ecg_actual = signal.filtfilt(b, a, ecg)
        r_peaks = preprocessing.get_r_peaks(ecg, ecg_actual)
        s_peaks = preprocessing.get_systolic_peaks(ppg, ppg_actual)
        actual_ppg_aligned, actual_ecg_aligned = preprocessing.alignment(ppg_actual, s_peaks, ecg_actual, r_peaks)

        np.testing.assert_array_equal(actual_ppg_aligned, expected_ppg_aligned, "test_alignment {} failed!".format(index))
        np.testing.assert_array_equal(actual_ecg_aligned, expected_ecg_aligned, "test_alignment failed!")
    print("test_alignment passed!")

# Testing if normalization of PPG signal is the same as that in paper
def test_normalization(data):
    for index in range(data.size):
        ecg = data[index, 0]['ecg_II'][:, 0]
        ppg = data[index, 0]['ppg'][:, 0]

        _, _, _, _, _, _,  ppg_normalized, _, _, _, _, _, _, _, _ = paper.signal_preprocessing(ecg, ppg, 125, 20, 10, 4)
        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 10 / (125 / 2)], "bandpass")
        ppg_actual = signal.filtfilt(b, a, ppg)
        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 20 / (125 / 2)], "bandpass")
        ecg_actual = signal.filtfilt(b, a, ecg)
        r_peaks = preprocessing.get_r_peaks(ecg, ecg_actual)
        s_peaks = preprocessing.get_systolic_peaks(ppg, ppg_actual)
        actual_ppg_aligned, actual_ecg_aligned = preprocessing.alignment(ppg_actual, s_peaks, ecg_actual, r_peaks)
       
        actual_ppg_normalized = preprocessing.normalise(actual_ppg_aligned)

        np.testing.assert_array_almost_equal(actual_ppg_normalized, ppg_normalized, 5, "test normalization failed!")
    
    print("test normalization passed!")


# Testing if segmentation of PPG and ECG signals is same as in paper
def test_segmentation(data):
    for index in range(data.size):
        ecg = data[index, 0]['ecg_II'][:, 0]
        ppg = data[index, 0]['ppg'][:, 0]

        _, _, _, _, _, _, _, ecg_final, ppg_final, _, _, _, _, _, _ = paper.signal_preprocessing(ecg, ppg, 125, 20, 10, 4)

        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 10 / (125 / 2)], "bandpass")
        ppg_actual = signal.filtfilt(b, a, ppg)
        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 20 / (125 / 2)], "bandpass")
        ecg_actual = signal.filtfilt(b, a, ecg)
        r_peaks = preprocessing.get_r_peaks(ecg, ecg_actual)
        s_peaks = preprocessing.get_systolic_peaks(ppg, ppg_actual)
        actual_ppg_aligned, actual_ecg_aligned = preprocessing.alignment(ppg_actual, s_peaks, ecg_actual, r_peaks)

        actual_ppg_normalized = preprocessing.normalise(actual_ppg_aligned)

        actual_final_ppg = preprocessing.segmentation(actual_ppg_normalized, 4)
        actual_ecg_final = preprocessing.segmentation(actual_ecg_aligned, 4)

        np.testing.assert_array_almost_equal(actual_ecg_final, ecg_final, 5, "test segmentation failed!")
        np.testing.assert_array_almost_equal(actual_final_ppg, ppg_final, 5, "test segmentation failed!")
    print("test segmentation passed!")

# Testing if splitting of PPG and ECG signals is same as in paper
def test_splitting(data):
    for index in range(data.size):
        ecg = data[index, 0]['ecg_II'][:, 0]
        ppg = data[index, 0]['ppg'][:, 0]

        _, _, _, _, _, _, _, _, _, train_ppg, train_ecg, validation_ppg, validation_ecg, test_ppg, test_ecg = paper.signal_preprocessing(ecg, ppg, 125, 20, 10, 4)

        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 10 / (125 / 2)], "bandpass")
        ppg_actual = signal.filtfilt(b, a, ppg)
        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 20 / (125 / 2)], "bandpass")
        ecg_actual = signal.filtfilt(b, a, ecg)
        r_peaks = preprocessing.get_r_peaks(ecg, ecg_actual)
        s_peaks = preprocessing.get_systolic_peaks(ppg, ppg_actual)
        actual_ppg_aligned, actual_ecg_aligned = preprocessing.alignment(ppg_actual, s_peaks, ecg_actual, r_peaks)

        actual_ppg_normalized = preprocessing.normalise(actual_ppg_aligned)

        actual_train_ecg, actual_validation_ecg, actual_test_ecg = preprocessing.data_splitting(actual_ecg_aligned)
        actual_train_ppg, actual_validation_ppg, actual_test_ppg = preprocessing.data_splitting(actual_ppg_normalized)

        actual_train_ecg = preprocessing.fit_format(preprocessing.segmentation(actual_train_ecg, 4))
        actual_validation_ecg = preprocessing.fit_format(preprocessing.segmentation(actual_validation_ecg, 4))
        actual_test_ecg = preprocessing.fit_format(preprocessing.segmentation(actual_test_ecg, 4))
        actual_train_ppg = preprocessing.fit_format(preprocessing.segmentation(actual_train_ppg, 4))
        actual_validation_ppg = preprocessing.fit_format(preprocessing.segmentation(actual_validation_ppg, 4))
        actual_test_ppg = preprocessing.fit_format(preprocessing.segmentation(actual_test_ppg, 4))

        np.testing.assert_array_almost_equal(actual_train_ecg, train_ecg, 5, "test splitting failed!")
        np.testing.assert_array_almost_equal(actual_train_ppg, train_ppg, 5, "test splitting failed!")
        np.testing.assert_array_almost_equal(actual_validation_ecg, validation_ecg, 5, "test splitting failed!")
        np.testing.assert_array_almost_equal(actual_validation_ppg, validation_ppg, 5, "test splitting failed!")
        np.testing.assert_array_almost_equal(actual_test_ecg, test_ecg, 5, "test splitting failed!")
        np.testing.assert_array_almost_equal(actual_test_ppg, test_ppg, 5, "test splitting failed!")
    print("test splitting passed!")

# Testing if our model output on the test ppg data produces the same output as the paper
def test_model_output(data):
    for index in range(5): # Only test first 5 patients due to time
        ecg = data[index, 0]['ecg_II'][:, 0]
        ppg = data[index, 0]['ppg'][:, 0]

        _, _, _, _, _, _, _, _, _, train_ppg, train_ecg, validation_ppg, validation_ecg, test_ppg, test_ecg = paper.signal_preprocessing(ecg, ppg, 125, 20, 10, 4)
        regularizer = L1L2(l1=0.0001, l2=0.0001)
        lstm_model, history = paper.fit_model(train_ppg, train_ecg, 1, 1, 25, regularizer)
        expected_test_ecg = paper.get_test_ecg(lstm_model, test_ppg)

        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 10 / (125 / 2)], "bandpass")
        ppg_actual = signal.filtfilt(b, a, ppg)
        b, a = signal.cheby2(4, 20, [0.5 / (125 / 2), 20 / (125 / 2)], "bandpass")
        ecg_actual = signal.filtfilt(b, a, ecg)
        r_peaks = preprocessing.get_r_peaks(ecg, ecg_actual)
        s_peaks = preprocessing.get_systolic_peaks(ppg, ppg_actual)
        actual_ppg_aligned, actual_ecg_aligned = preprocessing.alignment(ppg_actual, s_peaks, ecg_actual, r_peaks)

        actual_ppg_normalized = preprocessing.normalise(actual_ppg_aligned)

        actual_train_ecg, actual_validation_ecg, actual_test_ecg = preprocessing.data_splitting(actual_ecg_aligned)
        actual_train_ppg, actual_validation_ppg, actual_test_ppg = preprocessing.data_splitting(actual_ppg_normalized)

        actual_train_ecg = preprocessing.fit_format(preprocessing.segmentation(actual_train_ecg, 4))
        actual_validation_ecg = preprocessing.fit_format(preprocessing.segmentation(actual_validation_ecg, 4))
        actual_test_ecg = preprocessing.fit_format(preprocessing.segmentation(actual_test_ecg, 4))
        actual_train_ppg = preprocessing.fit_format(preprocessing.segmentation(actual_train_ppg, 4))
        actual_validation_ppg = preprocessing.fit_format(preprocessing.segmentation(actual_validation_ppg, 4))
        actual_test_ppg = preprocessing.fit_format(preprocessing.segmentation(actual_test_ppg, 4))

        model = preprocessing.model_construct(actual_train_ppg, actual_train_ecg, 25, 1, 1)
        prediction = model.predict(actual_test_ppg, batch_size = 1)

        np.testing.assert_array_almost_equal(prediction, expected_test_ecg, 5, "test model failed!")
    print("test model passed!")


if __name__ == "__main__":
    # Load in the ppg and ecg data for all patients
    data = scipy.io.loadmat("original_paper/Records.mat")["records"]
    test_high_low_pass(data)
    test_peaks(data)
    test_alignment(data)
    test_normalization(data)
    test_segmentation(data)
    test_splitting(data)
    test_model_output(data)