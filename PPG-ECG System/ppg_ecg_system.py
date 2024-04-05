import scipy.io
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import original_paper.two_average_detector as two_average_detector
from ecgdetectors import Detectors
import tensorflow as tf
import sys

Fs = 125 # This is the sampling rate of dataset - it is fixed
detectors = Detectors(Fs)

def unpack_list(ls):
    """
    Helper function to help process data loaded in from matlab. Turns a list of single element
    lists into a list which simply just contains the elements without additional nestnig.

    Input: A list, where every element of the list is a single element list [[a], [b], ...]
    Output: A 1D list, where every element of the list is a single value [a, b, ...]
    """

    return_ls = []

    for elem in ls:
        return_ls.append(elem[0])

    return return_ls
    
def alignment(ppg_cleaned, s_peaks, ecg_cleaned, r_peaks):
    """
    Takes in a PPG and ECG signal and returns both signals so that they are aligned such that the 3rd systolic
    peak in the PPG signal is aligned with the corresponding peak in the ECG signal. 

    Input: Cleaned PPG signal, and Corresponding S-Peaks, Cleaned ECG signal, and Corresponding R-Peaks
    Output: Aligned PPG signal, Aligned ECG signal
    """

    i = 2
    flag = False
    # Section 1: Iterate through the S peaks, starting from the third
    while i < len(s_peaks):
        current_peak_time = s_peaks[i]

        # Subsection 1a: For the current S peak, look at all the R peaks that occured before it
        earlier_r_peaks = []
        for r_peak_time in r_peaks:
            if r_peak_time < current_peak_time:
                earlier_r_peaks.append(r_peak_time)

        s_interval = s_peaks[i + 1] - s_peaks[i] # Calculate the interval between current S peak and the one before

        j = len(earlier_r_peaks) - 1
        # Subsection 1b: Go through all the previous intervals between previous consecutive R peaks. If the previous
        # interval is similar in length to current S peak interval, break from both loops
        while j > 0:
            r_interval = r_peaks[j + 1] - r_peaks[j]
            # The criterion used to determine whether the intervals are similar in length is whether their lengths 
            # within 0.05 seconds of each other
            if abs(s_interval - r_interval) <= (0.05 * Fs):
                flag = True
                break
            j -= 1
        
        if flag:
            break

        i += 1

    # Section 2: Shift the PPG signal forward so that the peaks found in the loop are aligned
    time_difference = s_peaks[i] - r_peaks[j] # Represents the time difference between the s peak and corresponding r peak
    ppg_aligned = ppg_cleaned[time_difference + 1 :] # Effectively shifting forward PPG forward by time_difference
    ecg_aligned = ecg_cleaned[1: len(ecg_cleaned) - time_difference] # Truncating ECG signal so that it's the same length as PPG signal

    return (ppg_aligned, ecg_aligned)

def get_exact_peaks(approximate_peaks, cleaned):
    """
    Takes in a cleaned ECG/PPG signal as well as a list of approximate peaks, and returns a
    list of values, representing the times of the exact peaks in the cleaned signal.

    Input: list of times of approximate peaks, cleaned ECG/PPG signal
    Output: list of times of exact peaks
    """
    
    WINDOW = 15
    new_peaks = []

    # Iterate through approximate peaks
    for approximate_peak in approximate_peaks:
        # Look at a window of time around the approximate peak, and take highest signal in that region as the new peak
        beginning = approximate_peak - WINDOW
        if beginning < 1:
            beginning = 1

        end = approximate_peak + WINDOW
        if end > len(cleaned):
            end = len(cleaned)

        subsection = cleaned[beginning:end] # Extract window around approximate peak
        new_peak = np.argmax(subsection) + beginning # Get position of highest signal in window
        new_peaks.append(new_peak)

    return new_peaks


def get_r_peaks(original_ecg, cleaned_ecg):
    """
    Takes in a cleaned and original ECG signal and returns a list of the indices, representing
    the times of the r peaks in the cleaned signal.

    Input: Original (non-cleaned) ECG signal, Cleaned ECG signal
    Output: List of values of times in the cleaned signal, where the R peaks occur
    """
    # This is the algorithm specified in the paper for getting approximate R peaks from an ECG signal
    approximate_peaks = detectors.pan_tompkins_detector(original_ecg)

    return get_exact_peaks(approximate_peaks, cleaned_ecg)

def get_systolic_peaks(original_ppg, cleaned_ppg):
    """
    Takes in a cleaned and original PPG signal and returns a list of the indices, representing
    the times of the systolic peaks in the cleaned signal.

    Input: Original (non-cleaned) PPG signal, Cleaned PPG signal
    Output: List of values of times in the cleaned signal, where the Systolic peaks occur
    """
    # This is the algorithm specified in the paper for getting approximate systolic peaks from a PPG signal (block method)
    approximate_peaks = two_average_detector.extract(original_ppg, Fs)

    return get_exact_peaks(approximate_peaks, cleaned_ppg)

def normalise(arr):
    """
    Takes in a signal and returns an array respersenting the normalised signal, normalised from [0, 1].

    Input: A signal
    Output: The same signal, normalised from 0 to 1
    """
    normalised_arr = []
    max_val = np.amax(arr)
    min_val = np.amin(arr)

    # Iterate through signal
    for val in arr:
        # Calculation ensures value lies from [0, 1], normalising the current value
        normalised_arr.append((val - min_val)/(max_val - min_val)) 

    normalised_arr = np.array(normalised_arr)
    return normalised_arr.reshape(-1, 1)


def data_splitting(arr):
    """ 
    Takes in a list representing a signal and returns 3 lists:
    The first list represents the training set, and is to contain the first 48 seconds of the signal
    The second list represents the validation set, and is to contain the next 12 seconds of the signl
    The third list reprsents the test set, and is to contain the next 228 seconds of the signal
    
    Input: List representing a signal
    Output: 3 lists, split in the way described above
    """

    train = []
    validate = []
    test = []

    i = 0
    while i < 288 * Fs:
        # First 48 seconds for training
        if i < 48 * Fs:
            train.append(arr[i])
        # Next 12 seconds for validation
        elif 48 * Fs <= i < 60 * Fs:
            validate.append(arr[i])
        # Final 228 seconds for testing
        else:
            test.append(arr[i])

        i += 1

    return train, validate, test


def segmentation(arr, n):
    """
    Takes in a list representing a signal, and then returns a 2D array. Each inner list of this 2D array is
    an "n" second segment of the original array.

    Input: Signal
    Output: 2D array of the form [[segment 1], [segment 2], ...]
    Takes in an array of data and segments the array into 'n' second intervals 
    """
    arr = arr[0 : 288 * Fs]
    n *= Fs
    segmented = [arr[i : i + n] for i in range(0, len(arr), n)]
    segmented = np.squeeze(segmented) # Makes each segment a 1D list

    return segmented

def fit_format(data):
    """
    Reshapes a 2D array of shape (m, n) to a 3D array of shape (m, n, 1). This is so that the data is ready
    to be passed directly into the Bi-LSTM model.

    Input: 2D array of shape (m, n)
    Output: Same array, but now 3D with shape (m, n, 1)
    """

    data = np.array(data)
    result = data.reshape(data.shape[0], data.shape[1], 1)
    return result

def model_construct(train_data, train_output, num_neuron, num_epoch, batch_size):
    """
    Construts a model with a BiLSTM layer and Dense layer, and train it using the given data.

    Input: PPG training signals, ECG training signals, number of neurons in Bi-LSTM layer, 
    number of epochs to train the model for, batch size to be used in training.
    Output: Trained model
    """
    tf.random.set_seed(1234)
    regulariser = tf.keras.regularizers.L1L2(l1 = 0.0001, l2 = 0.0001)
    
    # Creating model, with Bi-LSTM and Dense Layer
    model = tf.keras.Sequential()
    model_LSTM = tf.keras.layers.LSTM(num_neuron, return_sequences = True, input_shape = (train_data.shape[1],train_data.shape[2]), kernel_regularizer = regulariser)
    model_BiLSTM = tf.keras.layers.Bidirectional(model_LSTM)
    model_Dense = tf.keras.layers.Dense(1)

    # Add optimiser to model and compile it
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.add(model_BiLSTM)
    model.add(model_Dense)
    model.compile(loss = 'mean_squared_error', optimizer = optimizer)

    # Train model
    model.fit(train_data, train_output, epochs = num_epoch, batch_size = batch_size, verbose = 0, shuffle = False)
    return model

if __name__ == "__main__":
    # Section 0: If user passes in additional command line argument, then graphs should be generated
    graph = False
    if len(sys.argv) > 1:
        graph_region = 2 * Fs # Current time segment to graph is 2 seconds - but change be changed
        graph = True

    # Section 1: Load in the ppg and ecg data for all patients
    data = scipy.io.loadmat("original_paper/Records.mat")["records"]

    # Loop through each patient to clean their data, and create a model from this
    for patient in data:
        # Section 2a: This is the first step of preprocessing, where we apply a high and lowpass filter to the ECG data
        # The frequency cutoff for high and low pass filter were 0.5 Hz and 20 Hz respectively.
        # Note that the signal.cheby function requires the frequency cutoff to be passed in as frequency/(sampling rate / 2)
        ecg = unpack_list(patient["ecg_II"][0].tolist())
        b, a = signal.cheby2(4, 20, 0.5 / (Fs / 2), "highpass")
        filtered_ecg_1 = signal.filtfilt(b, a, ecg)
        b, a = signal.cheby2(4, 20, 20 / (Fs / 2), "lowpass") 
        filtered_ecg_2 = signal.filtfilt(b, a, filtered_ecg_1)   

        # For visualisation of ECG noise removal:
        if graph:
            plt.subplot(1, 2, 1)
            plt.plot(ecg[0:graph_region], color = "r")
            plt.title("Original ECG data")
            plt.subplot(1, 2, 2)
            plt.plot(filtered_ecg_2[0:graph_region], color = "r")
            plt.title("Cleaned ECG data")
            plt.show()

        # Section 2b: This is still the first step of preprocesing, where we repeat Section 2a, but for PPG data
        # The frequency cutoff for high and low pass filter were 0.5 and 10 Hz respectively.
        ppg = unpack_list(patient["ppg"][0].tolist())
        b, a = signal.cheby2(4, 20, 0.5 / (Fs / 2), "highpass")
        filtered_ppg_1 = signal.filtfilt(b, a, ppg)
        b, a = signal.cheby2(4, 20, 10 / (Fs / 2), "lowpass") 
        filtered_ppg_2 = signal.filtfilt(b, a, filtered_ppg_1)   

        # For visualisation of PPG noise removal:
        if graph:
            plt.subplot(1, 2, 1)
            plt.plot(ppg[0:graph_region], color = "b")
            plt.title("Original PPG data")
            plt.subplot(1, 2, 2)
            plt.plot(filtered_ppg_2[0:graph_region], color = "b")
            plt.title("Cleaned PPG data")
            plt.show()

        # Section 3a: This is the second step of preprocessing where we obtain the R peaks from the ECG data
        # and the Systolic peaks from PPG data 
        r_peak = get_r_peaks(ecg, filtered_ecg_2)
        s_peak = get_systolic_peaks(ppg, filtered_ppg_2)

        # For visualisation of ECG and PPG peak finding:
        if graph:
            plt.subplot(1, 2, 1)
            plt.plot(filtered_ecg_2[0:graph_region], color = "r")
            plt.title("ECG data with peaks found")
            for peak in r_peak:
                if peak < graph_region:
                    plt.axvline(x = (peak), color = "r")

            plt.subplot(1, 2, 2)
            plt.plot(filtered_ppg_2[0:graph_region], color = "b")
            plt.title("PPG data with peaks found")
            for peak in s_peak:
                if peak < graph_region:
                    plt.axvline(x = (peak), color = "b")
            plt.show()

        # Section 3b: This is still the second step of preprocessing, where we use the peaks obtained from the above
        # and align the ECG and PPG signal according to this
        ppg_aligned, ecg_aligned = alignment(filtered_ppg_2, s_peak, filtered_ecg_2, r_peak)

        # For visualisation of Alignment:
        if graph:
            plt.subplot(1, 2, 1)
            plt.plot(filtered_ppg_2[:graph_region], "b")
            plt.plot(filtered_ecg_2[:graph_region], "r")
            plt.title("Unaligned ECG/PPG data")
            plt.subplot(1, 2, 2)
            plt.plot(ppg_aligned[:graph_region], "b")
            plt.plot(ecg_aligned[:graph_region], "r")
            plt.title("Aligned ECG/PPG data")
            plt.show()

        # Section 4: This is the third step of preprocessing, where we normalise the PPG signal from [0, 1]
        normalised_ppg = normalise(ppg_aligned)

        # For visulation of normalisation 
        if graph:
            plt.subplot(1, 2, 1)
            plt.title("Unnormalised PPG data")
            plt.plot(ppg_aligned[:graph_region], "b")
            plt.subplot(1, 2, 2)
            plt.title("Normalised PPG data")
            plt.plot(normalised_ppg[:graph_region], "b")
            plt.show()

        # Section 5: This is the fourth step of preprocessing, where we split the signals into
        # 48, 12, 228 second segments to be used for training, validation and testing
        train_ecg, val_ecg, test_ecg = data_splitting(ecg_aligned)
        train_ppg, val_ppg, test_ppg = data_splitting(normalised_ppg)

        # Section 6: This is the final step of preprocessing, where we segment the data into 1 second segments,
        # and reshape it such that it is ready to be used for training
        seg_train_ecg = fit_format(segmentation(train_ecg, 1))
        seg_train_ppg = fit_format(segmentation(train_ppg, 1))
        seg_test_ecg = fit_format(segmentation(test_ecg, 1))
        seg_test_ppg = fit_format(segmentation(test_ppg, 1))

        # For visualisation of segmentation
        if graph:
            fig, axes = plt.subplot_mosaic("""
            AB
            AC
            """)
            axes["A"].set_title("Unsegmented PPG/ECG signal")
            axes["A"].plot(ecg_aligned[:graph_region], "r")
            axes["A"].plot(normalised_ppg[:graph_region], "b")

            axes["B"].set_title("Segment I")
            axes["B"].plot(seg_train_ecg[0], "r")
            axes["B"].plot(seg_train_ppg[0], "b")
            axes["C"].set_title("Segment II")
            axes["C"].plot(seg_train_ecg[1], "r")
            axes["C"].plot(seg_train_ppg[1], "b")
            plt.show()

        # Section 7: In this step, we train the model using the training data and then run it on the test data
        # The model is trained using 25 neurons in the Bi-LSTM layer, and is trained in 10 epochs. The batch-size is 1.
        model = model_construct(seg_train_ppg, seg_train_ecg, 25, 10, 1)
        prediction = model.predict(seg_test_ppg, batch_size = 1)

        # Section 8: The output from the model is a list of 1 second segments. In this step, we stitch together the 
        # 1 seconds segments to get the final signal
        prediction = prediction.ravel()

        # For visulation of Predicted ECG from Model
        if graph:
            plt.subplot(1, 2, 1)
            plt.plot(prediction[:graph_region], "r")
            plt.title("Predicted ECG data")
            plt.subplot(1, 2, 2)
            plt.plot(test_ecg[:graph_region], "r")
            plt.title("Actual ECG data")
            plt.show()

            input("") # To pause after each patient