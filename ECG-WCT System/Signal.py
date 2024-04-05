import scipy
import numpy as np
import matplotlib.pyplot as plt

class Signal:
    """
    Represents the signal of a segment. Each signal has the following attributes:
    - the actual signal itself (i.e. list of values)
    - a type (e.g. lead 1, lead 2, WCT, etc.)
    - the sampling rate of the signal
    """

    def __init__(self, type, signal, Fs):
        self.type = type
        self.signal = signal
        self.Fs = Fs

    def remove_noise(self, high_frequency_cutoff, low_frequency_cutoff):
        """
        Removes high and low frequency noise from the current signal, according to the cutoffs given.
        The cleaned signal is not outputted, but rather, saved back into the self.signal attribute.

        Input: the high and low frequency cutoff to use in the bandpass filter
        Output: None
        """

        # The frequency is divided by sampling rate/2, as required by the function scipy.signal.cheby
        b, a = scipy.signal.cheby2(4, 20, low_frequency_cutoff / (self.Fs / 2), "highpass") 
        filtered_signal_1 = scipy.signal.filtfilt(b, a, self.signal)
        b, a = scipy.signal.cheby2(4, 20, high_frequency_cutoff / (self.Fs / 2), "lowpass") 
        self.signal = scipy.signal.filtfilt(b, a, filtered_signal_1)

    def normalise(self):
        """
        Normalises the current signal if it is an ECG signal, since these are the signals to be inputted into
        the model. The normalised signal is saved back into the signal attribute of the object, and nothing
        is returned. The normalisation is done from 0 to 1.

        Input: None
        Output: None
        """

        # Don't want to normalise the WCT signal, since this is what is outputted by the model
        if self.type == "WCT":
            return

        normalised_arr = []
        max_val = np.amax(self.signal)
        min_val = np.amin(self.signal)

        for val in self.signal:
            normalised_arr.append((val - min_val)/(max_val - min_val)) # Calculation ensures value lies from [0, 1]

        normalised_arr = np.array(normalised_arr)
        self.signal = normalised_arr

    def visualise(self, start_time, end_time, subplot):
        """
        Helper function which visualises the signal from the start_time to end_time by producing graph. 
        If multiple calls to this function are made, where a different subplot parameter is passed in then a
        figure to be produced with multiple subplots.

        Inputs: start time of signal to graph, end time of signal to graph, subplot which the current graph belongs to
        Output: None (the graph is produced directly)
        """

        plt.subplot(*subplot)
        
        start_index = int(start_time * self.Fs)
        end_index = int(end_time * self.Fs)
        plt.plot(self.signal[start_index : end_index], "b")
        plt.title("{} Signal".format(self.type))