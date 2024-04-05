import numpy as np

class Segment:
    """
    Represents a segment. A segment is an "n" second long section of a set of signals. In this set, one of the signals is the
    output data i.e. WCT signal, while the other signals are the input data i.e. ECG signal.
    A segment is important in this context, as it represents one sample of (input, output) data for the model.

    A segment has the following attributes:
    - the segment length in seconds
    - a list of signal object, representing the signals comprising this segment
    - the sampling rate of the signals in the segment
    """

    def __init__(self, length, signals, Fs):
        self.length = length # Length of segment in seconds
        self.signals = signals # List of signal objects
        self.Fs = Fs

    def remove_noise(self, high_frequency_cutoff, low_frequency_cutoff):
        """
        Removes noise (high frequency bumps and low frequency baseline drift) from the segment, using a bandpass filter
        with the high frequency and low frequency cutoffs given. Nothing is returned, and the noise removal is done by altering
        the attributes of the object.

        Input: high frequency cutoff, low frequency cutoff
        Output: None
        """

        for signal in self.signals:
            signal.remove_noise(high_frequency_cutoff, low_frequency_cutoff)

    def normalise(self):
        """
        Normalises the signals within the segment. The output signal contained in the segment isn't normalised, since 
        the output signal produced by the model should be accurate to what is recorded in real life, and therefore, should not be
        normalised. Nothing is returned, and the normalisation is done by altering the attributes of the object.

        Input: None
        Output: None
        """

        for signal in self.signals:
            signal.normalise()
    
    def format_segment(self):
        """
        Formats the segment in a way that is appropriate to be inputted into the machine learning model.
        The machine learning model requires that the input signals be formatted in the following 2D array:

        [[signal_1[0], signal_2[0], ..., signal_n-1[0]],
        [signal_1[1], signal_2[1], ..., signal_n-1[1]],
        [signal_1[2], signal_2[2], ..., signal_n-1[2]],
        ...
        [signal_1[t], signal_2[t], ..., signal_n-1[t]]]

        The ouptut signal must be formatted in the following 2D array:

        [[signal_1[0]],
        [signal_1[1]],
        [signal_1[2]],
        ...
        [signal_1[t]]]

        Input: None
        Output: 2D array for the formatted input, 2D array for the formatted output
        """

        input_segment = []
        output_segment = []

        # Loops through every unit of time from the signals
        t = 0
        while t < self.length * self.Fs:
            # At each unit of time, collate the input signals and the 
            # output signals, and append them to their appropriate arrays
            current_time = []
            
            for signal in self.signals:
                if signal.type == "WCT":
                    output_segment.append([signal.signal[t]])
                else:
                    current_time.append(signal.signal[t])
                
            input_segment.append(current_time)
            t += 1

        
        return np.array(input_segment), np.array(output_segment)