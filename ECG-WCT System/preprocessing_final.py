import wfdb
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pickle
import sys
from Segment import Segment
from Signal import Signal

def record_to_segments(record, signal_start, signal_finish, segment_len, Fs):
    """
    Takes as input a wfdb record, which contains a sample with all 12 ECG and 1 WCT signals, and then creates a list of
    segment objects from the record according to the segment length defined in the function. For instance,
    an 8 second wfdb record will be converted into 8 segment objects, each which contains a 1 second piece of all 13 signals.

    Input: wfdb record with all signals, start time of signal, end time of signal, length of segment to be produced, sampling rate
    Output: list of segment objects produced from given signal
    """

    record_length = signal_finish - signal_start
    no_segments = int(record_length // segment_len)
    res = []

    # Iterate through each segment to be created
    for i in range(no_segments):
            # Slice the segment from the wfdb record and convert to numpy array
            segment_start = math.floor(i * Fs * segment_len)
            segment_finish = math.floor((i + 1) * Fs * segment_len)
            segment = record.p_signal[segment_start : segment_finish, :]
            segment = np.array(segment).transpose()

            # Generate signal objects from sliced segment
            signal_list = [
                Signal(record.sig_name[i], segment[i], Fs) for i in range(len(segment))
                ]

            # Generate a segment object from the signals objects created
            segment = Segment(segment_len, signal_list, Fs)
            res.append(segment)

    return res

def save_segments_to_file(segments, folder_name):
    """
    Save a list of segments to a subfolder given by folder_name. Each of the segments will 
    be a binary file in folder_name.

    Input: list of segment objects, folder name to save objects to
    Output: None
    """

    # Iterate through all segments
    for i, segment in zip(range(len(segments)), segments):
        path = folder_name + "/" + "segment {}".format(i)

        # Save segment to file
        with open(path, 'wb') as f:
            pickle.dump(segment, f)

if __name__ == "__main__":
    # User passes in an extra command line argument if they want graphs to be produced
    graph = False
    if len(sys.argv) > 1:
        graph = True

    Fs = 800 # Sampling rate of signal. Fixed for our particular use case.
    low_frequency_cutoff = 0.5 # Frequency cutoffs used in bandpass filter
    high_frequency_cutoff = 20

    signal_start = 1 # Point in time to start the signal, determined via visual inspection across all data.
    signal_finish = 9 # Point in time to finish the signal, determined via visual inspection across all data.
    training_percent = 0.6 # Proportion of signal to allocate to training
    validation_percent = 0.1 # Proportion of signal to allocate for validation
    testing_percent = 0.3 # Proportion of signal to allocate for testing
    segment_len = 1 # Length of each segment, that a signal is split into

    signal_channels = [18, 19, 20, 21, 22, 23, 24, 25, 26, 36] # Signals we're using (9 ECG and 1 WCT)
    path = "./raw-data" # Path to dataset

    training_segments = [] # list of Segment objects for training
    validation_segments = [] # list of Segment objects for validation
    testing_segments = []  # list of Segment objects for testing

    # Section 1: Loop through each patient (where each patient corresponds to a subfolder) in the dataset path
    for patient in sorted(os.listdir(path)):
        # Skip subfolders which are not associated with a patient
        if "patient" not in patient:
            continue
        
        patient_path = path + "/" + patient
        # Section 2: Loop through the files within each subfolder. This accesses the samples for each patient.
        for sample in sorted(os.listdir(patient_path)):
            # Skip files which are not associated with a signal
            if ".hea" in sample:
                continue

            sample_path = patient_path + "/" + sample[:-4] # Removes .dat extension
            # Load a record using the 'rdrecord' function from 1 to 9 second interval, so 8 seconds all up
            record = wfdb.rdrecord(sample_path, sampfrom = signal_start * Fs, sampto = signal_finish * Fs, channels = signal_channels)
            # Segment the record according to segment_len and produce Segment objects from it
            segments = record_to_segments(record, signal_start, signal_finish, segment_len, Fs)

            # Apply cleaning to each of the segments, with graphs to show cleaning if required
            for segment in segments:
                if graph:
                    segment.signals[0].visualise(0, 1, (1, 3, 1))

                segment.remove_noise(high_frequency_cutoff, low_frequency_cutoff)
                if graph:
                    segment.signals[0].visualise(0, 1, (1, 3, 2))

                segment.normalise()
                if graph:
                    segment.signals[0].visualise(0, 1, (1, 3, 3))
                    plt.show()

            # Splitting segments into training, testing and validation
            for i in range(len(segments)):
                if i < math.floor(len(segments) * training_percent):
                    training_segments.append(segments[i])
                elif i < math.floor(len(segments) * (training_percent + testing_percent)):
                    testing_segments.append(segments[i])
                else:
                    validation_segments.append(segments[i])

            # Pause after patient is finished if required
            if graph:
                input("")

    # Save all segments to file at the end
    save_segments_to_file(training_segments, "cleaned-data/cleaned_training_WCT_data")
    save_segments_to_file(validation_segments, "cleaned-data/cleaned_validation_WCT_data")
    save_segments_to_file(testing_segments, "cleaned-data/cleaned_testing_WCT_data")