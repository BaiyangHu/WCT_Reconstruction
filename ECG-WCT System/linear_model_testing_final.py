import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from sklearn.linear_model import LinearRegression
from model_training_final import load_segments_from_file
from model_testing_final import evaluate_accuracy
from model_testing_final import paired_t_test
from linear_model_training_final import format_segments_linear
from Segment import Segment

if __name__ == "__main__":
    # User passes in an extra command line argument to create graphs if required
    graph = False
    if len(sys.argv) > 1:
        graph = True

    # Load in model and subset
    with open("models/best_linear_model", "rb") as f:
        best_model = pickle.load(f)
    with open("models/best_linear_model's_subset", "rb") as f:
        best_subset = pickle.load(f)

    # Load and format training/validation/test set
    training_segments = load_segments_from_file("cleaned-data/cleaned_training_WCT_data")
    training_input_data, training_expected_output = format_segments_linear(training_segments)
    validation_segments = load_segments_from_file("cleaned-data/cleaned_validation_WCT_data")
    validation_input_data, validation_expected_output = format_segments_linear(validation_segments)
    testing_segments = load_segments_from_file("cleaned-data/cleaned_testing_WCT_data")
    testing_input_data, testing_expected_output = format_segments_linear(testing_segments)

    # Use model to get prediction on each of these sets
    training_actual_output = best_model.predict(training_input_data[:, best_subset])
    validation_actual_output = best_model.predict(validation_input_data[:, best_subset])
    testing_actual_output = best_model.predict(testing_input_data[:, best_subset])

    # Reshaping the output so that it is separated into segments again
    Fs = 800
    segment_len = 1
    training_expected_output = np.reshape(training_expected_output, (-1, Fs * segment_len))
    training_actual_output = np.reshape(training_actual_output, (-1, Fs * segment_len))
    validation_expected_output = np.reshape(validation_expected_output, (-1, Fs * segment_len))
    validation_actual_output = np.reshape(validation_actual_output, (-1, Fs * segment_len))
    testing_expected_output = np.reshape(testing_expected_output, (-1, Fs * segment_len))
    testing_actual_output = np.reshape(testing_actual_output, (-1, Fs * segment_len))

    # Evaluate accuracy of output
    training_rms, training_pearson = evaluate_accuracy(training_expected_output, training_actual_output, 'results/training_linear', graph)
    validation_rms, validation_pearson = evaluate_accuracy(validation_expected_output, validation_actual_output, 'results/validation_linear', graph)
    testing_rms, testing_pearson = evaluate_accuracy(testing_expected_output, testing_actual_output, 'results/testing_linear', graph)
    test_stats = paired_t_test(testing_expected_output, testing_actual_output)

    print("Average RMS in test set is: {}".format(testing_rms))
    print("Average Pearsons Correlation Coefficient in test set is: {}".format(testing_pearson))
    print("Paired t test test statistic: {} and pvalue: {}".format(test_stats.statistic, test_stats.pvalue))