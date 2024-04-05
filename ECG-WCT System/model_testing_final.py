import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.stats
import sys
from Segment import Segment
from model_training_final import load_segments_from_file
from model_training_final import format_segments

def paired_t_test(expected_data, actual_data):
    """
    Takes the WCT output signal from the model and the WCT output signal recorded from the patient
    and performs a paired t test on them.

    Input: WCT output from patient, WCT output from model
    Output: t-test statistics on the two WCT signals
    """

    expected_data = np.ndarray.flatten(expected_data)
    actual_data = np.ndarray.flatten(actual_data)
    test_stats = scipy.stats.ttest_rel(expected_data, actual_data)
    return test_stats

def evaluate_accuracy(expected_outputs, actual_outputs, folder_name, graph = False):
    """
    Given an array of expected_outputs and actual_outputs, return the average accuracy i.e. similarity
    between these two outputs averaged across all signals. This will be done using RMS error and Pearon's
    correlation coefficient.

    It will also save the graphs of the signals to the foldername specified, and the RMS/Coefficient of the
    signals to a csv file in the foldername specified.

    Input: expected WCT output, actual WCT output, foldername to save results to, "graph" indicates whether to generate
    a graph or not.
    Output: average RMS, average Pearson's correlation coefficient
    """
    rmses = []
    pearsons = []

    # Open csv file to write RMS and Coefficient of each signal into
    f = open(folder_name + '.csv', 'w')
    f.write('Figure Index, Root Mean Square, Pearson Correlation Coefficent\n')

    # For each signal
    for expected_output, actual_output, i in zip(expected_outputs, actual_outputs, range(len(expected_outputs))):
        # Flattens the data so it's a 1D array again
        expected_output = expected_output.ravel()
        actual_output = actual_output.ravel()

        # Calculate accuracy metrics:
        rms = (((expected_output - actual_output) ** 2).mean()) ** (1/2) # Get the RMS error between signals
        rmses.append(rms)
        pearson = scipy.stats.pearsonr(expected_output, actual_output).statistic # Get the coefficient between signals
        pearsons.append(pearson)

        # Saving reuslts into the csv folder
        f.write('{}, {}, {}\n'.format(i, rms, pearson))

        # For visualisation of the expected output and actual output if required
        # These figures generated are also saved to file
        if graph:
            plt.subplot(1, 2, 1)
            plt.plot(expected_output, "b")
            plt.title("Expected WCT from patient")
            plt.subplot(1, 2, 2)
            plt.plot(actual_output, "r")
            plt.title("Actual WCT from model")
            plt.savefig(folder_name + '/' + str(i) + '.png')
            plt.show()

    f.close()
    return sum(rmses) / len(rmses), sum(pearsons) / len(pearsons)

if __name__ == "__main__":
    # User passes in an extra command line argument to create graph if they want
    graph = False 
    if len(sys.argv) > 1:
        graph = True

    # Loads in model
    model_name = "epochs={},neurons={},rate={},l1={},l2={}".format(100, 25, 0.001, 0.0001, 0.0001) # (can change) Change according to which model to load
    model = tf.keras.models.load_model("models/{}".format(model_name))

    # Load and format training/validation/test set
    training_segments = load_segments_from_file("cleaned-data/cleaned_training_WCT_data")
    training_input_data, training_expected_output = format_segments(training_segments)

    validation_segments = load_segments_from_file("cleaned-data/cleaned_validation_WCT_data")
    validation_input_data, validation_expected_output = format_segments(validation_segments)

    testing_segments = load_segments_from_file("cleaned-data/cleaned_testing_WCT_data")
    testing_input_data, testing_expected_output = format_segments(testing_segments)

    # Run model on training/validation/test set
    training_actual_output = model.predict(training_input_data)
    validation_actual_output = model.predict(validation_input_data)
    testing_actual_output = model.predict(testing_input_data)

    # Evaluate accuracy on training/validation/test set
    training_rms, training_pearson = evaluate_accuracy(training_expected_output, training_actual_output, 'results/training_bilstm', graph)
    validation_rms, validation_pearson = evaluate_accuracy(validation_expected_output, validation_actual_output, 'results/validation_bilstm', graph)
    testing_rms, testing_pearson = evaluate_accuracy(testing_expected_output, testing_actual_output, 'results/testing_bilstm', graph)
    test_stats = paired_t_test(testing_actual_output, testing_expected_output)

    print("Average RMS in validation set is: {}".format(validation_rms))
    print("Average Pearsons Correlation Coefficient in validation set is: {}".format(validation_pearson))
    print("Average RMS in test set is: {}".format(testing_rms))
    print("Average Pearsons Correlation Coefficient in test set is: {}".format(testing_pearson))
    print("Paired t test test statistic: {} and pvalue: {} (for test set)".format(test_stats.statistic, test_stats.pvalue))
