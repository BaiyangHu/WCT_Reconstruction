import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
from Segment import Segment

def load_segments_from_file(folder_name):
    """
    Takes all segment files in the given foldername and loads them as a list of objects, which it returns

    Input: folder_name, which contains a list of Segment files
    Output: list of Segment objects
    """
    ls = []

    # Go through all segment files in the given folder
    for segment_file in sorted(os.listdir(folder_name)):
        file_name = folder_name + "/" + segment_file

        # Open file and load in segment object
        with open(file_name, "rb") as f:
            segment = pickle.load(f)

        ls.append(segment)

    return ls

def format_segments(segments):
    """
    Takes a list of segments and then converts these segments into two lists, one containing the input (ECG) signals
    from all the segments, and the other containing the output (WCT) signal from all the segments. The returned lists
    are in the correct format to train the machine learning model.

    Input: list of segment objects
    Output: input list which contains the ECG signals from all the segment objects, output list which contains WCT signals
    from all the segment objects
    """

    input_data = []
    output_data = []

    # Collate segments into a single array ready to be used to train the model
    for segment in segments:
        input, output = segment.format_segment()
        input_data.append(input)
        output_data.append(output)

    return np.array(input_data), np.array(output_data)

def model_construct(train_data, train_output, num_neuron, num_epoch, batch_size, l1, l2, learning_rate):
    """
    Construct a model with a BiLSTM + Dense layer, and train it using the given data.

    Input: train_data (input), train_output (output), num_neurons (number of neurons in Bi-LSTM layer), num_epoch
    (number of epochs in training), batch_size, l1, l2 (to be used in regulariser), learning_rate
    Output: The trained model
    """
    
    regulariser = tf.keras.regularizers.L1L2(l1 = l1, l2 = l2)
    # Creating model, with Bi-LSTM and Dense Layer
    model = tf.keras.Sequential()
    model_LSTM = tf.keras.layers.LSTM(num_neuron, return_sequences = True, input_shape = (train_data.shape[1], train_data.shape[2]), kernel_regularizer = regulariser)
    model_BiLSTM = tf.keras.layers.Bidirectional(model_LSTM)
    model_Dense = tf.keras.layers.Dense(1)

    # Add optimiser to model and compile it
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.add(model_BiLSTM)
    model.add(model_Dense)
    model.compile(loss = 'mean_squared_error', optimizer = optimizer)

    # Train model
    # verbose = 0 ensures no output to terminal, and shuffle = false maintains order of training data
    model.fit(train_data, train_output, epochs = num_epoch, batch_size = batch_size, verbose = 0, shuffle = False)
    return model

if __name__ == "__main__":
    tf.random.set_seed(1234)
    num_epochs = 1000 # Number of epochs to run the training for (can change)
    num_neuron = 25 # Number of neurons in BiLSTM layer (can change)
    learning_rate = 0.001 # Learning rate of optimiser (can change)
    l1 = 0.0001 # L1 and L2 of regulariser (can change)
    l2 = 0.0001
    batch_size = 1 # Batch size (generally don't change)

    # Load and format cleaned training data so that it's ready to be passed into the model
    training_segments = load_segments_from_file("cleaned-data/cleaned_training_WCT_data")
    training_input_data, training_output_data = format_segments(training_segments)

    # Train the model using the training data
    model = model_construct(training_input_data, training_output_data, num_neuron, num_epochs, batch_size, l1, l2, learning_rate)

    # Saving model to file, with a name determined by the hyperparameters used
    model_desc = "epochs={},neurons={},rate={},l1={},l2={}".format(num_epochs, num_neuron, learning_rate, l1, l2)
    model.save("models/{}".format(model_desc))