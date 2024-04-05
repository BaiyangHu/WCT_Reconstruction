import matplotlib.pyplot as plt
import numpy as np
import itertools
import pickle
from sklearn.linear_model import LinearRegression
from model_training_final import load_segments_from_file
from Segment import Segment

def format_segments_linear(segments):
    """
    Takes a list of segments and then converts these segments into two lists, one containing the input signals
    from all the segments, and the other containing the output signals from all the segments. The returned lists
    is in the correct format to train the machine learning model (for linear regression).

    Input: list of segment objects
    Output: input array containing all the ECG signals from the segments, output array containins all the WCT signals
    from the segments
    """

    input_data = []
    output_data = []

    # Iterature through segments
    for segment in segments:
        input, output = segment.format_segment()
        
        # Collate input/output signals into their appropriate lists
        for point in input:
            input_data.append(point)
        for point in output:
            output_data.append(point[0])

    return np.array(input_data), np.array(output_data)

def model_construct(train_data, train_output):
    """
    Construct linear regression model train it using the given data.

    Input: training data input, training data output
    Output: trained linear model
    """

    model = LinearRegression()
    model.fit(train_data, train_output)

    return model

def model_selection(train_data, train_output, validation_data, validation_expected_output):
    """
    Creates 2^n different linear models (where n is the number of features), one for each subset of features.
    Note that in this case the features are the 9 ECG signals, so a linear model is created for each subset
    of these 9 ECG signals. It then evaluates them all to determine which subset of features returns the highest performing linear model
    according to the AIC metric.

    Input: training input data, training output data, validation input data, validation output data
    Output: best subset of features, best linear model trained from this subset
    """

    columns = list(range(len(train_data[0]))) # Produces a list from 0, ...., 8. Each element represents an ECG signal.
    best_aic = np.inf
    best_subset = None
    best_model = None

    # Iterate through each possible size subset of features we can take i.e. 1 to 9
    for num_variables in range(1, len(columns) + 1):
        # For a given size, generate all feature subsets with this size
        for subset in itertools.combinations(columns, num_variables):
            train_data_subset = train_data[:, subset]

            # Train the model using the chosen subset of the training data
            model = model_construct(train_data_subset, train_output)

            # Test the model using the chosen subset of the validation data
            validation_actual_output = model.predict(validation_data[:, subset])

            # Calculating AIC of model
            mean_error_squared = np.mean((validation_expected_output - validation_actual_output) ** 2)
            aic = len(validation_actual_output) * np.log(mean_error_squared) + 2 * num_variables

            # Store best AIC
            if aic < best_aic:
                best_aic = aic
                best_subset = subset
                best_model = model

    return best_subset, best_model

if __name__ == "__main__":
    # Load training/validation/testing data
    training_segments = load_segments_from_file("cleaned-data/cleaned_training_WCT_data")
    validation_segments = load_segments_from_file("cleaned-data/cleaned_validation_WCT_data")
    # Format segments so that they are ready to be used to train the linear model
    training_input_data, training_output_data = format_segments_linear(training_segments)    
    validation_input_data, validation_expected_output = format_segments_linear(validation_segments)

    # Find which subset of features results in best performing model, and return this model
    best_subset, best_model = model_selection(training_input_data, training_output_data, validation_input_data, validation_expected_output)
    print("Subset of variables resulting in best model are: {}".format(best_subset))

    # Save this model to file
    with open("models/best_linear_model", 'wb') as f:
        pickle.dump(best_model, f)
    with open("models/best_linear_model's_subset", 'wb') as f:
        pickle.dump(best_subset, f)