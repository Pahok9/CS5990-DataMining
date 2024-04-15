# -------------------------------------------------------------------------
# AUTHOR: Chanrady Ho
# FILENAME: KNN
# SPECIFICATION: KNN model training and testing
# FOR: CS 5990- Assignment #3
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np

# 11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

# defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

# reading the training data
training_df = pd.read_csv('weather_training.csv', header=0)
# reading the test data
# hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
test_df = pd.read_csv('weather_test.csv', header=0)

x_training = np.array(training_df.values[:, 1:-1].astype('f'))
y_training = np.array(training_df.values[:, -1].astype('f'))

x_test = np.array(test_df.values[:, 1:-1].astype('f'))
y_test = np.array(test_df.values[:, -1].astype('f'))

normalized_y_training = preprocessing.normalize([y_training])
temperature_midpoint = (np.max(normalized_y_training) - np.min(normalized_y_training)) / 2
y_training = np.where(y_training <= temperature_midpoint, 'low', 'high')

normalized_y_test = preprocessing.normalize([y_test])
y_test = np.where(y_test <= temperature_midpoint, 'low', 'high')

# loop over the hyperparameter values (k, p, and w) ok KNN
# --> add your Python code here
highest_accuracy = 0
for k in k_values:
    for p in p_values:
        for w in w_values:
            # fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(x_training, y_training)

            # make the KNN prediction for each test sample and start computing its accuracy
            # hint: to iterate over two collections simultaneously, use zip()
            # Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            # to make a prediction do: clf.predict([x_testSample])
            # the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            # to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            # --> add your Python code here
            correct_prediction = 0
            for (x_testSample, y_testSample) in zip(x_test, y_test):
                if clf.predict([x_testSample]) == y_testSample:
                    correct_prediction += 1

            # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            # with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            # --> add your Python code here
            current_accuracy = correct_prediction / len(y_test)
            if current_accuracy > highest_accuracy:
                highest_accuracy = current_accuracy
                print(f'Highest KNN accuracy so far: {highest_accuracy}, Parameters: k={k}, p={p}, w={w}')
