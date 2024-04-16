#-------------------------------------------------------------------------
# AUTHOR: Chanrady Ho
# FILENAME: naive bayes
# SPECIFICATION: naive bayes model training and testing
# FOR: CS 5990- Assignment #3
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


#11 classes after discretization
def discretize_value(value):
    classes = [i for i in range(-22, 40, 6)]
    return min(classes, key=lambda x: abs(x - value))

#reading the training data
#--> add your Python code here
training_df = pd.read_csv('weather_training.csv', header=0)
x_training = np.array(training_df.values[:, 1:-1]).astype('f')
#update the training class values according to the discretization (11 values only)
#--> add your Python code here
y_training = np.array([discretize_value(y) for y in training_df.values[:, -1]]).astype('f')

#reading the test data
#--> add your Python code here
test_df = pd.read_csv('weather_test.csv', header=0)
x_test = np.array(test_df.values[:, 1:-1]).astype('f')
#update the test class values according to the discretization (11 values only)
#--> add your Python code here
y_test = np.array([discretize_value(y) for y in test_df.values[:, -1]]).astype('f')


#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(x_training, y_training)

#make the naive_bayes prediction for each test sample and start computing its accuracy
#the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
#to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
#--> add your Python code here
correct_prediction = 0
for predicted_value, real_value in zip(clf.predict(x_test), y_test):
    percentage_difference = 100 * abs(predicted_value - real_value) / real_value
    if percentage_difference <= 15:
        correct_prediction += 1

#print the naive_bayes accuracyy
#--> add your Python code here
accuracy = correct_prediction / len(y_test)
print(f"naive_bayes accuracy: {accuracy}")



