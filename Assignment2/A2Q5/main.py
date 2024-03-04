# -------------------------------------------------------------------------
# AUTHOR: Chanrady Ho
# FILENAME: decision tree prediction on cheat dataset
# SPECIFICATION: run 10 times of each dataset, predict the testing dataset, compare the results, and calculate accuracy
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)  # reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:, 1:]  # creating a training matrix without the id (NumPy library)

    # transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    # be converted to a float.
    for train in data_training:
        refund, marital_status, taxable_income, cheat = train[0], train[1], train[2], train[3]
        refund = 1 if refund == 'Yes' else 2
        taxable_income = int(taxable_income[:-1])
        marital_status_label = {'Single': [1, 0, 0], 'Married': [0, 1, 0], 'Divorced': [0, 0, 1]}
        X.append([refund] + marital_status_label[marital_status] + [taxable_income])

        # transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
        # --> add your Python code here
        class_label = {'Yes': 1, 'No': 2}
        Y.append(class_label[cheat])

    # loop your training and test tasks 10 times here
    accuracy = 0
    for i in range (10):

        # fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        # plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()

        # read the test data and add this data to data_test NumPy
        # --> add your Python code here
        data_test = pd.read_csv('cheat_test.csv', sep=',', header=0)
        data_test = np.array(data_test.values)[:, 1:]

        correct_prediction = 0
        for data in data_test:
            # transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            element = []
            refund, marital_status, taxable_income, cheat = data[0], data[1], data[2], data[3]
            refund = 1 if refund == 'Yes' else 2
            taxable_income = int(taxable_income[:-1])
            marital_status_label = {'Single': [1, 0, 0], 'Married': [0, 1, 0], 'Divorced': [0, 0, 1]}
            element.append([refund] + marital_status_label[marital_status] + [taxable_income])
            class_predicted = clf.predict(element)[0]

            # compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            class_label = {'Yes': 1, 'No': 2}
            if class_predicted == class_label[cheat]:
                correct_prediction += 1
            # print(correct_prediction)

        # find the average accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        accuracy = correct_prediction / len(data_test)

    # print the accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    # --> add your Python code here
    print(f'Final accuracy when training on {ds}: {accuracy}')
