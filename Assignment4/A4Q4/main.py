#-------------------------------------------------------------------------
# AUTHOR: Chanrady Ho
# FILENAME: Base classifier and ensemble, and random Forest
# SPECIFICATION: Base classifier prediction and ensemble on 20 base classifiers prediction, and random Forest prediction
# FOR: CS 5990- Assignment #4
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = [] #this array will be used to count the votes of each classifier

#reading the training data from a csv file and populate dbTraining
#--> add your Python code here
dbTraining = pd.read_csv('optdigits.tra')
dbTraining = np.array(dbTraining.values.astype('f'))

#reading the test data from a csv file and populate dbTest
#--> add your Python code here
dbTest = pd.read_csv('optdigits.tes')
dbTest = np.array(dbTest.values.astype('f'))

#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here
for _ in dbTest:
    classVotes.append([0] * 10)

print("Started my base and ensemble classifier ...")

correct_base_prediction = 0
correct_ensemble_prediction = 0
for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

    bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

    #populate the values of X_training and y_training by using the bootstrapSample
    #--> add your Python code here
    X_training = bootstrapSample[:, :-1]
    y_training = bootstrapSample[:, -1]

    #fitting the decision tree to the data
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
    clf = clf.fit(X_training, y_training)

    for i, testSample in enumerate(dbTest):
        #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
        # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
        # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
        # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
        # this array will consolidate the votes of all classifier for all test samples
        #--> add your Python code here
        class_predict = clf.predict([testSample[:-1]])[0]
        classVotes[i][int(class_predict)] += 1

        if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
            #--> add your Python code here
            if class_predict == testSample[-1]:
                correct_base_prediction += 1

    if k == 0: #for only the first base classifier, print its accuracy here
        #--> add your Python code here
        accuracy = correct_base_prediction / len(dbTest)
        print("Finished my base classifier (fast but relatively low accuracy) ...")
        print("My base classifier accuracy: " + str(accuracy))
        print("")

    #now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
    #--> add your Python code here
    for i, testSample in enumerate(dbTest):
        if classVotes[i].index(max(classVotes[i])) == int(testSample[-1]):
            correct_ensemble_prediction += 1

#printing the ensemble accuracy here
accuracy = correct_ensemble_prediction / (len(dbTest) * 20)
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
random_forest_clf = clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
#--> add your Python code here
correct_random_forest_prediction = 0
for i, testSample in enumerate(dbTest):
    rf_predictions = random_forest_clf.predict([testSample[:-1]])[0]

    #compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    #--> add your Python code here
    if int(rf_predictions) == int(testSample[-1]):
        correct_random_forest_prediction += 1

#printing Random Forest accuracy here
accuracy = correct_random_forest_prediction / len(dbTest)
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
