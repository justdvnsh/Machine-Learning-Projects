# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
# preprocessing the string vars to int numbers using skcikits label encoders
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set

## importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Create your classifier here

classifier = Sequential()

## adding the input layer and first hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))

## adding the second layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

## adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

## compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## fitting the ANN to the training set.

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = ( y_pred > 0.5 )

"""
Prediction of the customer churn rate for this particular customer
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""

new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1,1, 50000]])))
new_pred = (new_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## Now , since we get different acccuracies when we train models different times, we are facing the Bias-Variance tradeoff.
## So , in order to avoid that we need to apply the K-Fold Cross Validation method, which is nothing, but , it 
## divides the training model into separate training folds and test the model, on the same exact fold.

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    
    classifier = Sequential()

    ## adding the input layer and first hidden layer

    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

    ## adding the second layer

    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

    ## adding the output layer

    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    ## compiling the ANN

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

print('Mean Accuracy: {0} and Mean Variance: {1}'.format(mean, variance))

## Tuning the ANN


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    
    classifier = Sequential()

    ## adding the input layer and first hidden layer

    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

    ## adding the second layer

    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

    ## adding the output layer

    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    ## compiling the ANN

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
paramters = {
    'batch_size': [25, 35, 50],
    'nb_epoch': [200, 500, 800],
    'optimizer': ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator = classifier, param_grid = paramters, scoring = 'accuracy', cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_paramters = grid_search._best_params_
best_accuracy = grid_search._best_score_

print('Best Paramters are: {0} and Best Accuracy is: {1} '.format(best_paramters, best_accuracy))