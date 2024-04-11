# ADULT-INCOME-CLASSIFICATION-PROJECT

In this project i have predicted income of adults by various factors. 

# Predict adult income by Decison tree classifier

# import libraries 

import pandas as pd 

# read dataset 

data = pd.read_csv('decision tree.csv')

# check for null values 

data.isnull().sum(axis = 0)

data.dtypes 

data_prep = pd.get_dummies(data, drop_first = True)

# create the X and Y variables 

X = data_prep.iloc[:,:-1]
Y = data_prep.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1234, stratify=Y)

from sklearn.tree import DecisionTreeClassifier

# TRAIN THE MODEL 

dtc = DecisionTreeClassifier(random_state = 1234)
dtc.fit(X_train,Y_train)

Y_predict = dtc.predict(X_test)

# Evaluate the model 

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(Y_test,Y_predict)

score = dtc.score(X_test,Y_test)  

# PROJECT END
