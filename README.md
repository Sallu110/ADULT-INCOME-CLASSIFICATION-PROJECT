
# Predicting Adult Income using Decision Tree Classifier
This project aims to predict whether an adult's income exceeds $50K/year based on demographic and employment-related features using a Decision Tree Classifier.

# steps of project
Introduction
Data Preprocessing
Model Training
Evaluation
Conclusion

# Introduction
The project utilizes a Decision Tree Classifier to classify adults into income categories based on features such as education level, race, gender, hours worked per week, company type, and marital status.

# Data Preprocessing
Import Libraries: Import pandas for data handling:
# import pandas as pd
Read Data: Read the dataset and handle missing values if any:
data = pd.read_csv('decision_tree.csv')
Data Preparation: Encode categorical variables into numeric format using one-hot encoding:
data_prep = pd.get_dummies(data, drop_first=True)
Split Data: Split the dataset into features (X) and the target variable (Y):
X = data_prep.drop(['income_>50K'], axis=1)
Y = data_prep['income_>50K']

Train-Test Split: Split the dataset into training and testing sets:
# from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

# Model Training
Select and Train Decision Tree Classifier: Choose a Decision Tree Classifier and train the model:
# from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)
dtc.fit(X_train, Y_train)


# Evaluation
Evaluate the Model: Evaluate the model's performance on the test set using accuracy metrics:
score = dtc.score(X_test, Y_test)
Confusion Matrix: Optionally, analyze model predictions using a confusion matrix for further insights:
# from sklearn.metrics import confusion_matrix
Y_predict = dtc.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict)

![Screenshot 2024-07-18 164552](https://github.com/user-attachments/assets/cfef7992-8a5a-4167-b3af-5bf0eaad4dd6)

![Screenshot 2024-07-18 164641](https://github.com/user-attachments/assets/a0df5a59-af00-425a-92ac-2dc60eb01a16)


# Conclusion
This project demonstrates the application of a Decision Tree Classifier to predict adult income based on demographic and employment-related features. Model performance is evaluated using accuracy metrics and optionally a confusion matrix.





