# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: P PARTHIBAN
RegisterNumber:  212223230145
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('/content/Employee.csv')
data
X = data.iloc[:,[0,1,2,3,4,5,7,8,9]]
X
y = data['left']
y

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
## Output:

## data
![image](https://github.com/user-attachments/assets/38f522fd-1318-42c5-a4df-153b52bd15d6)

## X_variables
![image](https://github.com/user-attachments/assets/dec5672f-8c37-46ff-8dd7-83b8b5e07f38)

## y_variables
![image](https://github.com/user-attachments/assets/df5789b6-cd32-4a51-8f7c-24c434f256be)

## y_pred
![image](https://github.com/user-attachments/assets/1760f955-a4b3-41a9-8091-a9768cc051b4)

## confusion matrix
![image](https://github.com/user-attachments/assets/ef0732c2-0929-4303-9e39-c8231c988a68)

![image](https://github.com/user-attachments/assets/55783c3a-e779-40fd-80e9-f68c6ce50217)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
