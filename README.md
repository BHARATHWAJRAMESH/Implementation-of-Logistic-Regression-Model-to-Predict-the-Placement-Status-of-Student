# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated()function respectively.
3.LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset. 5.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6.Apply new unknown values. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BHARATHWAJ R
RegisterNumber:  212222240019
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#remove the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
1.Placement Data

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/0e3d4b35-e0a4-4b75-b3e0-b5fff49af6d1)


2.Salary Data

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/3b6bd619-de38-4b86-9689-a2e92454699e)


3.Checking the null() function

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/8bbb7f1c-7995-4bfa-a98e-caf641324b99)


4.Data Duplicate

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/e38b946b-96ca-4171-bcba-77d8eacf0657)


5.Print Data

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/4335fcb7-8e1e-46dd-a7e2-488734bf3fe4)


6.Data-status

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/2b214310-bbfa-486b-a447-f5e0d0e38d1e)


7.y_prediction array

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/b2107a91-23bb-415c-9e33-b170a9e6c611)


8.Accuracy Value

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/13e59d96-3192-45b2-9bba-def345e36fcc)


9.Confusion Array

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/7d0a04e8-6535-4172-aa93-54817dbb8b98)


10.Classification Report

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/6480bcbf-14cd-44fc-8227-3ca51356221f)


![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/20db6e43-1d8f-4f6f-8502-2d8fe5f1a405)


11.Prediction of LR

![image](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/9a496e35-c184-432e-a760-be7b72d42eb1)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
