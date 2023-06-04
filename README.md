# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Naadira Sahar N
RegisterNumber: 212221220034 

print("Result Output:")
import chardet 
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

print("data head:")
data.head()

print("data info:")
data.info()

print("data isnull:")
data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

print("y_prediction  value:")
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

print("Accuracy Value:")
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![Screenshot (105)](https://github.com/Naadira/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135126/abe253dd-806d-4bae-babc-82a7631f3d99)
![Screenshot (106)](https://github.com/Naadira/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135126/5c29fa0d-842f-4661-8d44-c0ed066421d1)
![Screenshot (107)](https://github.com/Naadira/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135126/c3b1622f-b208-4d53-a5f1-f48fa701879b)
![Screenshot (108)](https://github.com/Naadira/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135126/43c12ae8-2894-432e-8ed0-3a60d0af67b8)
![Screenshot (109)](https://github.com/Naadira/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135126/5ee73fbb-a403-4cfc-85a8-96790914b394)
![Screenshot (110)](https://github.com/Naadira/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135126/3278d137-28e5-4165-9489-53352d76b8c0)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
