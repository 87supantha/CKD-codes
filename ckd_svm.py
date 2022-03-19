import pandas as pd
#1: from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('ckd.csv')
print(df.head())
print('\nDescription: ')
print(df.describe())
print('\nTransposed Description: ')
print(df.describe().transpose)
print('\nShape: ')
print(df.shape)

X = df.drop('class', axis=1)
y = df['class']
print(X.shape)
print(y.shape)

#1: Split our data into training and testing sets using normal method
#X_train, X_test, y_train, y_test = train_test_split(X, y) # 75% : 25% split by default
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#2: Split our data into training and testing sets using k-fold CV method
kf = KFold(n_splits=5, random_state=None, shuffle=True) # define the split into 5 folds 
for train_index, test_index in kf.split(X):
    #print('\n\nTRAIN:', train_index, '\n\nTEST:', test_index)
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
# Training the model
clf = svm.SVC(kernel='linear', C=1, gamma=1)
clf.fit(X_train, y_train)

# Predictions and Evaluation
predictions = clf.predict(X_test)
print('\nConfusion Matrix:~ ')
print(confusion_matrix(y_test, predictions))
print('\nStatistical measures derived from confusion matrix:~ \n')
print(classification_report(y_test, predictions))
