import pandas as pd
import numpy as np
# Import models from scikit learn module:
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn import metrics


#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("\nAccuracy: %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 10 folds
  kf = KFold(data.shape[0], n_folds=10)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    # Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("\nCross-Validation Score: %s" % "{0:.3%}".format(np.mean(error)) )

  # Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


# Reading the dataset in a dataframe using Pandas
df = pd.read_csv('ckd.csv') 

# Building Logistic Regression model
outcome_var = 'class'
model = LogisticRegression()
#predictor_var = ['Credit_History','ApplicantIncome','CoapplicantIncome','Loan_Amount_Term']
predictor_var = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane']
print("\nPerformance of Logistic Regression model:~ ") 
classification_model(model, df, predictor_var, outcome_var)
# to see the feature importance matrix from which we???ll take the most important features
#create the RFE model and select 3 attributes
#rfe = RFE(model, 3)
#rfe = rfe.fit(df, df[outcome_var])
#summarize the selection of the attributes
#print(rfe.support_)
#print(rfe.ranking_)




