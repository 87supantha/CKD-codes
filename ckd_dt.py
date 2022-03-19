import pandas as pd
import numpy as np
# Import models from scikit learn module:
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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

# Building Decision Tree model
outcome_var = 'class'
model = DecisionTreeClassifier()
predictor_var = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane']
#to see the feature importance matrix from which we’ll take the most important features
#featimp = pd.Series(model.feature_importances_,index=predictor_var).sort_values(ascending=False)
#print(featimp)
print("\nPerformance of Decision Tree model:~ ") 
classification_model(model, df, predictor_var, outcome_var)




