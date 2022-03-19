import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Reading the dataset in a dataframe using Pandas
df = pd.read_csv('chronic_kidney_disease.csv') 

# Gathering Info
print('\nBasic Info: ');
print(df.head(10)) 
print(df.info) # look at the info of "df"
print('\nInfo of numeric columns: ');
print(df.describe()) # takes numeric values only
# print(df.describe(include="all"))
print('\nInfo of all coumns: ');
print(df.columns)
print(df.dtypes)
print('\nSize Info: ');
print(df.shape)
# print(df['Property_Area'].value_counts())

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
print(df.head(5))
    
# Checking missing values in the dataset
print('\nMissing values info in the dataset: ');
#in pandas axis=0 represents rows (default) and axis=1 represents columns.
print(df.apply(lambda x: sum(x.isnull()), axis=0))

# Replacing missing values for numerical attributes
v1 = df["age"].astype("float").mean(axis = 0)
df["age"].replace(np.nan, v1, inplace = True)
df["age"] = df["age"].astype("float")

v2 = df["bp"].astype("float").mean(axis = 0)
df["bp"].replace(np.nan, v2, inplace = True)
df["bp"] = df["bp"].astype("float")

v10 = df["bgr"].astype("float").mean(axis = 0)
df["bgr"].replace(np.nan, v10, inplace = True)
df["bgr"] = df["bgr"].astype("float")

v11 = df["bu"].astype("float").mean(axis = 0)
df["bu"].replace(np.nan, v11, inplace = True)
df["bu"] = df["bu"].astype("float")

v12 = df["sc"].astype("float").mean(axis = 0)
df["sc"].replace(np.nan, v12, inplace = True)
df["sc"] = df["sc"].astype("float")

v13 = df["sod"].astype("float").mean(axis = 0)
df["sod"].replace(np.nan, v13, inplace = True)
df["sod"] = df["sod"].astype("float")

v14 = df["pot"].astype("float").mean(axis = 0)
df["pot"].replace(np.nan, v14, inplace = True)
df["pot"] = df["pot"].astype("float")

v15 = df["hemo"].astype("float").mean(axis = 0)
df["hemo"].replace(np.nan, v15, inplace = True)
df["hemo"] = df["hemo"].astype("float")

v16 = df["pcv"].astype("float").mean(axis = 0)
df["pcv"].replace(np.nan, v16, inplace = True)
df["pcv"] = df["pcv"].astype("float")

v17 = df["wbcc"].astype("float").mean(axis = 0)
df["wbcc"].replace(np.nan, v17, inplace = True)
df["wbcc"] = df["wbcc"].astype("float")

v18 = df["rbcc"].astype("float").mean(axis = 0)
df["rbcc"].replace(np.nan, v18, inplace = True)
df["rbcc"] = df["rbcc"].astype("float")

# Replacing missing values for categorical attributes
df['sg'].fillna('1.02',inplace=True)
df["sg"] = df["sg"].astype("float")
df['al'].fillna('0',inplace=True)
df['su'].fillna('0',inplace=True)
df['rbc'].fillna('normal',inplace=True)
df['pc'].fillna('normal',inplace=True)
df['pcc'].fillna('notpresent',inplace=True)
df['ba'].fillna('notpresent',inplace=True)
df['htn'].fillna('no',inplace=True)
df['dm'].fillna('no',inplace=True)
df['cad'].fillna('no',inplace=True)
df['appet'].fillna('good',inplace=True)
df['pe'].fillna('no',inplace=True)
df['ane'].fillna('no',inplace=True)

# Converting all categorical variables into numeric by encoding the categories
var_mod = ['al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','class']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

print('\nInfo once more: ');
print(df.dtypes) 

# Save to local machine
df.to_csv('ckd.csv')