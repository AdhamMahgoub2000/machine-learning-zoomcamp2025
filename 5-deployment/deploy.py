import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


output_file = 'model.bin'
# Load data

df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv')

# Data preparation

categorical= df.dtypes[df.dtypes == 'object'].index.to_list()
numerical  = df.dtypes[df.dtypes != 'object'].index.to_list()
numerical.remove('converted')

df[numerical]   = df[numerical].fillna(0)
df[categorical] = df[categorical].fillna('NA')


# Train-validation-test split

full_train,df_test = train_test_split(df        ,test_size=0.2  , random_state=1)
df_train  ,df_val  = train_test_split(full_train,test_size= 0.25, random_state=1)


# Model training and prediction

def train(df_train, y_train,C=1):
    dicts_train= df_train[categorical+ numerical].to_dict(orient='records')
    dv         = DictVectorizer(sparse = False)
    X_train    = dv.fit_transform(dicts_train)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train,y_train)
    return model , dv

def predict(df_val, model, dv):
    dicts_val= df_val[categorical+ numerical].to_dict(orient='records')
    X_val    =  dv.transform(dicts_val)

    y_pred   = model.predict_proba(X_val)[:, 1]
    return y_pred
    

y_train  = df_train['converted'].values
y_val  = df_val['converted'].values

model,dv = train(df_train,y_train)

y_pred = predict(df_val,model,dv)

print('ROC AUC:', roc_auc_score(y_val,y_pred))


with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)
