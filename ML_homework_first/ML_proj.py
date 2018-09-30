# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 08:38:13 2018

@author: Junk Chen
"""

# 1. Import lib
import numpy as np
import pandas as pd

# 2. read data
train_data = pd.read_csv('train1.csv')
test_data = pd.read_csv('test.csv')

# 3、数据预处理
df = pd.concat([train_data, test_data])

TARGET = df['TARGET']

df = df.drop('TARGET', axis=1)

# one-hot encoding
# label encoding
# nan_as_category = True
# zz = pd.get_dummies(df, dummy_na = nan_as_category)
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

df = one_hot_encoder(df, nan_as_category=True)

df1 = df[0]
df1 = df1.fillna(0)
# df1 = df1.fillna(np.mean) # for logistic regression

df1['TARGET'] = TARGET

# Impute  most frequent

# 4. 
train_df = df1[df1['TARGET'].notnull()]
test_df = df1[df1['TARGET'].isnull()]

# random forest, SVM
# LightGMB, XGBOSDT, CATBOOST

feats = [f for f in train_df.columns if f not in ['ID', 'TARGET']]

train_x = train_df[feats]
train_y = train_df['TARGET']

test_x = test_df.drop(['ID', 'TARGET'], axis=1)


# -----------------------------------------------
# Random forest classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10000,
                             class_weight='balanced',
                             verbose=1000,
                             max_depth=32,
                             oob_score=True,
                             warm_start=True,
                             random_state=1,
                             n_jobs=-1
        )

# Train data
clf.fit(train_x, train_y)

# Predict result
prediction = clf.predict(test_x)
pridiction_proba = clf.predict_proba(test_x)[:, 1]

submit_result = pd.DataFrame(np.array([test_df['ID'], pridiction_proba]).T, 
                             columns=['ID', 'TARGET'])
submit_result.to_csv('submit_result09.csv', index=False)

submit_result = pd.DataFrame({'TARGET':pridiction_proba})
submit_result['ID'] = test_df['ID']
submit_result = submit_result[['ID', 'TARGET']]

submit_result.to_csv('submit_result04.csv', index=False)


# -----------
# Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(train_x, train_y)

prediction_lr = lr.predict(test_x)
prediction_proba_lr = lr.predict_proba(test_x)[:, 1]

lr_result = pd.DataFrame({'ID' : test_df['ID'],
                          'TARGET' : prediction_proba_lr})
    
lr_result.to_csv('lr_result01.csv', index=False)


# ****************SVM******************************
from sklearn import svm
svc = svm.SVC(gamma='auto',
          kernel='rbf',
          class_weight='balanced',
          probability=True,
          verbose=True)
svc = svm.LinearSVC(class_weight='balanced',
                    verbose=100)

svc.fit(train_x, train_y)

prediction_svc = svc.predict(test_x)
prediction_proba_svc = svc.predict_proba(test_x)[:, 1]

result = pd.DataFrame({'ID' : test_df['ID'],
                       'TARGET' : prediction_proba_svc})
result.to_csv('svc_result02.csv', index=False)

# ********* naive bayes *************
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

clf.fit(train_x, train_y)

prediction_gnb = clf.predict(test_x)
prediction_proba_gnb = clf.predict_proba(test_x)[:, 1]

gnb_result = pd.DataFrame({'ID' : test_df['ID'],
                           'TARGET' : prediction_proba_gnb})

gnb_result.to_csv('gnb_result01.csv', index=False)



# 分类预测
def prediction(classifier, file_name):
    classifier.fit(train_x, train_y)
    prediction_proba = classifier.predict_proba(test_x)[:, 1]
    result = pd.DataFrame({'ID' : test_df['ID'],
                           'TARGET' : prediction_proba})
    result.to_csv(file_name+'.csv', index=False)

from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV()
prediction(clf, 'lrcv_result02')

clf_lr = LogisticRegression(penalty='l2',
                            tol=1e-4,
                            C=1.0,
                            class_weight='balanced',
                            solver='newton-cg', # newton-cg
                            max_iter=1000,
                            multi_class='multinomial', # 'ovr','multinomial'
                            warm_start=True,
                            verbose=100,
                            n_jobs=-1
        )
prediction(clf_lr, 'lr_result16')

from sklearn.tree import DecisionTreeClassifier
prediction(DecisionTreeClassifier(), 'dtc_result01')
