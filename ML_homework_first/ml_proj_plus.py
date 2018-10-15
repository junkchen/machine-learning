# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:21:26 2018

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
# test_y = test_df['TARGET']

# 把数据集划分为训练数据集和测试数据集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, 
                                                    test_size=0.2)

'''
*****构造模型*****
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = []
models.append(('Random Forest', RandomForestClassifier(n_estimators=1000,
                             class_weight='balanced',
                             verbose=100,
                             max_depth=16,
                             oob_score=True,
                             warm_start=True,
                             random_state=1,
                             n_jobs=-1
        )))
models.append(('Logistic Regression', LogisticRegression(penalty='l2',
                            tol=1e-4,
                            C=1.0,
                            class_weight='balanced',
                            solver='newton-cg', # newton-cg
                            max_iter=1000,
                            multi_class='multinomial', # 'ovr','multinomial'
                            warm_start=True,
                            verbose=100,
                            n_jobs=-1
        )))
models.append(('Decision Tree', DecisionTreeClassifier()))


# 分别训练模型，并计算平均分
results = []
for name, model in models:
    model.fit(X_train, Y_train)
    results.append((name, model.score(X_test, Y_test)))
for i in range(len(results)):
    print('name: {}; score: {}'.format(results[i][0], results[i][1]))


# Cross validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

results = []
for name, model in models:
    kfold = KFold(n_splits=5)
    cv_result = cross_val_score(model, train_x, train_y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print('name: {}; cross val score: {}'.format(
            results[i][0], results[i][1].mean()))

name, model = models[0]
predict_result = model.predict_proba(test_x)[:, 1]
predict_result = []
for name, model in models:
    if(name == 'Random Forest'):
        predict_result = model.predict_proba(test_x)[:, 1]

# 模型训练及分析
clf = RandomForestClassifier(n_estimators=1000,
                             class_weight='balanced',
                             verbose=100,
                             max_depth=16,
                             oob_score=True,
                             warm_start=True,
                             random_state=1,
                             n_jobs=-1
        )
clf.fit(X_train, Y_train)
kfold = KFold(n_splits=5)
cv_result =cross_val_score(clf, train_x, train_y, cv=kfold)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)
print('train score: {}; test score: {}'.format(train_score, test_score))
prediction_proba = clf.predict_proba(test_x)[:, 1]
result = pd.DataFrame({'ID' : test_df['ID'],
                           'TARGET' : prediction_proba})
result.to_csv('rcf.csv', index=False)


# 特征选择
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=2)
X_new = selector.fit_transform(train_x, train_y)


# 超参数调优
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

params = {'n_estimators':[500, 100, 50, 10], 
          'max_depth':[3, 5, 1, 8, 16]}
rfc = RandomForestClassifier(verbose=100)
# clf = RandomizedSearchCV(rfc, params, cv=5)
clf = GridSearchCV(rfc, params, cv=5)
clf.fit(train_x, train_y)
cv_result = pd.DataFrame.from_dict(clf.cv_results_)

best_param = clf.best_params_
# best_score = clf.best_scores_
print(best_param)


# 
import lightgbm as lgb

selector = SelectKBest(k=30)
new_train_x = selector.fit_transform(train_x, train_y)

gbm = lgb.LGBMClassifier(n_estimators=500,
                         learning_rate=0.03,
                         max_depth=16)
gbm.fit(X_train, Y_train)
gbm.fit(train_x, train_y)
gbm.fit(new_train_x, train_y)
# kfold = KFold(n_splits=5)
# cross_val_score(gbm, train_x, train_y, cv=kfold)
train_score = gbm.score(X_train, Y_train)
test_score = gbm.score(X_test, Y_test)
print('train score: {}, test score: {}'.format(train_score, test_score))

prediction_proba = gbm.predict_proba(test_x)[:, 1]
result = pd.DataFrame({'ID' : test_df['ID'],
                           'TARGET' : prediction_proba})
result.to_csv('lgbm.csv', index=False)

# 
train_data = lgb.Dataset(X_train, label=Y_train)
test_data = lgb.Dataset(X_test, label=Y_test)
param = {
         'boosting_type': 'gbdt',
         'boosting': 'dart',
         'objective': 'binary',
         'metric': 'binary_logloss',
         
         'learning_rate': 0.01,
         'num_leaves':25,
         'max_depth':3,
         
         'max_bin':10,
         'min_data_in_leaf':8,
         
         'feature_fraction': 0.6,
         'bagging_fraction': 1,
         'bagging_freq':0,
         
         'lambda_l1': 0,
         'lambda_l2': 0,
         'min_split_gain': 0
}
num_round = 10
lgb.cv(param, train_data, num_round, nfold=5)
bst = lgb.train(param, train_data, num_round, valid_sets=test_data, 
          early_stopping_rounds=10)

train_score = bst.score(X_train, Y_train)
test_score = bst.score(X_test, Y_test)
print('train score: {}, test score: {}'.format(train_score, test_score))
br = bst.predict(X_test)
br = bst.predict(test_x)
br = br > .5
