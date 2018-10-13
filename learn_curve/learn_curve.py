# -*- coding: utf-8 -*-
"""
Learn curve demo
"""

import numpy as np
import matplotlib.pyplot as plt

# 准备数据
n_dots = 200

X = np.linspace(0, 1, n_dots)
Y = np.sqrt(X) + 0.2*np.random.rand(n_dots) - 0.1

# 因为 sklearn 的接口里，需要用到 n_sample x n_feature 的矩阵
# 所以需要转换为 200 x 1 的矩阵
x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)

# 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression()
    # 这是一个流水线，先增加多项式阶数，然后再用线性回归算法来拟合数据
    pipeline = Pipeline([('polynomial_features', polynomial_features),
                         ('linear_regression', linear_regression)])
    return pipeline


# 
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean,+ train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean,+ test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', 
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')
    
    plt.legend(loc='best')
    return plt

# 为了让学习曲线更平滑，计算 10 次交叉验证数据集的分数
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
titles = ['Learning Curves (Under Fitting)',
          'Learning Curves',
          'Learning Curves (Over Fitting)']
degrees = [1, 3, 10]

plt.figure(figsize=(18, 4), dpi=200)
for i in range(len(degrees)):
    plt.subplot(1, 3, i+1)
    plot_learning_curve(polynomial_model(degrees[i]), titles[i],
                        x, y, ylim=(0.75, 1.01), cv=cv)

plt.show()
