# coding=utf-8
import warnings

import numpy as np
from deepforest import CascadeForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def logistic_regression_classifier(train_X, train_y):
    """
    Logistic Regression model
    :param train_X: features
    :param train_y: labels
    :return: Logistic Regression model
    """
    model = LogisticRegression()
    model.fit(train_X, train_y)
    return model


def random_forest_classifier(train_X, train_y):
    """
    Random Forest model
    :param train_X: features
    :param train_y: labels
    :return: Random Forest model
    """
    model = RandomForestClassifier()
    model.fit(train_X, train_y)
    return model


def decision_tree_classifier(train_X, train_y):
    """
    Decision Tree model
    :param train_X: features
    :param train_y: labels
    :return: Decision Tree model
    """
    model = tree.DecisionTreeClassifier()
    model.fit(train_X, train_y)
    return model


def naive_bayes_classifier(train_X, train_y):
    """
    Naive Bayes model
    :param train_X: features
    :param train_y: labels
    :return: Naive Bayes model
    """
    model = GaussianNB()
    model.fit(train_X, train_y)
    return model


def adaboost(training_data_x, training_data_y):
    """
    AdaBoost model
    :param training_data_x: features
    :param training_data_y: labels
    :return: AdaBoost model
    """
    model = AdaBoostClassifier()
    model.fit(training_data_x, training_data_y)
    return model


def xgboost(training_data_x, training_data_y):
    """
    XGBoost model
    :param training_data_x: features
    :param training_data_y: labels
    :return: XGBoost model
    """
    model = XGBClassifier()
    model.fit(training_data_x, training_data_y)
    return model


def deep_forest(training_data_x, training_data_y):
    """
    Deep Forest model
    :param training_data_x: features
    :param training_data_y: labels
    :return: Deep Forest model
    """
    training_data_x = np.array(training_data_x)
    training_data_y = np.array(training_data_y)
    model = CascadeForestClassifier(random_state=1, n_jobs=1)
    model.fit(training_data_x, training_data_y)
    return model


def knn(training_data_x, training_data_y):
    """
    KNN model
    :param training_data_x: features
    :param training_data_y: labels
    :return: KNN model
    """
    training_data_x = np.array(training_data_x)
    training_data_y = np.array(training_data_y)
    model = KNeighborsClassifier(10, weights="uniform")
    model.fit(training_data_x, training_data_y)
    return model
