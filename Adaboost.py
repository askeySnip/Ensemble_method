import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


'''caculate the error rate.'''
def get_error_rate(prediction, target):
    return sum(prediction != target) / float(len(prediction))

'''Adaboost Algorithm implementation'''
def adaboost_clf(x_train, y_train, x_test, y_test, iteration, clf):
    n_train, n_test = len(x_train), len(x_test)
    # init weights with the same value.
    w = np.ones(n_train) / n_train
    pred_train, pred_test = np.zeros(n_train), np.zeros(n_test)

    for i in range(iteration):
        # fit the base classifier
        clf.fit(x_train, y_train, sample_weight=w)
        pred_train_t = clf.predict(x_train)
        pred_test_t = clf.predict(x_test)
        # caculate the error prediction
        miss = [int(x) for x in (pred_train_t != y_train)]
        # caculate the error rate in Gm(x)
        e_m = np.dot(w, miss) * 1.0
        # Alpha
        alpha_m = 0.5 * np.log( (1 - e_m) / float(e_m) )
        # Dm'
        dm_help = [x if x==1 else -1 for x in miss]
        dm_ = np.dot(w, np.exp([float(x) * alpha_m for x in dm_help]))
        # Zm
        zm = sum(dm_)
        # weight(Dm)
        w = dm_ / zm

        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_t])]
        pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_t])]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    return pred_train, pred_test

'''Adaboost Classifier'''
class Adaboost_classifier:

    iteration = 50
    base_clf = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    alpha = []
    clf = []

    '''recive iteration & base classifier as parameter'''
    def __init__(self, iteration, base_clf):
        self.iteration = iteration
        self.base_clf = base_clf

    '''caculate the error rate.'''
    def get_error_rate(self, prediction, target):
        return sum(prediction != target) / float(len(prediction))

    '''Adaboost Algorithm train'''
    def fit(self, x, y):
        n = len(x)
        # init weights
        w = np.ones(n) / n

        for i in range(self.iteration):
            # fit the base classifier
            self.base_clf.fit(x, y, sample_weight=w)
            pred = self.base_clf.predict(x)
            # caculate the wrong prediction
            miss = [int(x) for x in (pred != y)]
            # caculate the error rate
            e_m = np.dot(w, miss) * 1.0
            # Alpha
            alpha_m = 0.5 * np.log( (1 - e_m) / float(e_m) )
            # Dm'
            dm_help = [x if x == 1 else -1 for x in miss]
            dm_ = np.dot(w, np.exp([float(x) * alpha_m for x in dm_help]))
            # Zm
            zm = sum(dm_)
            # append to clf
            self.clf.append(base_clf)
            self.alpha.append(alpha_m)
            # weight update
            w = dm_ / zm

    '''Predicition'''
    def predict(self, x):
        pred = np.zeros(len(x))
        for i in range(self.iteration):
            pred_i = self.clf[i].predict(x)
            pred = [sum(x) for x in zip(pred, [x * self.alpha[i] for x in pred_i])]
        pred = np.sign(pred)
        return pred

    '''Adaboost Algorithm implementation'''
    def adaboost_clf(self, x_train, y_train, x_test, y_test):
        n_train, n_test = len(x_train), len(x_test)
        # init weights with the same value.
        w = np.ones(n_train) / n_train
        pred_train, pred_test = np.zeros(n_train), np.zeros(n_test)

        for i in range(self.iteration):
            # fit the base classifier
            self.base_clf.fit(x_train, y_train, sample_weight=w)
            pred_train_t = self.base_clf.predict(x_train)
            pred_test_t = self.base_clf.predict(x_test)
            # caculate the error prediction
            miss = [int(x) for x in (pred_train_t != y_train)]
            # caculate the error rate in Gm(x)
            e_m = np.dot(w, miss) * 1.0
            # Alpha
            alpha_m = 0.5 * np.log( (1 - e_m) / float(e_m) )
            # Dm'
            dm_help = [x if x==1 else -1 for x in miss]
            dm_ = np.dot(w, np.exp([float(x) * alpha_m for x in dm_help]))
            # Zm
            zm = sum(dm_)
            # weight(Dm)
            w = dm_ / zm

            # Add to prediction
            pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_t])]
            pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_t])]

        pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
        return pred_train, pred_test
