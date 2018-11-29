import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pickle


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
        dm_ = np.multiply(w, np.exp([float(x) * alpha_m for x in dm_help]))
        # Zm
        zm = sum(dm_)
        # weight(Dm)
        w = dm_ / zm

        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_t])]
        pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_t])]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    return pred_train, pred_test

'''Base Adaboost Classifier for 0-1 classifier'''
class Adaboost_classifier(object):

    '''Init'''
    def __init__(self, iteration, base_clf):
        self.iteration = iteration
        self.base_clf = base_clf
        self.aloha = []
        self.models = []

    '''Get the error rate of predicition'''
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
            dm_ = np.multiply(w, np.exp([float(x) * alpha_m for x in dm_help]))
            # Zm
            zm = sum(dm_)
            # append to clf
            self.models.append(pickle.dumps(self.base_clf))
            self.alpha.append(alpha_m)
            # weight update
            w = dm_ / zm

    '''Prediciton'''
    def predict(self, x):
        pred = np.zeros(len(x))
        for i in range(self.iteration):
            pred_i = pickle.loads(self.models[i]).predict(x)
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
            dm_ = np.multiply(w, np.exp([float(x) * alpha_m for x in dm_help]))
            # Zm
            zm = sum(dm_)
            # weight(Dm)
            w = dm_ / zm

            # Add to prediction
            pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_t])]
            pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_t])]

        pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
        return pred_train, pred_test

'''the paper adaboost'''
class Adaboost(object):

    def __init__(self, base_clf=None, iteration=100, target = 0.001, x_train = np.array([]), y_train = np.array([]),
                 x_test = np.array([]), y_test = np.array([])):
        self.base_clf = base_clf
        self.iteration = iteration
        self.target = target
        # Alpha
        self.Alpha = []
        # base classifiers
        self.clfs = []
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # weights
        n = len(self.x_train)
        self.weights = np.ones(n) / n
        self.bootstrap = range(0, len(self.x_train))

    '''the function to repreduct the set for train'''
    def _boostrap_sample(length):
        idx = np.random.randint(0, length, size=(length))
        return idx

    def _bagging_end(self, i):
        pass

    '''train iteration'''
    def _train_iteration(self, i):
        self._bagging_end(i)
        clf = self.base_clf()
        clf.fit(self.x_train[self.bootstrap], self.y_train[self.bootstrap])
        y_train_pred = clf.prediction(self.x_train[self.bootstrap])
        # preduct the miss array
        miss = [int(x) for x in (y_train_pred != self.y_train[self.bootstrap])]
        # caculate the error rate
        e_m = np.dot(w, miss) * 1.0

        # if error rate is bigger than 0.5. then redo bootstrap
        if e_m > 0.5:
            n = len(self.x_train)
            self.bootstrap = self._boostrap_sample(n)
            self.weights = np.ones(n) / n
            return

        # if error rate is too small then do bootstrap also to fit different data set
        elif e_m < 1e-5:
            self.Alpha.append(1e-10)
            n = len(self.x_train)
            self.bootstrap = self._boostrap_sample(n)
            self.weights = np.ones(n) / n

        else:
            # save Alpha
            alpha_m = 0.5 * np.log( (1 - e_m) / float(e_m) );
            self.Alpha.append(alpha_m)
            # Dm'
            dm_help = [x if x == 1 else -1 for x in miss]
            dm_ = np.multiply(w, np.exp([float(x) * alpha_m for x in dm_help]))
            # Zm
            zm = sum(dm_)
            self.weights = dm_ / zm

        self.clfs.append(clf)


    '''use vote algorithm to train'''
    def train(self):
        for i in range(self.iteration):
            self._train_iteration(i)
        result = []
        for i in range(len(self.x_test)):
            result.append([])
        for index, estimator in enumerate(self.clfs):
            y_test_result = estimator.predict(self.x_test)
            for index2, res in enumerate(result):
                res.append([y_test_result[index2], np.log(1/self.Alpha)])
        final_result = []

        # vote for the final result
        for res in result:
            dic = {}
            for r in res:
                dic[r[0]] = r[1] if not dic.has_key(r[0]) else dic.get(r[0]) + r[1]
            final_result.append(sorted(dic, key=lambda x:dic[x])[-1])
        print float(np.sum(final_result == self.y_test)) / len(self.y_test)

        return final_result


class MultiBoost(Adaboost):

    def __init__(self, base_clf=None, iteration=100, target = 0.001, x_train=np.array([]), y_train=np.array([]),
                 x_test = np.array([]), y_test = np.array([])):
        super(MultiBoost, self).__init__(base_clf, iteration, target, x_train, y_train, x_test, y_test)
        self.iterations = []
        self.current_iterat = 0;
        self._set_iterations()

    ''' set stop iteration'''
    def _set_iterations(self):
        n = int(float(self.iteration)**0.5)
        for i in range(n):
            self.iterations.append(int( ( (i+1) * self.iteration + n - 1) / n))
        for i in range(self.iteration):
            self.iterations.append(self.iteration)

    '''add the check iteration step'''
    def _bagging_end(self, i):
        if self.iterations[self.current_iterat] == i:
            n = len(self.x_train)
            self.bootstrap = self._boostrap_sample(n)
            self.weights = np.ones(n) / n
            self.current_iterat += 1
