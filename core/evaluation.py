import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import warnings
warnings.simplefilter("ignore")


class Evaluation():

    def __init__(self, classifier, stream_name, method_name, experiment_name, tqdm=True):
        self.__classifier = classifier
        self.__stream_name = stream_name
        self.__method_name = method_name
        self.__experiment_name = experiment_name
        self.__tqdm = tqdm

        self.__data = None
        self.__classes = None
        self.__X = []
        self.__true_y = []
        self.__predict_y = []
        self.__predict_probas = []
        self.__tn = []
        self.__fp = []
        self.__fn = []
        self.__tp = []

        self.__step_size = None

    def test_and_train(self, data, classes, steps=20, initial_steps=1, initial_size=None, step_size=None, online=False):

        if step_size is None:
            self.__step_size = len(data)/steps
            self.__steps = steps
        else:
            self.__step_size = step_size
            self.__steps = int(len(data)/step_size)

        if initial_size is None or initial_size == 0:
            self.__initial_size = self.__step_size*initial_steps
            self.__initial_steps = initial_steps
        else:
            self.__initial_size = initial_size
            self.__initial_steps = int(initial_size/self.__step_size)

        initial_data = data[0:self.__initial_size]
        X, y, classes_ = self.prepare_data(initial_data)
        self.__classes = classes

        if online:
            self.__classifier.fit(X, y, self.__classes)

            for i in tqdm(range(self.__initial_size, len(data)), desc=self.__method_name):
                X, y, c = self.prepare_data(data[i:(i+1)])

                predict = self.__classifier.predict(X)
                self.__classifier.partial_fit(X, y, self.__classes)

                self.__gather_data(X, y, predict)

            return self.__classifier

        else:
            self.__classifier.partial_fit(X, y, self.__classes)

            if(self.__tqdm):
                for i in tqdm(range(self.__initial_steps, self.__steps), desc=self.__method_name):
                    chunk = data[(i*self.__step_size):((i+1)*self.__step_size)]
                    X, y, c = self.prepare_data(chunk)

                    predict = self.__classifier.predict(X)
                    self.__classifier.partial_fit(X, y, self.__classes)
                    self.__gather_data(X, y, predict)
            else:
                for i in range(self.__initial_steps, self.__steps):
                    chunk = data[(i*self.__step_size):((i+1)*self.__step_size)]
                    X, y, c = self.prepare_data(chunk)

                    predict = self.__classifier.predict(X)
                    self.__classifier.partial_fit(X, y, self.__classes)
                    self.__gather_data(X, y, predict)

            return self.__classifier

    def __gather_data(self, X, y_true, y_pred):
        self.__X.extend(X)
        self.__true_y.extend(y_true)
        self.__predict_y.extend(y_pred)

        y_pred[y_pred == "positive"] = "1"
        y_pred[y_pred == "negative"] = "0"
        y_pred = y_pred.astype(int)

        y_true[y_true == "positive"] = "1"
        y_true[y_true == "negative"] = "0"
        y_true = y_true.astype(int)

        P = y_true == 1
        N = y_true == 0

        tp = np.sum(y_pred[P] == 1)
        fp = np.sum(y_pred[N] == 1)
        tn = np.sum(y_pred[N] == 0)
        fn = np.sum(y_pred[P] == 0)

        self.__tn.append(tn)
        self.__fp.append(fp)
        self.__fn.append(fn)
        self.__tp.append(tp)

    def save_to_csv(self, filename=None):
        if filename is None:
            filename = "results/raw_preds/%s/%s.csv" % (self.__stream_name, self.__method_name)

        if not os.path.exists("results/raw_preds/%s/" % self.__stream_name):
            os.makedirs("results/raw_preds/%s/" % self.__stream_name)

        df = pd.DataFrame(data=[
                                self.__true_y,
                                self.__predict_y
                               ])
        df = df.T
        df.to_csv(filename, header=False)

    def save_to_csv_confmat(self, filename=None):
        if filename is None:
            filename = "results/raw_conf/%s/%s/%s.csv" % (self.__experiment_name, self.__stream_name, self.__method_name)

        if not os.path.exists("results/raw_conf/%s/%s/" % (self.__experiment_name, self.__stream_name)):
            os.makedirs("results/raw_conf/%s/%s/" % (self.__experiment_name, self.__stream_name))

        df = pd.DataFrame(data=[
                                self.__tn,
                                self.__fp,
                                self.__fn,
                                self.__tp
                               ])
        df = df.T
        df.to_csv(filename, header=False)

    def prepare_data(self, data):
        df = pd.DataFrame(data)
        features = df.iloc[:, 0:-1].values.astype(float)
        labels = df.iloc[:, -1].values.astype(str)
        classes = np.unique(labels)
        return features, labels, classes
