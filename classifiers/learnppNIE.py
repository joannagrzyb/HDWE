from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from core import minority_majority_split, minority_majority_name
import math
import warnings
from sklearn.base import clone


class LearnppNIE(ClassifierMixin, BaseEstimator):

    """
    References
    ----------
    .. [1] Ditzler, Gregory, and Robi Polikar. "Incremental learning of
           concept drift from streaming imbalanced data." IEEE Transactions
           on Knowledge and Data Engineering 25.10 (2013): 2283-2301.
    """

    def __init__(self,
                 base_classifier=KNeighborsClassifier(),
                 number_of_classifiers=10,
                 param_a=2,
                 param_b=2):

        self.base_classifier = base_classifier
        self.number_of_classifiers = number_of_classifiers
        self.classifier_array = []
        self.classifier_weights = []
        self.sub_ensemble_array = []
        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.param_a = param_a
        self.param_b = param_b
        self.label_encoder = None
        self.iterator = 1

    def partial_fit(self, X, y, classes=None):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        y = self.label_encoder.transform(y)

        if self.minority_name is None or self.majority_name is None:
            self.minority_name, self.majority_name = minority_majority_name(y)

        self.sub_ensemble_array += [self._new_sub_ensemble(X, y)]

        beta_mean = self._calculate_weights(X, y)

        self.classifier_weights = []
        for b in beta_mean:
            self.classifier_weights.append(math.log(1/b))

        if len(self.sub_ensemble_array) >= self.number_of_classifiers:
            ind = np.argmax(beta_mean)
            del self.sub_ensemble_array[ind]
            del self.classifier_weights[ind]

        self.iterator += 1

    def _calculate_weights(self, X, y):
        beta = []
        for i in range(len(self.sub_ensemble_array)):
            epsilon = 1-metrics.f1_score(y, self._sub_ensemble_predict(i, X))
            if epsilon > 0.5:
                if i is len(self.sub_ensemble_array) - 1:
                    self.sub_ensemble_array[i] = self._new_sub_ensemble(X, y)
                    epsilon = 0.5
                else:
                    epsilon = 0.5
            beta.append(epsilon / float(1 - epsilon))

        sigma = []
        a = self.param_a
        b = self.param_b
        t = len(self.sub_ensemble_array)
        k = np.array(range(t))

        sigma = 1/(1 + np.exp(-a*(t-k-b)))

        sigma_mean = []
        for k in range(t):
            sigma_sum = 0
            for j in range(t-k):
                sigma_sum += sigma[j]
            sigma_mean.append(sigma[k]/sigma_sum)

        beta_mean = []
        for k in range(t):
            beta_sum = 0
            for j in range(t-k):
                beta_sum += sigma_mean[j]*beta[j]
            beta_mean.append(beta_sum)

        return beta_mean

    def predict(self, X):
        predictions = np.asarray([self._sub_ensemble_predict(i, X) for i in range(len(self.classifier_weights))]).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.classifier_weights)), axis=1, arr=predictions)
        maj = self.label_encoder.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        probas_ = [self._sub_ensemble_predict_proba(i, X) for i in range(len(self.classifier_weights))]
        return np.average(probas_, axis=0, weights=self.classifier_weights)

    def _new_sub_ensemble(self, X, y):
        y = np.array(y)
        X = np.array(X)

        minority, majority = minority_majority_split(X, y,
                                                     self.minority_name,
                                                     self.majority_name)

        T = self.number_of_classifiers
        N = len(X)
        sub_ensemble = []
        for k in range(T):
            number_of_instances = int(math.floor(N/float(T)))
            df = pd.DataFrame(majority)
            sample = df.sample(number_of_instances, replace=True)
            res_X = np.concatenate((sample, minority), axis=0)
            res_y = len(sample)*[self.majority_name] + len(minority)*[self.minority_name]
            new_classifier = clone(self.base_classifier).fit(res_X, res_y)
            sub_ensemble += [new_classifier]
        return sub_ensemble

    def _sub_ensemble_predict_proba(self, i, X):
        probas_ = [clf.predict_proba(X) for clf in self.sub_ensemble_array[i]]
        return np.average(probas_, axis=0)

    def _sub_ensemble_predict(self, i, X):
        predictions = np.asarray([clf.predict(X) for clf in self.sub_ensemble_array[i]]).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                  axis=1,
                                  arr=predictions)
        return maj
