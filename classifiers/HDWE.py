from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import KFold
import numpy as np
from math import sqrt
import strlearn as sl

class HDWE(ClassifierMixin, BaseEnsemble):
    
    """
    References
    ----------
    .. [1] Wang, Haixun, et al. "Mining concept-drifting data streams using ensemble classifiers." Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining. 2003.
    .. [2] Cieslak, David A., et al. "Hellinger distance decision trees are robust and skew-insensitive." Data Mining and Knowledge Discovery 24.1 (2012): 136-158.
    """
    
    def __init__(self, base_estimator=None, n_estimators=10, n_splits=5, pred_type="soft"):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_splits = n_splits
        self.pred_type = pred_type
        self.candidate_scores = []
        self.weights_ = []
        self.hd_weights = []

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Train new estimator
        candidate = clone(self.base_estimator).fit(self.X_, self.y_)
            
        # Calculate its scores
        scores = np.zeros(self.n_splits)
        kf = KFold(n_splits=self.n_splits)
        for fold, (train, test) in enumerate(kf.split(X)):
            fold_candidate = clone(self.base_estimator).fit(self.X_[train], self.y_[train])
            scores[fold] = self.hellinger_distance(fold_candidate, self.X_[test], self.y_[test])
        
        # Save scores
        candidate_weight = np.mean(scores)
        
        # Calculate weights of current ensemble
        self.weights_ = [self.hellinger_distance(clf, self.X_, self.y_) for clf in self.ensemble_]

        # Add new model
        self.ensemble_.append(candidate)
        self.weights_.append(candidate_weight)

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            worst_idx = np.argmin(self.weights_)
            del self.ensemble_[worst_idx]
            del self.weights_[worst_idx]
            
        # Normalization of weights
        if sum(self.weights_) != 0:
            normalized_weights = [float(w)/sum(self.weights_) for w in self.weights_]  
            self.weights_ = normalized_weights
        else:
            mean_ = 1/len(self.weights_)
            self.weights_ = [mean_ for i in self.weights_]
            
        return self

    # Calculate Hellinger distance based on ref. [2] 
    def hellinger_distance(self, clf, X, y):
        # TPR - True Positive Rate (or sensitivity, recall, hit rate)
        tprate = sl.metrics.recall(y, clf.predict(X))
        # TNR - True Negative Rate (or specificity, selectivity)
        tnrate = sl.metrics.specificity(y, clf.predict(X))
        # FPR - False Positive Rate (or fall-out)
        fprate = 1 - tnrate 
        # FNR - False Negative Rate (or miss rate)
        fnrate = 1 - tprate
        # Calculate Hellinger distance
        if tprate > fnrate:
            hd = sqrt((sqrt(tprate)-sqrt(fprate))**2 + (sqrt(1-tprate)-sqrt(1-fprate))**2)
        else:
            hd = 0
        return hd

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])
    
    def predict(self, X):
        if self.pred_type == "soft":
            return self.predict_soft(X)
        elif self.pred_type == "hard":
            return self.predict_hard(X)

    # Prediction without calculated weights
    def predict_hard(self, X):
        """
        Predict classes for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]

    # Prediction (making decision) use Hellinger distance weights 
    def predict_soft(self, X):
        """
        Predict classes for X.
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
    
        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """
    
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")
    
        # Weight support before acumulation
        weighted_support = (
            self.ensemble_support_matrix(
                X) * np.array(self.weights_)[:, np.newaxis, np.newaxis]
        )
    
        # Acumulate supports
        acumulated_weighted_support = np.sum(weighted_support, axis=0)
        prediction = np.argmax(acumulated_weighted_support, axis=1)
    
        # Return prediction
        return self.classes_[prediction]