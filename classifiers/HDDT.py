from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from math import sqrt
import operator


class HDDT(BaseEstimator, ClassifierMixin):
    """
    HDDT (Hellinger Distance Decision Tree).
    References
    ----------
    ..[1]   David A. Cieslak, T. Ryan Hoens, Nitesh V. Chawla and W. Philip Kegelmeyer, Data Min Knowl Disc 2011
            https://www3.nd.edu/~dial/papers/DMKD11.pdf
    ..[2]   Learning Decision Trees for Unbalanced Data, David A. Cieslak and Nitesh V. Chawla, ECML 2008
            https://www3.nd.edu/~dial/papers/ECML08.pdf
    ..[3]   Implementation of the HDDT in R programming language
            https://github.com/kaurao/HDDT
    """

    def __init__(self, C=2):
        # TODO: grid search - for the best C
        self.C = C

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        if type(y) is not np.ndarray:
            y = np.array(y)
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        self.labels = np.unique(y)

        # Check if is continuous or discrete
        self.n_features = X.shape[1]
        self.features_type = []
        for f in range(self.n_features):
            if isinstance(X[0,f], float):
                self.features_type.append("continuous")
            else:
                self.features_type.append("discrete")

        self.root = self.HDDT_func(X, y, self.C)

        return self

    def predict(self, X):
        n_row = X.shape[0]
        y = np.full(n_row, -1)
        for i in range(n_row):
            # search the tree until we find a leaf node
            node = self.root
            while isinstance(node.get("v"), (int, float)):
                if node["type"] == "discrete":
                    if X[i,node["i"]] == node["v"]:
                        node = node["childLeft"]
                    else:
                        node = node["childRight"]

                elif node["type"] == "continuous":
                    if X[i,node["i"]] <= node["v"]:
                        node = node["childLeft"]
                    else:
                        node = node["childRight"]
            y[i] = node["label"]
        # Return prediction classes, for example 0 or 1
        return y

    def predict_proba(self, X):
        n_row = X.shape[0]
        probas = np.full((n_row,2), -1)
        for i in range(n_row):
            # search the tree until we find a leaf node
            node = self.root
            while isinstance(node.get("v"), (int, float)):
                if node["type"] == "discrete":
                    if X[i,node["i"]] == node["v"]:
                        node = node["childLeft"]
                    else:
                        node = node["childRight"]

                elif node["type"] == "continuous":
                    if X[i,node["i"]] <= node["v"]:
                        node = node["childLeft"]
                    else:
                        node = node["childRight"]
            probas[i][0] = node["proba"][0]
            probas[i][1] = node["proba"][1]
        # Return prediction probabilities of occurrence class 0 or 1, for example 0,238 and 0,762
        return probas

    def HDDT_func(self, X, y, C):
        """
        HDDT function to create and use Hellinger distance decision tree (HDDT). It is a recursive function that calls itself with subsets of training data that matches the decision criterion using a list to create the tree structure.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples, )
            Classes/labels
        C : integer
            Minimum size of the training set at a node to attempt a split. Size of cut-off.
        labels : array-like, optional
            The default is None.

        Returns
        -------
        node : tuple
            The root node of the deicison tree
        """

        # Node of the tree. When called for the first time, this will be the root
        node = {}
        node["C"] = C
        node["labels"] = self.labels

        # If there is only one class, this is leaf
        if len(np.unique(y)) == 1:
            node["label"] = y[0]
            if y[0] == 0:
                node["proba"] = [1, 0]
            else:
                node["proba"] = [0, 1]
            return node

        # If y is smaller than minimum size of the training set, this is leaf
        elif len(y) < C:
            # Count number of samples of every class
            classes, counts = np.unique(y, return_counts=True)

            # Use Laplace smoothing, by adding 1 to count of each class
            counts[0] += 1
            counts[1] += 1

            # Count probabilities of every class
            p = counts[0] / len(y)
            if classes[0] == 0:
                node["proba"] = [p, 1-p]
            else:
                node["proba"] = [1-p, p]

            # Return label of that class(0 or 1), where there is more samples
            if counts[0] > counts[1]:
                node["label"] = classes[0]
            else:
                node["label"] = classes[1]
            return node

        # this is node
        else:
            # HD is 2D-array, it contains Hellinger Distance, value (place of split) and type (discrete or continuous)
            HD = []
            # hd contains only Hellinger Distance
            hd = []
            # Use function HDDT_dist in a recursive way
            for i in range(self.n_features):
                hel_dist = self.HDDT_dist(X[:,i], y, self.features_type[i])
                HD.append(hel_dist)
                hd.append(hel_dist[0])

            i  = np.argmax(hd)

            # Save node attributes
            node["i"] = i       # feature
            node["d"] = HD[i][0]
            node["v"] = HD[i][1]
            node["type"] = HD[i][2]

            if node["type"] == "discrete":
                # j contains True and False values, True is when sample in X is equal v
                j = np.array((X[:,i] == node["v"]))
                node["childLeft"] = self.HDDT_func(X[j,:], y[j], C)
                opposite_j = [operator.not_(value_j) for value_j in j]
                node["childRight"] = self.HDDT_func(X[opposite_j,:], y[opposite_j], C)

            elif node["type"] == "continuous":
                # j contains True and False values, True is when sample in X is lower or equal v
                j = np.array((X[:,i] <= node["v"]))
                node["childLeft"] = self.HDDT_func(X[j,:], y[j], C)
                opposite_j = [operator.not_(value_j) for value_j in j]
                node["childRight"] = self.HDDT_func(X[opposite_j,:], y[opposite_j], C)

        return node

    def HDDT_dist(self, f, y, f_type):

        """
        Calculate Hellinger distance for a given feature vector.
        Attributes can be discrete or continuous.
        It returns Hellinger distance, "value" of the feature that is used as decision criterion (splitting) and type of feature (discrete or continuous).
        It works ONLY for binary labels.
        """

        # Count number of samples of every class
        classes, counts = np.unique(y, return_counts=True)

        hellinger = -1
        # Number of samples for each class
        T0 = counts[0]
        T1 = counts[1]
        val = 0

        cl = f_type

        # Check if the feature is discrete or continuous
        if cl == "discrete":
            for v in np.unique(f):

                # Number of class 0 and class1 of value v in f (features)
                v_index = np.argwhere(f==v).ravel()
                Tfv1 = len(np.argwhere(y[v_index]==classes[1]).ravel())
                Tfv0 = len(np.argwhere(y[v_index]==classes[0]).ravel())

                Tfw1 = T1 - Tfv1
                Tfw0 = T0 - Tfv0

                # Calculate Hellinger distance
                cur_value = (sqrt(Tfv1/T1) - sqrt(Tfv0/T0))**2 + (sqrt(Tfw1/T1) - sqrt(Tfw0/T0))**2

                if cur_value > hellinger:
                    hellinger = cur_value
                    val = v

        elif cl == "continuous":
            for v in np.unique(f):
                v_index = np.argwhere(f<=v).ravel()
                Tfv1 = len(np.argwhere(y[v_index]==classes[1]).ravel())
                Tfv0 = len(np.argwhere(y[v_index]==classes[0]).ravel())
                Tfw1 = T1 - Tfv1
                Tfw0 = T0 - Tfv0

                # Calculate Hellinger distance
                cur_value = (sqrt(Tfv1/T1) - sqrt(Tfv0/T0))**2 + (sqrt(Tfw1/T1) - sqrt(Tfw0/T0))**2

                if cur_value > hellinger:
                    hellinger = cur_value
                    val = v
        # return 3 values in tuple (Hellinger distance, value of split, type of the feature)
        return (sqrt(hellinger), val, cl)
