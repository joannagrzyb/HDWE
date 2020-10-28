import strlearn as sl
import numpy as np
import os
import warnings

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from classifiers import HDDT
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from classifiers import HDWE

warnings.filterwarnings("ignore")

clfs = [
    HDWE(GaussianNB(), pred_type="hard"),
    HDWE(MLPClassifier(hidden_layer_sizes=(10)), pred_type="hard"),
    HDWE(DecisionTreeClassifier(random_state=20), pred_type="hard"),
    HDWE(HDDT(), pred_type="hard"),
    HDWE(KNeighborsClassifier(), pred_type="hard"),
    # HDWE(SVC(probability=True), pred_type="hard"),
    # sl.ensembles.SEA(GaussianNB()) 
]
clf_names = [
    "HDWE-GNB", # DONE
    "HDWE-MLP", # DONE
    "HDWE-CART", # DONE
    "HDWE-HDDT",
    "HDWE-KNN", # DONE
    # "HDWE-SVC", # DONE od 3% i 1% stat, dla 1% problem z 1 klasą
    # "SEA" # Only for test strlearn
]

# Declaration of the data stream with given parameters
random_states = [1111, 1234, 1567]
# random_states = [123] # testing
st_stream_weights = [
    [0.01, 0.99], 
    [0.03, 0.97], 
    [0.05, 0.95], 
    [0.1, 0.9], 
    [0.15, 0.85], 
    [0.2, 0.8], 
    [0.25, 0.75]
]
d_stream_weights = [
    (2, 5, 0.99), 
    (2, 5, 0.97), 
    (2, 5, 0.95), 
    (2, 5, 0.9), 
    (2, 5, 0.85), 
    (2, 5, 0.8), 
    (2, 5, 0.75)
]
concept_kwargs = {
    "n_chunks": 200, # 200
    "chunk_size": 500,
    "n_classes": 2,
    "n_drifts": 5,
    "n_features": 20,
    "n_informative": 15,
    "n_redundant": 5,
    "n_repeated": 0,
}

metrics = [
    sl.metrics.specificity, 
    sl.metrics.recall, 
    sl.metrics.precision,
    sl.metrics.f1_score, 
    sl.metrics.balanced_accuracy_score, 
    sl.metrics.geometric_mean_score_1, 
    sl.metrics.geometric_mean_score_2
]     
metric_names = [
    "specificity",
    "recall",
    "precision",
    "f1_score",
    "balanced_accuracy_score",
    "geometric_mean_score_1",
    "geometric_mean_score_2",
]
drifts = ['sudden', 'incremental']


for drift in drifts:
    if drift == 'incremental':
        concept_kwargs["incremental"] = True
        concept_kwargs["concept_sigmoid_spacing"] = 5
        
    # Loop for experiment for stationary imbalanced streams
    for weights in st_stream_weights:
        for random_state in random_states:
            one_class = False
            metric_score = []
            stream_name = "stat_ir%s_rs%s" % (weights, random_state)
            # Generate stream
            stream = sl.streams.StreamGenerator(**concept_kwargs, random_state=random_state, weights=weights)
            
            # Initialize evaluator with given metrics - stream learn evaluator
            evaluator = sl.evaluators.TestThenTrain(metrics)
            evaluator.process(stream, clfs)
            scores = evaluator.scores
            # Shape of the score: 1st - classfier, 2nd - chunk, 3rd - metric. Every matrix is different classifier, every row is test chunks and every column is different metric            
            
            # Save scores to csv file in specific directories
            for i_metric, metric_name in enumerate(metric_names):
                for j_clf, clf_name in enumerate(clf_names):
                    metric_score = scores[j_clf, :, i_metric]
                    filename = "results/experiment2/metrics/gen/%s/%s/%s/%s.csv" % (drift, stream_name, metric_name, clf_name)
                    if not os.path.exists("results/experiment2/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name)):
                        os.makedirs("results/experiment2/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name))
                    np.savetxt(fname=filename, fmt="%f", X=metric_score)
                    
                    print("DONE: %s, %s, %s, %s" % (drift, stream_name, metric_name, clf_name))
        
    # Loop for experiment for dynamically imbalanced streams
    for weights in d_stream_weights:
        for random_state in random_states:
            metric_score = []
            stream_name = "d_ir%s_rs%s" % (weights, random_state)
            # Generate stream
            stream = sl.streams.StreamGenerator(**concept_kwargs, random_state=random_state, weights=weights)
                        
            # Initialize evaluator with given metrics - stream learn evaluator
            evaluator = sl.evaluators.TestThenTrain(metrics)
            evaluator.process(stream, clfs)
            scores = evaluator.scores
            # Shape of the score: 1st - classfier, 2nd - chunk, 3rd - metric. Every matrix is different classifier, every row is test chunks and every column is different metric
            
            # Save scores to csv file in specific directories
            for i_metric, metric_name in enumerate(metric_names):
                for j_clf, clf_name in enumerate(clf_names):
                    metric_score = scores[j_clf, :, i_metric]
                    filename = "results/experiment2/metrics/gen/%s/%s/%s/%s.csv" % (drift, stream_name, metric_name, clf_name)
                    if not os.path.exists("results/experiment2/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name)):
                        os.makedirs("results/experiment2/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name))
                    np.savetxt(fname=filename, fmt="%f", X=metric_score)
                    
                    print("DONE: %s, %s, %s, %s" % (drift, stream_name, metric_name, clf_name))
