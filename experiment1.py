import strlearn as sl
import numpy as np
import os

from sklearn.naive_bayes import GaussianNB
from classifiers import HDWE


# List of classifiers, partial_fit() method is mandatory to use StreamGenerator
clfs = [
    sl.ensembles.AWE(GaussianNB()),
    HDWE(GaussianNB()),
]
clf_names = [
    "AWE",
    "HDWE",
]

# Declaration of the data stream with given parameters
n_streams = 10
# random_states = list(range(1000, 1000+n_streams*55, 55))
random_states = [1231, 1345, 1789] # testing
st_stream_weights = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]
d_stream_weights = [(2, 5, 0.9), (2, 5, 0.8), (2, 5, 0.7), (2, 5, 0.6)]
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
            metric_score = []
            stream_name = "stat_ir%s_rs%s" % (weights, random_state)
            # Generate stream
            stream = sl.streams.StreamGenerator(**concept_kwargs, random_state=random_state, weights=weights)
            # Initialize evaluator with given metrics - stream learn evaluator
            evaluator = sl.evaluators.TestThenTrain(metrics)
            evaluator.process(stream, clfs)
            scores = evaluator.scores
            # Shape of the score: 1st - classfier, 2nd - chunk, 3rd - metric. Every matrix is different classifier, every row is test chunks and every column is different metric
            # print(scores.shape)
            
            # Save scores to csv file in specific directories
            for i_metric, metric_name in enumerate(metric_names):
                for j_clf, clf_name in enumerate(clf_names):
                    metric_score = scores[j_clf, :, i_metric]
                    filename = "results/experiment1/metrics/gen/%s/%s/%s/%s.csv" % (drift, stream_name, metric_name, clf_name)
                    if not os.path.exists("results/experiment1/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name)):
                        os.makedirs("results/experiment1/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name))
                    np.savetxt(fname=filename, fmt="%f", X=metric_score)
        
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
            # print(scores.shape)
            
            # Save scores to csv file in specific directories
            for i_metric, metric_name in enumerate(metric_names):
                for j_clf, clf_name in enumerate(clf_names):
                    metric_score = scores[j_clf, :, i_metric]
                    filename = "results/experiment1/metrics/gen/%s/%s/%s/%s.csv" % (drift, stream_name, metric_name, clf_name)
                    if not os.path.exists("results/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name)):
                        os.makedirs("results/experiment1/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name))
                    np.savetxt(fname=filename, fmt="%f", X=metric_score)
