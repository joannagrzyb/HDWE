import strlearn as sl
import numpy as np
import os

from sklearn.base import clone
from sklearn.svm import SVC
from classifiers import HDWE
from classifiers import LearnppCDS
from classifiers import LearnppNIE
from classifiers import OUSE
from classifiers import REA

from joblib import Parallel, delayed
import logging
import traceback
from time import time
import warnings

clfs = [
    HDWE(SVC(probability=True), pred_type="hard"), 
    sl.ensembles.SEA(SVC(probability=True)), 
    sl.ensembles.AWE(SVC(probability=True)), 
    LearnppCDS(SVC(probability=True)), 
    LearnppNIE(SVC(probability=True)), 
    OUSE(SVC(probability=True)), 
    REA(SVC(probability=True)), 
]
clf_names = [
    "HDWE-SVC",
    "SEA-SVC",
    "AWE-SVC",
    "LearnppCDS-SVC",
    "LearnppNIE-SVC",
    "OUSE-SVC",
    "REA-SVC",
]

# Declaration of the data stream with given parameters
random_states = [1111, 1234, 1567]
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
    "n_chunks": 200,
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

logging.basicConfig(filename='experiment3a.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
logging.info("--------------------------------------------------------------------------------")
logging.info("-------                        NEW EXPERIMENT                            -------")
logging.info("--------------------------------------------------------------------------------")

def compute(clf_name, clf, metric_names, metrics, random_state, weights, drift):

    logging.basicConfig(filename='experiment3a.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')

    try:
        warnings.filterwarnings("ignore")

        metric_score = []
        if drift == 'incremental':
            concept_kwargs["incremental"] = True
            concept_kwargs["concept_sigmoid_spacing"] = 5

        if len(weights) == 2:
            # Stationary imbalance ratio
            stream_name = "stat_ir%s_rs%s" % (weights, random_state)
        elif len(weights) == 3:
            # Dynamically imbalance ratio
            stream_name = "d_ir%s_rs%s" % (weights, random_state)
        else:
            raise ValueError("Bad value in weight error")

        print("START: %s, %s, %s" % (drift, stream_name, clf_name))
        logging.info("START - %s, %s, %s" % (drift, stream_name,  clf_name))
        start = time()

        # Generate stream
        stream = sl.streams.StreamGenerator(**concept_kwargs, random_state=random_state, weights=weights)

        # Initialize evaluator with given metrics - stream learn evaluator
        evaluator = sl.evaluators.TestThenTrain(metrics)
        evaluator.process(stream, clone(clf))
        # Shape of the score: 1st - classfier, 2nd - chunk, 3rd - metric. Every matrix is different classifier, every row is test chunks and every column is different metric

        # Save scores to csv file in specific directories
        for i_metric, metric_name in enumerate(metric_names):
                metric_score = evaluator.scores[0, :, i_metric]
                filename = "results/experiment3a/metrics/gen/%s/%s/%s/%s.csv" % (drift, stream_name, metric_name, clf_name)
                if not os.path.exists("results/experiment3a/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name)):
                    os.makedirs("results/experiment3a/metrics/gen/%s/%s/%s/" % (drift, stream_name, metric_name))
                np.savetxt(fname=filename, fmt="%f", X=metric_score)

        end = time()-start

        print("DONE: %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, end))
        logging.info("DONE - %s, %s, %s (Time: %d [s])" % (drift, stream_name, clf_name, end))

    except Exception as ex:
        logging.exception("Exception in %s, %s, %s" % (drift, stream_name, clf_name))
        print("ERROR: %s, %s, %s" % (drift, stream_name, clf_name))
        traceback.print_exc()
        print(str(ex))
        
# Multithread; n_jobs - number of threads, where -1 all threads, safe for my computer 2
Parallel(n_jobs=-1)(delayed(compute)
                    (clf_name, clf, metric_names, metrics, random_state, weights, drift)
                    for clf_name, clf in zip(clf_names, clfs)
                    for drift in drifts
                    for weights in st_stream_weights+d_stream_weights
                    for random_state in random_states
                    )
