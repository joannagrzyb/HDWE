import strlearn as sl
import numpy as np
import os

from sklearn.base import clone
from classifiers import HDDT
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
    HDWE(HDDT(), pred_type="hard"),
    sl.ensembles.SEA(HDDT()),
    sl.ensembles.AWE(HDDT()),
    LearnppCDS(HDDT()),
    LearnppNIE(HDDT()),
    OUSE(HDDT()),
    REA(HDDT()),
    HDWE(SVC(probability=True), pred_type="hard"),
    sl.ensembles.SEA(SVC(probability=True)),
    sl.ensembles.AWE(SVC(probability=True)),
    LearnppCDS(SVC(probability=True)),
    LearnppNIE(SVC(probability=True)),
    OUSE(SVC(probability=True)),
    REA(SVC(probability=True)),
]
clf_names = [
    "HDWE-HDDT",
    "SEA-HDDT",
    "AWE-HDDT",
    "LearnppCDS-HDDT",
    "LearnppNIE-HDDT",
    "OUSE-HDDT",
    "REA-HDDT",
    "HDWE-SVC",
    "SEA-SVC",
    "AWE-SVC",
    "LearnppCDS-SVC",
    "LearnppNIE-SVC",
    "OUSE-SVC",
    "REA-SVC",
]

# Loading of the real data stream
streams = []                                                        # size   n_chunks
# streams.append(("real-data/covtypeNorm-1-2vsAll-pruned.arff",    2000,    int(267000/2000)))
# streams.append(("real-data/poker-lsn-1-2vsAll-pruned.arff",      2000,    int(359999/2000)))
streams.append(("real-data/2vsA_INSECTS-abrupt_imbalanced_norm.arff",      2000,    int(355274/2000)))


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

logging.basicConfig(filename='experiment_real.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
logging.info("--------------------------------------------------------------------------------")
logging.info("-------                        NEW EXPERIMENT                            -------")
logging.info("--------------------------------------------------------------------------------")

def compute(clf_name, clf, chunk_size, n_chunks, metric_names, metrics, streams, stream_name):

    logging.basicConfig(filename='experiment_real.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')

    try:
        warnings.filterwarnings("ignore")

        metric_score = []

        print("START: %s, %s" % (stream_name, clf_name))
        logging.info("START - %s, %s" % (stream_name,  clf_name))
        start = time()

        stream = sl.streams.ARFFParser(stream_name, chunk_size, n_chunks)

        # Initialize evaluator with given metrics - stream learn evaluator
        evaluator = sl.evaluators.TestThenTrain(metrics)
        evaluator.process(stream, clone(clf))
        # Shape of the score: 1st - classfier, 2nd - chunk, 3rd - metric. Every matrix is different classifier, every row is test chunks and every column is different metric

        stream_name = stream_name.split("/")[-1]
        # Save scores to csv file in specific directories
        for i_metric, metric_name in enumerate(metric_names):
                metric_score = evaluator.scores[0, :, i_metric]
                filename = "results/experiment_real/metrics/%s/%s/%s.csv" % (stream_name, metric_name, clf_name)
                if not os.path.exists("results/experiment_real/metrics/%s/%s/" % (stream_name, metric_name)):
                    os.makedirs("results/experiment_real/metrics/%s/%s/" % (stream_name, metric_name))
                np.savetxt(fname=filename, fmt="%f", X=metric_score)

        end = time()-start

        print("DONE: %s, %s (Time: %d [s])" % (stream_name, clf_name, end))
        logging.info("DONE - %s, %s (Time: %d [s])" % (stream_name, clf_name, end))

    except Exception as ex:
        logging.exception("Exception in %s, %s" % (stream_name, clf_name))
        print("ERROR: %s, %s" % (stream_name, clf_name))
        traceback.print_exc()
        print(str(ex))

# Multithread; n_jobs - number of threads, where -1 all threads, safe for my computer 2
Parallel(n_jobs=2)(
                delayed(compute)
                (clf_name, clf, chunk_size, n_chunks, metric_names, metrics, streams, stream_name)
                for clf_name, clf in zip(clf_names, clfs)
                for stream_name, chunk_size, n_chunks in streams
                )
