import numpy as np
from utils.plotting import plot, save_plot
from utils.statistictest import calc_ranks, friedman_test
# import Orange
import os

# Copy these values from experiment, it has to be the same to correctly load files
clf_names = [
    "HDWE-SVC",
    "AWE-SVC",
    "LearnppNIE-SVC",
    "LearnppCDS-SVC",
    "REA-SVC",
    "OUSE-SVC",
    "SEA-SVC",
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
metric_alias = [
    "Specificity",
    "Recall",
    "Precision",
    "F1",
    "BAC",
    "G-mean1",
    "G-mean2",
]
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
drifts = ['sudden', 'incremental']
n_chunks = 200-1

sigma = 2 # Parameter to gaussian filter

n_streams = len(drifts)*(len(st_stream_weights)*len(random_states)+len(d_stream_weights)*len(random_states))

plot_data = np.zeros((len(clf_names), n_chunks, len(metric_names)))
mean_scores = np.zeros((len(metric_names), n_streams, len(clf_names)))

# Loading data from files, drawing and saving figures in png and eps format
for drift_id, drift in enumerate(drifts):
    # Loading data from files, drawing and saving figures in png and eps format
    for weight_id, weights in enumerate(st_stream_weights+d_stream_weights):
        for rs_id, random_state in enumerate(random_states):
            if len(weights) == 2:
                # Stationary imbalance ratio
                s_name = "stat_ir%s_rs%s" % (weights, random_state)
            elif len(weights) == 3:
                # Dynamically imbalance ratio
                s_name = "d_ir%s_rs%s" % (weights, random_state)
            else:
                raise ValueError("Bad value in weight error")

            stream_id = drift_id*len(st_stream_weights+d_stream_weights)*len(random_states) + weight_id*len(random_states) + rs_id

            for metric_id, (metric_a, metric_name) in enumerate(zip(metric_alias, metric_names)):
                plot_name = "p_gen_%s_ir%s_%s_rs%s" % (drift, weights, metric_name, random_state)
                plotfilename_png = "results/experiment3a/plots/gen/%s/%s/%s.png" % (drift, metric_name, plot_name)
                plotfilename_eps = "results/experiment3a/plots/gen/%s/%s/%s.eps" % (drift, metric_name, plot_name)
                if not os.path.exists("results/experiment3a/plots/gen/%s/%s/" % (drift, metric_name)):
                    os.makedirs("results/experiment3a/plots/gen/%s/%s/" % (drift, metric_name))
                clf_indexes = []
                for clf_id, clf_name in reversed(list(enumerate(clf_names))):
                    try:
                        # Load data from file
                        filename = "results/experiment3a/metrics/gen/%s/%s/%s/%s.csv" % (drift, s_name, metric_name, clf_name)
                        plot_data = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                        # Plot metrics of each stream
                        plot_object = plot(plot_data, clf_name, clf_id, sigma)

                        # Save average of scores into mean_scores, 1 stream = 1 avg
                        scores = plot_data.copy()
                        mean_score = np.mean(scores)
                        mean_scores[metric_id, stream_id, clf_id] = mean_score
                        clf_indexes.append(clf_id)

                    except IOError:
                        # print("File", filename, "not found")
                        print("File not found")
                        # continue if file not found

                # Save plots of metrics of each stream
                save_plot(plot_object, drift, metric_name, metric_a, np.array(clf_names)[list(reversed(clf_indexes))], n_chunks, plotfilename_png, plotfilename_eps)

# print("\nMean scores:\n", mean_scores)

# for metric_id, metric_a in enumerate(metric_alias):
#     ranks, mean_ranks = calc_ranks(mean_scores, metric_id)
#     critical_difference = Orange.evaluation.compute_CD(mean_ranks, n_streams, test='nemenyi')
#
#     # Friedman test, implementation from Demsar2006
#     print("\n", metric_a)
#     friedman_test(clf_names, mean_ranks, n_streams, critical_difference)
#
#     # CD diagrams to compare base classfiers with each other based on Nemenyi test (post-hoc)
#     fnames = [('results/experiment3a/plot_ranks/cd_%s.png' % metric_a), ('results/experiment3a/plot_ranks/cd_%s.eps' % metric_a)]
#     if not os.path.exists('results/experiment3a/plot_ranks/'):
#         os.makedirs('results/experiment3a/plot_ranks/')
#     for fname in fnames:
#         Orange.evaluation.graph_ranks(mean_ranks, clf_names, cd=critical_difference, width=6, textspace=1.5, filename=fname)
