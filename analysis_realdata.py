import numpy as np
from utils.plotting import plot, save_plot, plot_radars
import os

# Loading of the real data stream
streams = []                                                     # size   n_chunks
streams.append(("real-data/covtypeNorm-1-2vsAll-pruned.arff",    2000,    int(267000/2000)))
streams.append(("real-data/poker-lsn-1-2vsAll-pruned.arff",      2000,    int(359999/2000)))
streams.append(("real-data/2vsA_INSECTS-abrupt_imbalanced_norm.arff",         2000,   int(355274/2000)))
# stream_names = ["covtype", "poker"]

# Copy these values from experiment, it has to be the same to correctly load files
base_clf_names = [
    "SVC",
    "HDDT"
    ]
clf_names_ = [
    "HDWE",
    "AWE",
    "LearnppNIE",
    "LearnppCDS",
    "REA",
    "OUSE",
    "SEA",
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

sigma = 5       # Parameter to gaussian filter
n_streams = len(streams)

n_chunks = []
for stream in streams:
    n_chunks.append(stream[2]-2)
# n_chunks = max(n_chunks)

plot_data = np.zeros((len(clf_names_), max(n_chunks), len(metric_names)))
mean_scores = np.zeros((len(metric_names), n_streams, len(clf_names_)))

stream_names = []
for stream in streams:
    stream_names.append(stream[0].split("/")[-1])

# Loading data from files, drawing and saving figures in png and eps format
for base_clf_name in base_clf_names:
    clf_names = [clf+"-"+base_clf_name for clf in clf_names_]
    for stream_id, stream in enumerate(stream_names):
        for metric_id, (metric_a, metric_name) in enumerate(zip(metric_alias, metric_names)):
            plot_name = "%s_%s_%s" % (base_clf_name, stream, metric_name)
            plotfilename_png = "results/experiment_real/plots/%s/%s/%s.png" % (stream, metric_name, plot_name)
            plotfilename_eps = "results/experiment_real/plots/%s/%s/%s.eps" % (stream, metric_name, plot_name)
            if not os.path.exists("results/experiment_real/plots/%s/%s/" % (stream, metric_name)):
                os.makedirs("results/experiment_real/plots/%s/%s/" % (stream, metric_name))
            for clf_id, clf_name in reversed(list(enumerate(clf_names))):
                try:
                    # Load data from file
                    filename = "results/experiment_real/metrics/%s/%s/%s.csv" % (stream, metric_name, clf_name)
                    plot_data = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    # Plot metrics of each stream
                    plot_object = plot(plot_data, clf_name, clf_id, sigma)

                    # Save average of scores into mean_scores, 1 stream = 1 avg
                    scores = plot_data.copy()
                    mean_score = np.mean(scores)
                    mean_scores[metric_id, stream_id, clf_id] = mean_score

                except IOError:
                    print("File", filename, "not found")

            # Save plots of metrics of each stream
            save_plot(plot_object, stream, metric_name, metric_a, clf_names, n_chunks[stream_id], plotfilename_png, plotfilename_eps)

    # print("\nMean scores:\n", mean_scores)

    metric_nar = [
        "specificity",
        "recall",
        "precision",
        "f1_score",
        "balanced_accuracy_score",
        "geometric_mean_score_1",
    ]
    metric_ar = [
        "Specificity",
        "Recall",
        "Precision",
        "F1",
        "BAC",
        "G-mean1",
    ]

# If error:
# ValueError: The number of FixedLocator locations (7), usually from a call to set_ticks, does not match the number of ticklabels (6).
# Install old version of matplotlib:
# pip install matplotlib==3.2.1
    plot_radars(clf_names, stream_names, metric_nar, clf_names, metric_ar)
