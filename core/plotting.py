import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcdefaults
import matplotlib.lines as lines
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


def plot_streams_matplotlib(methods, streams, metrics, experiment_name, gauss=0, methods_alias=None, metrics_alias=None):
    rcdefaults()

    if methods_alias is None:
        methods_alias = methods
    if metrics_alias is None:
        metrics_alias = metrics

    data = {}

    for stream_name in streams:
        for clf_name in methods:
            for metric in metrics:
                try:
                    filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                    data[stream_name, clf_name, metric] = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                except Exception:
                    data[stream_name, clf_name, metric] = None
                    print("Error in loading data", stream_name, clf_name, metric)

    for stream_name in tqdm(streams, "Plotting %s" % experiment_name):
        for metric, metric_a in zip(metrics, metrics_alias):

            for clf_name, method_a in zip(methods, methods_alias):
                if data[stream_name, clf_name, metric] is None:
                    continue

                plot_data = data[stream_name, clf_name, metric]

                if gauss > 0:
                    plot_data = gaussian_filter1d(plot_data, gauss)

                plt.plot(range(len(plot_data)), plot_data, label=method_a)

            filename = "results/plots/%s/%s/%s" % (experiment_name, metric, stream_name)

            stream_name_ = "/".join(stream_name.split("/")[0:-1])

            if not os.path.exists("results/plots/%s/%s/%s/" % (experiment_name, metric, stream_name_)):
                os.makedirs("results/plots/%s/%s/%s/" % (experiment_name, metric, stream_name_))

            plt.legend()
            plt.ylabel(metric_a)
            plt.xlabel("Data chunk")
            plt.gcf().set_size_inches(10, 5)
            plt.savefig(filename+".png", bbox_inches='tight')
            plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
            plt.clf()
            plt.close()


def plot_table_matplotlib_params(methods, streams, metrics, experiment_names, methods_alias=None, metrics_alias=None):
    rcdefaults()

    if methods_alias is None:
        methods_alias = methods
    if metrics_alias is None:
        metrics_alias = metrics

    data = np.zeros((len(methods), len(experiment_names), len(metrics)))
    for index_k, (metric, metric_a) in enumerate(zip(metrics, metrics_alias)):
        for index_j, experiment_name in enumerate(experiment_names):
            for index_i, clf_name in enumerate(methods):
                median_data = []
                for stream_name in streams:
                    try:
                        filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                        median_data.append(np.median(np.genfromtxt(filename, delimiter=',', dtype=np.float32)))
                    except Exception:
                        median_data.append(0)
                        print("Error in loading data", stream_name, clf_name, metric, experiment_name)
                data[index_i, index_j, index_k] = np.mean(median_data)

    # min = np.min(data)
    # max = np.max(data)
    hsize = len(methods)*0.5+1.4
    fig, axes = plt.subplots(1, len(experiment_names)*len(metrics), figsize=(len(metrics)*2.5, hsize))
    plt.subplots_adjust(wspace=0, left=0.14, right=0.93)
    cmap = cm.get_cmap('binary')

    for j, _ in enumerate(experiment_names):
        for k, _ in enumerate(metrics):
            index = j + k * len(experiment_names)

            data_ = np.expand_dims(data[:, j, k], axis=1)

            axes[index].imshow(data_, cmap=cmap)
            # axes[index].imshow(data_, vmin=0, vmax=1, cmap=cmap)
            # axes[index].imshow(data_, vmin=min, vmax=max, cmap=cmap)

            axes[index].set_xticks(np.arange(1))
            axes[index].set_xticklabels([experiment_names[j].upper().split("/")[-1]])
            axes[index].set_yticks([])
            axes[index].set_yticklabels([])
            # axes[index].set_title(experiment_names[k].upper().split("/")[-1])

            # Rotate the tick labels and set their alignment.
            plt.setp(axes[index].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(len(methods)):
                if data_[i] == np.max(data_):
                    axes[index].text(0, i, "%0.3f" % data_[i], ha="center", va="center", weight="bold", color="white")
                    axes[index].add_line(lines.Line2D([-0.4, 0.4], [i+.15, i+.15], color="white"))
                    continue
                if data_[i] > ((np.max(data_)-np.min(data_))/2) + np.min(data_):
                    axes[index].text(0, i, "%0.3f" % data_[i], ha="center", va="center", color="white")
                else:
                    axes[index].text(0, i, "%0.3f" % data_[i], ha="center", va="center", color="black")

    axes[0].set_yticks(np.arange(len(methods)))
    # axes[0].set_yticklabels(["DSE-B"+m for m in methods_alias])
    axes[0].set_yticklabels([m.split("_")[-1] for m in methods])

    for i, name in enumerate(metrics_alias):
        axes[i*len(experiment_names)+1].set_title(name, x=1.05)

    for i in range(1, len(metrics)+1):
        for j in range(i*len(experiment_names), len(metrics)*len(experiment_names)):
            box = axes[j].get_position()
            box.x1 = box.x1 + 0.03
            axes[j].set_position(box)

    if not os.path.exists("results/tables"):
        os.makedirs("results/tables")

    filename = "results/tables/table"
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.savefig(filename+".eps", format='eps', bbox_inches='tight')


def drift_metrics_table_mean(methods, streams, metrics, experiment_names, methods_alias=None, metrics_alias=None, streams_alias=None):

    if methods_alias is None:
        methods_alias = methods
    if metrics_alias is None:
        metrics_alias = metrics

    data = {}
    for experiment_name in experiment_names:
        for clf_name in methods:
            for metric in metrics:
                s_data = []
                for stream_name in streams:
                    try:
                        filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                        if np.isnan(np.mean(np.genfromtxt(filename, delimiter=',', dtype=np.float32))):
                            print("Nan in loading data", stream_name, clf_name, metric, experiment_name)
                        s_data.append(np.mean(np.genfromtxt(filename, delimiter=',', dtype=np.float32)))
                    except Exception:
                        s_data.append(0)
                        print("Error in loading data", stream_name, clf_name, metric, experiment_name)
                data[experiment_name, clf_name, metric, 'mean'] = np.mean(s_data)
                data[experiment_name, clf_name, metric, 'std'] = np.std(s_data)

    best = {}
    for experiment_name in experiment_names:
        for metric in metrics:
            vals = []
            for clf_name in methods:
                vals.append(data[experiment_name, clf_name, metric, 'mean'])

            best[experiment_name, metric] = methods[np.argmin(vals)]
    print(best)


    if not os.path.exists("results/drift_metrics"):
        os.makedirs("results/drift_metrics")
    for metric, metric_a in zip(metrics,metrics_alias):
        table_tex = "\\begin{table}[]\n\\centering\n\\caption{Mean %s on generated %s streams (less is better)}\n" % (metric_a, streams_alias)
        table_tex += "\\scalebox{0.87}{\n\\begin{tabular}{|l|" + "c|"*len(experiment_names) + "}\n"
        table_tex += "\\hline\n & " + " & ".join(experiment_names).upper() + " \\\\ \\hline\n"
        # table_tex = "\\begin{table}[]\n\\centering\n\\caption{$RT_M$ - Mean of recovery time, $RT_S$ - Standard deviation of recovery time, $PL_M$ - Mean of performance loss, $PL_S$ - Standard deviation of performance loss}\n\\begin{tabular}{|l|c|c|c|c|"
        # table_tex += "}\n\\hline\n" + " & $RT_M$ & $RT_S$ & $PL_M$ & $PL_S$" + " \\\\ \\hline\n"
        for method, method_a in zip(methods, methods_alias):
            table_tex += "%s " % method_a
            for experiment_name in experiment_names:
                mean = data[experiment_name, method, metric, 'mean']
                std = data[experiment_name, method, metric, 'std']
                if best[experiment_name, metric] == method:
                    table_tex += "& \\textbf{%0.4f}$\\pm$\\textbf{%0.4f} " % (mean, std)
                else:
                    table_tex += "& %0.4f$\\pm$%0.4f " % (mean, std)
            table_tex += "\\\\ \n"
        table_tex += "\\hline \n"


        table_tex += "\\end{tabular}}\n\\end{table}\n"

        filename = "results/drift_metrics/mean_%s_%s.tex" % (metric, streams_alias)

        print(filename)

        if not os.path.exists("results/drift_metrics/"):
            os.makedirs("results/drift_metrics/")

        with open(filename, "w") as text_file:
            text_file.write(table_tex)


def plot_streams_nexp(methods, streams, metrics, experiment_name, methods_alias=None, metrics_alias=None):
    rcdefaults()

    if methods_alias is None:
        methods_alias = methods
    if metrics_alias is None:
        metrics_alias = metrics

    data = {}

    random_states = []
    streams_ = []
    noise = []
    for stream_name in streams:
        random_states.append(stream_name.split("_")[-1])
        streams_.append("_".join(stream_name.split("_")[0:-1]))
        noise.append(stream_name.split("_")[-2][1:])

    random_states = list(dict.fromkeys(random_states))
    streams_ = list(dict.fromkeys(streams_))
    noise = list(dict.fromkeys(noise))

    for stream_name in streams:
        for clf_name in methods:
            for metric in metrics:
                try:
                    filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                    data["_".join(stream_name.split("_")[0:-1]), clf_name, metric, stream_name.split("_")[-1]] = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                except Exception:
                    data[stream_name, clf_name, metric] = None
                    print("Error in loading data", stream_name, clf_name, metric)

    width = 1/(len(methods)+1)

    for metric, metric_a in zip(metrics, metrics_alias):

        min = 1
        index = -len(methods)/2+0.5

        for clf_name, method_a in zip(methods, methods_alias):
            plot_data = []
            for stream_name in streams_:
                rs_data = []
                for rs in random_states:
                    if data[stream_name, clf_name, metric, rs] is None:
                        continue

                    rs_data.append(np.mean(data[stream_name, clf_name, metric, rs]))
                plot_data.append(np.mean(rs_data))
            if min > np.min(plot_data):
                min = np.min(plot_data)
            x = np.arange(len(streams_))
            plt.bar(x+width*index, plot_data, width, label=method_a)
            # plt.plot(x, plot_data, label=method_a)
            index += 1

        filename = "results/plots/%s/%s" % (experiment_name, metric)
        if not os.path.exists("results/plots/%s/" % (experiment_name)):
            os.makedirs("results/plots/%s/" % (experiment_name))

        plt.legend()
        plt.ylabel(metric_a)
        plt.xlabel("Noise [%]")
        plt.ylim(bottom=min-0.05)
        plt.title(metric_a.upper()+" "+experiment_name.split('/')[-1].upper())
        plt.legend(loc=3)
        plt.xticks(range(len(streams_)), labels=noise)
        plt.gcf().set_size_inches(6, 4)
        plt.savefig(filename+".png", bbox_inches='tight')
        plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
        plt.clf()
        plt.close()


def plot_streams_bexp(methods, streams, metrics, experiment_name, methods_alias=None, metrics_alias=None):
    rcdefaults()

    if methods_alias is None:
        methods_alias = methods
    if metrics_alias is None:
        metrics_alias = metrics

    data = {}

    random_states = []
    streams_ = []
    noise = []
    for stream_name in streams:
        random_states.append(stream_name.split("_")[-1])
        streams_.append("_".join(stream_name.split("_")[0:-1]))
        noise.append(stream_name.split("_")[-3][1:])

    random_states = list(dict.fromkeys(random_states))
    streams_ = list(dict.fromkeys(streams_))
    noise = list(dict.fromkeys(noise))

    for stream_name in streams:
        for clf_name in methods:
            for metric in metrics:
                try:
                    filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                    data["_".join(stream_name.split("_")[0:-1]), clf_name, metric, stream_name.split("_")[-1]] = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                except Exception:
                    data[stream_name, clf_name, metric] = None
                    print("Error in loading data", stream_name, clf_name, metric)

    width = 1/(len(methods)+1)

    for metric, metric_a in zip(metrics, metrics_alias):

        min = 1
        index = -len(methods)/2+0.5

        for clf_name, method_a in zip(methods, methods_alias):
            plot_data = []
            for stream_name in streams_:
                rs_data = []
                for rs in random_states:
                    if data[stream_name, clf_name, metric, rs] is None:
                        continue

                    rs_data.append(np.mean(data[stream_name, clf_name, metric, rs]))
                plot_data.append(np.mean(rs_data))
            if min > np.min(plot_data):
                min = np.min(plot_data)
            x = np.arange(len(streams_))
            plt.bar(x+width*index, plot_data, width, label=method_a)
            # plt.plot(x, plot_data, label=method_a)
            index += 1

        filename = "results/plots/%s/%s" % (experiment_name, metric)
        if not os.path.exists("results/plots/%s/" % (experiment_name)):
            os.makedirs("results/plots/%s/" % (experiment_name))

        plt.legend()
        plt.ylabel(metric_a)
        plt.xlabel("Imbalance ratio [%]")
        plt.ylim(bottom=min-0.05)
        plt.title(metric_a.upper()+" "+experiment_name.split('/')[-1].upper())
        plt.legend(loc=4)
        plt.xticks(range(len(streams_)), labels=noise)
        plt.gcf().set_size_inches(6, 4)
        plt.savefig(filename+".png", bbox_inches='tight')
        plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
        plt.clf()
        plt.close()
