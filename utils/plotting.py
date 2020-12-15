from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rcdefaults
from math import pi


# Plot figure: x - chunk number, y - quality (one of metrics)
def plot(plot_data, clf_name, idx, sigma=2):
    if sigma > 0:
            plot_data = gaussian_filter1d(plot_data, sigma)

    styles = ['-', '--', '--', '--', '--', '--', '--', '--']
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray']
    widths = [1.5, 1, 1, 1, 1, 1, 1, 1, 1]

    # styles = ['-', '-', '-', '-', '-', '-', '-', '-']
    # colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray']
    # widths = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    plt.plot(range(len(plot_data)), plot_data, label=clf_name, linestyle=styles[idx], color=colors[idx], linewidth=widths[idx])
    # plt.plot(range(len(plot_data)), plot_data, label=clf_name)
    return plt


# Save plot to the file png and eps of quality created above
def save_plot(plot, drift, metric_name, metric_alias, clf_names, n_chunks, plotfilename_png, plotfilename_eps):
    if not os.path.exists("results/experiment2/plots/gen/%s/%s/" % (drift, metric_name)):
        os.makedirs("results/experiment2/plots/gen/%s/%s/" % (drift, metric_name))

    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.legend()
    plt.legend(reversed(plt.legend().legendHandles), clf_names, framealpha=1)
    plt.grid(True, color="silver", linestyle=":")

    # plt.legend(framealpha=1)
    plt.ylabel(metric_alias)
    plt.xlabel("Data chunk")
    plt.axis([0, n_chunks, 0, 1])
    plt.gcf().set_size_inches(10, 5)  # Get the current figure
    plt.savefig(plotfilename_png)
    plt.savefig(plotfilename_eps)
    # plt.show()
    plt.clf()  # Clear the current figure
    plt.close()


# Prepared only for real data
def plot_radars(methods, streams, metrics, methods_alias=None, metrics_alias=None):

    N = len(metrics)

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
                    filename = "results/experiment_real/metrics/%s/%s/%s.csv" % (stream_name, metric, clf_name)
                    data[stream_name, clf_name, metric] = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                except Exception:
                    data[stream_name, clf_name, metric] = None
                    print("Error in loading data", stream_name, clf_name, metric)

    min = 1

    for stream_name in streams:

        ls = ['-', '--', '--', '--', '--', '--', '--', '--']
        lw = [1, .5, .5, .5, .5, .5, .5, .5, .5]
        colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray']

        # Angle of each axis in the plot - divide the plot / number of variable
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)

        # First axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Without border
        ax.spines["polar"].set_visible(False)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles, metrics_alias)

        for i, (clf_name, method_a) in reversed(list(enumerate(zip(methods, methods_alias)))):
            plot_data = []
            for metric in metrics:
                if data[stream_name, clf_name, metric] is None:
                    continue
                plot_data.append(np.mean(data[stream_name, clf_name, metric]))
            plot_data += plot_data[:1]
            if min > np.min(plot_data):
                min = np.min(plot_data)
            ax.plot(angles, plot_data, label=method_a, c=colors[i], ls=ls[i], lw=lw[i])

        # Add legend
        plt.legend()
        plt.legend(
            reversed(plt.legend().legendHandles),
            methods_alias,
            loc="lower center",
            ncol=2,
            columnspacing=1,
            frameon=False,
            bbox_to_anchor=(0.5, -0.4),
            fontsize=6,
        )

        # Add a grid
        plt.grid(ls=":", c=(0.7, 0.7, 0.7))

        # Add a title
        plt.title("%s" % (stream_name.split("-")[0]), size=8, y=1.09, fontfamily="serif")
        plt.tight_layout()

        # Draw labels
        a = np.linspace(0, 1, 6)
        plt.yticks(a[1:], ["%.1f" % f for f in a[1:]], fontsize=6, rotation=90)
        plt.ylim(0.0, 1.0)
        plt.gcf().set_size_inches(4, 3.5)
        plt.gcf().canvas.draw()
        angles = np.rad2deg(angles)

        ax.set_rlabel_position((angles[0] + angles[1]) / 2)

        har = [(a >= 90) * (a <= 270) for a in angles]

        for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
            x, y = label.get_position()
            lab = ax.text(
                x, y, label.get_text(), transform=label.get_transform(), fontsize=6,
            )
            lab.set_rotation(angle)

            if har[z]:
                lab.set_rotation(180 - angle)
            else:
                lab.set_rotation(-angle)
            lab.set_verticalalignment("center")
            lab.set_horizontalalignment("center")
            lab.set_rotation_mode("anchor")

        for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
            x, y = label.get_position()
            lab = ax.text(
                x,
                y,
                label.get_text(),
                transform=label.get_transform(),
                fontsize=4,
                c=(0.7, 0.7, 0.7),
            )
            lab.set_rotation(-(angles[0] + angles[1]) / 2)

            lab.set_verticalalignment("bottom")
            lab.set_horizontalalignment("center")
            lab.set_rotation_mode("anchor")

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        filename = "results/experiment_real/plots/radars/radar_%s_%s" % (stream_name.split("-")[0], methods_alias[0].split("-")[-1])
        if not os.path.exists("results/experiment_real/plots/radars/"):
            os.makedirs("results/experiment_real/plots/radars/")

        plt.savefig(filename+".png", bbox_inches='tight', dpi=500, pad_inches=0.0)
        plt.savefig(filename+".eps", bbox_inches='tight', dpi=500, pad_inches=0.0)
        plt.close()
