from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os

# Plot figure: x - chunk number, y - quality (one of metrics)
def plot(plot_data, clf_name, sigma=2):
    if sigma > 0:
            plot_data = gaussian_filter1d(plot_data, sigma)
    plt.plot(range(len(plot_data)), plot_data, label=clf_name)
    return plt
        
# Save plot to the file png and eps of quality created above
def save_plot(plot, drift, metric_name, metric_alias, n_chunks, plotfilename_png, plotfilename_eps):
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
        
    plt.legend(framealpha=1)
    plt.ylabel(metric_alias)
    plt.xlabel("Data chunk")
    plt.axis([0, n_chunks, 0, 1])
    plt.gcf().set_size_inches(10, 5) # Get the current figure
    plt.savefig(plotfilename_png)
    plt.savefig(plotfilename_eps)
    # plt.show()
    plt.clf() # Clear the current figure
    plt.close() 