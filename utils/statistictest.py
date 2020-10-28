import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import rankdata
from tabulate import tabulate
import os
from scipy import stats

# Calculate ranks for every method based on mean_scores; the higher the rank, the better the method
def calc_ranks(mean_scores, metric_id):
    ranks = []
    for ms in mean_scores[metric_id]:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    # print("\nRanks for", metric_a, ": ", ranks, "\n")
    mean_ranks = np.mean(ranks, axis=0)
    # print("\nMean ranks for", metric_a, ": ", mean_ranks, "\n")
    return(ranks, mean_ranks)

# Calculate Friedman statistics
def friedman_test(clf_names, mean_ranks, n_streams, critical_difference):
    N_ = n_streams
    k_ = len(clf_names)
    p_value = 0.05
    
    friedman = (12*N_/(k_*(k_+1)))*(np.sum(mean_ranks**2)-(k_*(k_+1)**2)/4)
    print("Friedman", friedman)
    iman_davenport = ((N_-1)*friedman)/(N_*(k_-1)-friedman)
    print("Iman-davenport", iman_davenport)
    f_dist = stats.f.ppf(1-p_value, k_-1, (k_-1)*(N_-1))
    print("F-distribution", f_dist)
    if f_dist < iman_davenport:
        print("Reject hypothesis H0")
    
    print("Critical difference", critical_difference)
    print(mean_ranks)

# Test t_ student for comparison methods with each other
def t_student(metric_names, metric_alias, mean_scores, clf_names, alfa = 0.05):
    t_statistics = np.zeros((len(metric_names), len(clf_names), len(clf_names)))
    p_values = np.zeros((len(metric_names), len(clf_names), len(clf_names)))
    for metric_id, metric_a in enumerate(metric_alias):
        ranks, mean_ranks = calc_ranks(mean_scores, metric_id)        
        for i in range(len(clf_names)):
            for j in range(len(clf_names)):
                t_statistics[metric_id, i, j], p_values[metric_id, i, j] = ttest_ind(ranks.T[i], ranks.T[j])
        # print("\nt-statistic:\n", t_statistics, "\n\np-value:\n", p_values)   
    
        headers = clf_names
        names_column = np.expand_dims(np.array(clf_names), axis=1)
        t_statistic_table = np.concatenate((names_column, t_statistics[metric_id]), axis=1)
        t_statistic_table = tabulate(t_statistics[metric_id], headers, floatfmt=".2f")
        p_value_table = np.concatenate((names_column, p_values[metric_id]), axis=1)
        p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
        # print("\nt-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
        
        advantage = np.zeros((len(clf_names), len(clf_names)))
        advantage[t_statistics[metric_id] > 0] = 1
        advantage_table = tabulate(np.concatenate(
            (names_column, advantage), axis=1), headers)
        # print("\nAdvantage:\n", advantage_table)
        
        significance = np.zeros((len(clf_names), len(clf_names)))
        significance[p_values[metric_id] <= alfa] = 1
        significance_table = tabulate(np.concatenate(
            (names_column, significance), axis=1), headers)
        # print("\nStatistical significance (alpha = 0.05):\n", significance_table)
    
        stat_better = significance * advantage
        stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
        print("\nStatistically significantly better for", metric_a, ":\n", stat_better_table)
    
    return(p_values, t_statistics)
    

# It can compare one method (my HDWE) with others
def rankings(metric_alias, metric_names, clf_names, p_values, t_statistics, treshold = 0.10):
    for metric_id, (metric_a, metric_name) in enumerate(zip(metric_alias, metric_names)):
        ranking = {}
        for clf_name in clf_names:
            ranking[clf_name] = {"win": 0, "lose": 0, "tie": 0}
            
        for i, method_1 in enumerate(clf_names):
            for j, method_2 in enumerate(clf_names):
                if p_values[metric_id, i, j] < treshold:
                    if t_statistics[metric_id, i, j] > 0:
                        ranking[method_1]["win"] += 1
                    else:
                        ranking[method_1]["lose"] += 1
                else:
                    ranking[method_1]["tie"] += 1
                
        rank_win = []
        rank_tie = []
        rank_lose = []
        rank_none = []
    
        for clf_name in clf_names:
            rank_win.append(ranking[clf_name]['win'])
            rank_tie.append(ranking[clf_name]['tie'])
            rank_lose.append(ranking[clf_name]['lose'])
            try:
                rank_none.append(ranking[clf_name]['none'])
            except Exception:
                pass
    
        rank_win.reverse()
        rank_tie.reverse()
        rank_lose.reverse()
        rank_none.reverse()
    
        rank_win = np.array(rank_win)
        rank_tie = np.array(rank_tie)
        rank_lose = np.array(rank_lose)
        rank_none = np.array(rank_none)
        ma = clf_names.copy()
        ma.reverse()
        plt.rc('ytick', labelsize=15)
    
        plt.barh(ma, rank_win, color="green", height=0.5)
        plt.barh(ma, rank_tie, left=rank_win, color="gold", height=0.5)
        plt.barh(ma, rank_lose, left=rank_win+rank_tie, color="crimson", height=0.5)
        try:
            plt.barh(ma, rank_none, left=rank_win+rank_tie+rank_lose, color="black", height=0.5)
        except Exception:
            pass
        plt.title(metric_a, fontsize=20)
        # plt.show()
        if not os.path.exists("results/experiment2/plot_ranks/gen/"):
            os.makedirs("results/experiment2/plot_ranks/gen/")
        plt.gcf().set_size_inches(5, 3)
        plt.savefig(fname="results/experiment2/plot_ranks/gen/%s_hbar.png" % (metric_name), bbox_inches='tight')
        plt.clf()
        
        print(ranking)
        