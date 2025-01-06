#Machine Learning-Driven Radiomic Profiling of Thalamus-Amygdala Nuclei for Prediction of Postoperative Delirium after STN-DBS in Parkinson's Disease Patients
#Radziunas A. et al. 2024

#Bar charts of results of statistical and machine learning algorithms

import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import stdev


def DBS_outcome_prediction_accuracy(res_acc, res_sens, res_spec, res_AUC, name):
    print(name)
    print("%4.2f  (%4.2f)    %4.2f (%4.2f)   %4.2f (%4.2f)   %4.2f (%4.2f)" % (
        100 * sum(res_acc) / len(res_acc), 100 * stdev(res_acc),
        100 * sum(res_sens) / len(res_sens), 100 * stdev(res_sens),
        100 * sum(res_spec) / len(res_spec), 100 * stdev(res_spec),
        100 * sum(res_AUC) / len(res_AUC), 100 * stdev(res_AUC)
    ))


# Define colors
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(4)]


def plot_barchart(ax, data, title):
    mean_values = [np.mean(metric) for metric in data]
    std_values = [np.std(metric) for metric in data]

    # Define Y-axis limits
    left_ylim = 100  # Maximum for left Y-axis (percentage)
    right_ylim = 1.0  # Maximum for right Y-axis (AUC)

    # Plot Acc, Sens, Spec on left Y-axis
    bar_positions = np.arange(len(mean_values))
    capped_std_left = [
        min(std, (left_ylim - 100 * mean) / 100) if 100 * mean + 100 * std > left_ylim else std
        for mean, std in zip(mean_values[:3], std_values[:3])
    ]
    ax.bar(bar_positions[:3], [100 * mean for mean in mean_values[:3]], align='center', alpha=0.7,
           color=colors[:3], ecolor='black', capsize=5, width=0.5, label='Acc, Sens, Spec')
    ax.errorbar(bar_positions[:3], [100 * mean for mean in mean_values[:3]],
                yerr=[100 * std for std in capped_std_left], fmt='none', ecolor='black', capsize=5)

    ax.set_ylabel("Acc, Sens, Spec (%)")
    ax.set_ylim(0, left_ylim)

    # Plot AUC on right Y-axis
    ax2 = ax.twinx()
    capped_std_right = [
        min(std, (right_ylim - mean)) if mean + std > right_ylim else std
        for mean, std in zip(mean_values[3:], std_values[3:])
    ]
    ax2.bar(bar_positions[3:], mean_values[3:], align='center', alpha=0.7,
            color=colors[3], ecolor='black', capsize=5, width=0.5, label='AUC')
    ax2.errorbar(bar_positions[3:], mean_values[3:], yerr=capped_std_right, fmt='none', ecolor='black', capsize=5)

    ax2.set_ylabel("AUC")
    ax2.set_ylim(0, right_ylim)

    ax.set_title(title)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['Acc', 'Sens', 'Spec', 'AUC'])

    # Remove only the top frame line, retain error bar caps
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)


datasets = [5, 10, 20]

for number_of_features in datasets:
    datadir = "data/" + str(number_of_features)

    accs = np.load(os.path.join(datadir, "ACC.npy"))
    sens = np.load(os.path.join(datadir, "SENS.npy"))
    spes = np.load(os.path.join(datadir, "SPEC.npy"))
    aucs = np.load(os.path.join(datadir, "AUC.npy"))

    names = {0: 'LR', 1: 'DT',
             2: 'LDA', 3: 'NB',
             4: 'SVM', 5: 'FNN',
             6: 'OC-SVM', 7: 'FNN-A'}

    # Set up 2x4 subplot layout
    fig, axs = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)

    for i, el in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
        DBS_outcome_prediction_accuracy(accs[el], sens[el], spes[el], aucs[el], names[el])

        # Map each subplot index to its location in the 2x4 grid
        ax = axs[i // 4, i % 4]
        plot_barchart(ax, [accs[el], sens[el], spes[el], aucs[el]], names[el])

    plt.tight_layout()

    figpath = os.path.join("figures", "Fig3_Bar_charts_" + str(number_of_features) + "_features.png")
    fig.savefig(figpath, dpi=300, bbox_inches='tight', format='png')
    plt.show()
        