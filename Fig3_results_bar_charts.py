#Machine Learning-Driven Radiomic Profiling of Thalamus-Amygdala Nuclei for Prediction of Postoperative Delirium after STN-DBS in Parkinson's Disease Patients
#Radziunas A. et al. 2024

#Bar charts of results of statistical and machine learning algorithms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statistics import stdev
import os



def DBS_outcome_prediction_accuracy(res_acc, res_sens, res_spec, res_AUC, name):
    print(name)
    print("%4.2f  (%4.2f)    %4.2f (%4.2f)   %4.2f (%4.2f)   %4.2f (%4.2f)" % (100*sum(res_acc)/len(res_acc), 100*stdev(res_acc), 100*sum(res_sens)/len(res_sens), 100*stdev(res_sens),
          100*sum(res_spec)/len(res_spec), 100*stdev(res_spec), 100*sum(res_AUC)/len(res_AUC), 100*stdev(res_AUC)))
    
# Define the colors
# Create a colormap instance
cmap = plt.get_cmap('tab10')

# Get the first four colors
colors = [cmap(i) for i in range(4)]
#or specific colors
#colors = ['dodgerblue', 'royalblue', 'mediumblue', 'darkblue']

def plot_barchart(ax, data, title):
    # Calculate mean and standard deviation
    mean_values = [np.mean(metric) for metric in data]
    std_values = [np.std(metric) for metric in data]

    # Create bar chart with error bars
    bar_positions = np.arange(len(mean_values))
    for pos, mean, yerr, color in zip(bar_positions, mean_values, std_values, colors):
        ax.bar(pos, mean, align='center', alpha=0.7, color=color, ecolor='black', capsize=5, width=0.5)
        ax.errorbar(pos, mean, yerr=yerr, fmt='none', ecolor='black', capsize=None)

    # Set chart title and labels
    ax.set_title(title)
    ax.set_xticks(bar_positions)
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(['Acc', 'Sens', 'Spec', 'AUC'])

    # Remove top border of subplot
    ax.spines['top'].set_visible(False)
    
datasets = [5, 10, 20]

# Determine the global y-axis limits
y_min = 0
y_max = 1

for number_of_features in datasets:
    datadir = "data/" +str(number_of_features)

    accs = np.load(os.path.join(datadir, "ACC.npy"))
    sens = np.load(os.path.join(datadir, "SENS.npy"))
    spes = np.load(os.path.join(datadir, "SPEC.npy"))
    aucs = np.load(os.path.join(datadir, "AUC.npy"))
    
    names = {0: 'LR', 1:'DT',
         2: 'LDA', 3: 'NB',
         4: 'SVM', 5: 'ANN',
         6: 'OC-SVM', 7: 'ANN-A'}

    fig, axs = plt.subplots(1, 8, figsize=(14, 4), constrained_layout=True, sharex=True, sharey=True)
    #fig.suptitle(f'Performance Metrics for {number_of_features} Features')

    for i, el in enumerate(list([0, 1, 2, 3, 4, 5, 6, 7])):
        DBS_outcome_prediction_accuracy( accs[el], sens[el], spes[el], aucs[el], names[el])
        
        # Plot bar charts
        #for 2x4
        #ax = axs[i//4, i%4]
        #for 1x8
        ax = axs[i]
        plot_barchart(ax, [accs[el], sens[el], spes[el], aucs[el]], names[el])
    
    plt.tight_layout()
    
    figpath = os.path.join("figures", "Fig3_Bar_charts_"+str(number_of_features)+"_features.png")
    fig.savefig(figpath, dpi=300, bbox_inches='tight', format='png')
    plt.show()
     
        
        
        