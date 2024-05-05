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

    # Limit std values to not exceed 1.0
    std_values = [min(std, 1.0) for std in std_values]

    # Create bar chart with error bars
    bar_positions = np.arange(len(mean_values))
    for pos, mean, yerr, color in zip(bar_positions, mean_values, std_values, colors):
        ax.bar(pos, mean, yerr=yerr, align='center', alpha=0.7, color=color, ecolor='black', capsize=5, width=0.5)

    # Set chart title and labels
    ax.set_title(title)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['Acc', 'Sens', 'Spec', 'AUC'])

    # Remove top border of subplot
    ax.spines['top'].set_visible(False)
    
datasets = [5, 10, 20]

base_fpr = np.linspace(0, 1, 101)

# Determine the global y-axis limits
y_min = 0
y_max = 1

for number_of_features in datasets:
    datadir = "data/" +str(number_of_features)

    accs = np.load(os.path.join(datadir, "ACC.npy"))
    sens = np.load(os.path.join(datadir, "SENS.npy"))
    spes = np.load(os.path.join(datadir, "SPEC.npy"))
    aucs = np.load(os.path.join(datadir, "AUC.npy"))
    
    if number_of_features == 5:
        names = {0: '1A Regularized Logistic Regression', 1:'1B Decision Tree Classifier',
         2: '1C Linear discriminant Analysis', 3: '1D Naive Bayes Classifier',
         4: '1E Support Vector Machine', 5: '1F Artificial Neural Network',
         6: '1G One Class Support Vector Machine', 7: '1H Autoencoder'}
    
    elif number_of_features == 10:
        names = {0: '2A Regularized Logistic Regression', 1:'2B Decision Tree Classifier',
         2: '2C Linear discriminant Analysis', 3: '2D Naive Bayes Classifier',
         4: '2E Support Vector Machine', 5: '2F Artificial Neural Network',
         6: '2G One Class Support Vector Machine', 7: '2H Autoencoder'}
    
    elif number_of_features == 20:
        names = {0: '3A Regularized Logistic Regression', 1:'3B Decision Tree Classifier',
         2: '3C Linear discriminant Analysis', 3: '3D Naive Bayes Classifier',
         4: '3E Support Vector Machine', 5: '3F Artificial Neural Network',
         6: '3G One Class Support Vector Machine', 7: '3H Autoencoder'}
    
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    fig.suptitle(f'Performance Metrics for {number_of_features} Features')

    for i, el in enumerate(list([0, 1, 2, 3, 4, 5, 6, 7])):
        DBS_outcome_prediction_accuracy( accs[el], sens[el], spes[el], aucs[el], names[el])
        
        # Plot bar charts
        ax = axs[i//4, i%4]
        plot_barchart(ax, [accs[el], sens[el], spes[el], aucs[el]], names[el])

    plt.tight_layout()
    
    figpath = os.path.join("figures", "Fig5_Bar_charts_"+str(number_of_features)+"_features.png")
    fig.savefig(figpath, dpi=300, bbox_inches='tight', format='png')
    plt.show()
        
        
        
        
        