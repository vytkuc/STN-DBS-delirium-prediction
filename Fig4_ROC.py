#Machine Learning-Driven Radiomic Profiling of Thalamus-Amygdala Nuclei for Prediction of Postoperative Delirium after STN-DBS in Parkinson's Disease Patients
#Radziunas A. et al. 2024

#ROC plots of statistical and machine learning algorithms


import numpy as np
import matplotlib.pyplot as plt
from statistics import stdev
import os

#Select 5, 10 or 20
number_of_features = 10
base_fpr = np.linspace(0, 1, 101)
datadir = "data"
accs = np.load(os.path.join(datadir, str(number_of_features), "ACC.npy"))
sens = np.load(os.path.join(datadir, str(number_of_features), "SENS.npy"))
spes = np.load(os.path.join(datadir, str(number_of_features), "SPEC.npy"))
aucs = np.load(os.path.join(datadir, str(number_of_features), "AUC.npy"))
tprs = np.load(os.path.join(datadir, str(number_of_features), "TPRS.npy"))



def DBS_outcome_prediction_accuracy(res_acc, res_sens, res_spec, res_AUC, tprs, axis, name):
    print(name)
    print("%4.2f  (%4.2f)    %4.2f (%4.2f)   %4.2f (%4.2f)   %4.2f (%4.2f)" % (100*sum(res_acc)/len(res_acc), 100*stdev(res_acc), 100*sum(res_sens)/len(res_sens), 100*stdev(res_sens),
          100*sum(res_spec)/len(res_spec), 100*stdev(res_spec), 100*sum(res_AUC)/len(res_AUC), 100*stdev(res_AUC)))
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    axis.plot(base_fpr, mean_tprs, 'b')
    axis.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    axis.plot([0, 1], [0, 1],'r--')
    axis.set_xlim([-0.01, 1.01])
    axis.set_ylim([-0.01, 1.01])
    axis.set_ylabel('True Positive Rate', fontsize=14)
    axis.set_xlabel('False Positive Rate', fontsize=14)
    axis.text(x = 0.5, y = 0.15, s="AUC = %4.2f (%4.2f)" % (sum(res_AUC)/len(res_AUC), stdev(res_AUC)), fontsize=14)
    axis.set_title(name, fontsize=18)

fig, axes = plt.subplots(2,4, figsize=(20, 8))
axes = axes.ravel()

names = {0: 'LR', 1:'DT',
         2: 'LDA', 3: 'NB',
         4: 'SVM', 5: 'ANN',
         6: 'OC-SVM', 7: 'ANN-A'}

print("Accuracy %  Sensitivity %  Specificity %  AUC")
for i, el in enumerate(list([0, 1, 2, 3, 4, 5, 6, 7])):
    DBS_outcome_prediction_accuracy( accs[el], sens[el], spes[el], aucs[el], tprs[el], axes[i], names[el])

plt.tight_layout()
#Increase Tick Font Size Globally
plt.rc('xtick', labelsize=12)  # Font size for x-axis ticks
plt.rc('ytick', labelsize=12)  # Font size for y-axis ticks


# PNG
figpath = os.path.join("figures", "Fig4_ROC_"+str(number_of_features)+"_features.png")
fig.savefig(figpath, dpi=300, bbox_inches='tight', format='png')

# TIFF
#figpath = os.path.join("figures", "Fig4_ROC_"+str(number_of_features)+"_features.tiff")
#fig.savefig(figpath, dpi=300, bbox_inches='tight', format='tiff')