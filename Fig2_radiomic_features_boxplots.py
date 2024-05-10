#Machine Learning-Driven Radiomic Profiling of Thalamus-Amygdala Nuclei for Prediction of Postoperative Delirium after STN-DBS in Parkinson's Disease Patients
#Radziunas A. et al. 2024

#Boxplots of selected radiomic features in two classes: 
#no postoperative STN-DBS delirium vs postoperative STN-DBS delirium

import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
import os


datadir = "data"
datafile = os.path.join(datadir, "data_radiomics_features_selected_mRMR.csv")
df = pd.read_csv(datafile, index_col=0)

fig, axes = plt.subplots(5,4, figsize=(15, 20))

for i,el in enumerate(list(df.columns.values)[:-1]):
    a = df.boxplot(el, by='STN-DBS delirium (1-yes, 2-no)', ax=axes.flatten()[i], patch_artist=True, boxprops=dict(facecolor='skyblue', color='black', linewidth=2), medianprops=dict(color='black', linewidth=3))
    a.grid('on', which='major', linewidth=1)
    title = a.set_title("\n".join(wrap(el, 30)))
    title.set_y(1.05)

plt.tight_layout() 
plt.suptitle('')
figpath = os.path.join("figures", "Fig2_radiomic_features_boxplots.png")
fig.savefig(figpath, dpi=400, bbox_inches='tight', format='png')
