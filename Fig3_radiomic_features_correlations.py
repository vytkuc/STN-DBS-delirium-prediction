#Machine Learning-Driven Radiomic Profiling of Thalamus-Amygdala Nuclei for Prediction of Postoperative Delirium after STN-DBS in Parkinson's Disease Patients
#Radziunas A. et al. 2024

#Spearman correlation coefficients between the selected radiomic features


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    return pvalues


plt.figure(figsize=(16, 6))
datadir = "data"
datafile = os.path.join(datadir, "data_radiomics_features_selected_mRMR.csv")
dataframe = pd.read_csv(datafile, index_col=0, usecols =[i for i in range(21)])

p_values = calculate_pvalues(dataframe)
p_values = np.where(p_values < 0.05, np.where(p_values < 0.01, '**', '*'), '')
np.fill_diagonal(p_values, '')
strings = p_values
results = dataframe.corr(method='spearman').to_numpy()

labels = (np.asarray(["{1:.2f}{0}".format(string, value)
                      for string, value in zip(strings.flatten(),
                                               results.flatten())])
         ).reshape(20, 20)


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(dataframe.corr(method='spearman'), vmin=-1, vmax=1, annot=labels, fmt='')

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right') 
figpath = os.path.join("figures", "Fig3_radiomic_features_correlations.png")
plt.savefig(figpath, dpi=300, bbox_inches='tight', format='png')

'''
Dataframe with correlation coefficients (abs) and p values
'''
# Calculate the correlation and p-values of each pair of features
correlations_and_pvalues = dataframe.apply(lambda x: pd.Series([spearmanr(x, dataframe[col])[0:2] for col in dataframe.columns], index=dataframe.columns))

# Unstack the correlation matrix into a DataFrame and reset the index
unstacked_correlations_and_pvalues = correlations_and_pvalues.unstack().reset_index()

# Rename the columns
unstacked_correlations_and_pvalues.columns = ['Feature 1', 'Feature 2', 'Correlation and P-value']

# Split the correlation and p-value into separate columns
unstacked_correlations_and_pvalues[['Correlation', 'P-value']] = pd.DataFrame(unstacked_correlations_and_pvalues['Correlation and P-value'].tolist(), index=unstacked_correlations_and_pvalues.index)

# Remove the 'Correlation and P-value' column
unstacked_correlations_and_pvalues = unstacked_correlations_and_pvalues.drop(columns=['Correlation and P-value'])

# Remove the correlations of a feature with itself
unstacked_correlations_and_pvalues = unstacked_correlations_and_pvalues[unstacked_correlations_and_pvalues['Feature 1'] != unstacked_correlations_and_pvalues['Feature 2']]

# Remove duplicates: keep only one of the pairs (Feature 1, Feature 2) and (Feature 2, Feature 1)
unstacked_correlations_and_pvalues['sorted_features'] = unstacked_correlations_and_pvalues.apply(lambda x: tuple(sorted([x['Feature 1'], x['Feature 2']])), axis=1)
unstacked_correlations_and_pvalues = unstacked_correlations_and_pvalues.drop_duplicates(subset='sorted_features')
unstacked_correlations_and_pvalues = unstacked_correlations_and_pvalues.drop(columns='sorted_features')

# Show absolute value of correlation coefficient
unstacked_correlations_and_pvalues['Correlation'] = unstacked_correlations_and_pvalues['Correlation'].abs()

# Sort the correlations in descending order
sorted_correlations_and_pvalues = unstacked_correlations_and_pvalues.sort_values(by='Correlation', ascending=False)

# Convert to DataFrame
df_corr = pd.DataFrame(sorted_correlations_and_pvalues)

# Save to Excel
#df_corr.to_excel('correlations_and_pvalues.xlsx', index=False)




