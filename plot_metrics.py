import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

results = pd.read_csv('results.csv')

w={'eq':-0.15, 'chi':0, 'mi':0.15}
results['k+w']=results['k']+results['w'].map(lambda x:w[x])
colors = ['r','g','b']
markers = ['o','s','^']
fill = ['none','top','full']

sat = results[results["dataset"]=='satimage']
sat_results = sat.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()
vowel = results[results["dataset"]=='vowel']
vowel_results = vowel.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()


plt.subplots(2,2,figsize=[10,10])
plt.subplot(2,2,3)
plt.title('Satimage Accuracy')
sns.scatterplot('k+w','acc',data=sat_results,hue='r',style='v')
plt.subplot(2,2,4)
plt.title('Satimage Efficiency')
sns.scatterplot('k+w','eff',data=sat_results,hue='r',style='v')
plt.subplot(2,2,1)
plt.title('Vowel Accuracy')

sns.scatterplot('k+w','acc',data=vowel_results,hue='r',style='v')
plt.subplot(2,2,2)
plt.title('Vowel Efficiency')
sns.scatterplot('k+w','eff',data=vowel_results,hue='r',style='v')

plt.show()