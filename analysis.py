from classify import *
from drop import drop2, drop3
from fcnn import fcnn

results = pd.read_csv('results.csv')

print('General results')
print('k results')
print(results.groupby(['k']).mean())
print('r results')
print(results.groupby(['r']).mean())
print('w results')
print(results.groupby(['w']).mean())
print('v results')
print(results.groupby(['v']).mean())
print('-------------------')

sat = results[results["dataset"]=='satimage']
sat_results = sat.groupby(['dataset', 'k', 'r', 'w', 'v']).mean()
print('Satimage')
print('k results')
print(sat_results.groupby(['k']).mean())
print('r results')
print(sat_results.groupby(['r']).mean())
print('w results')
print(sat_results.groupby(['w']).mean())
print('v results')
print(sat_results.groupby(['v']).mean())
print('Best combination for satimage')
print(sat_results.iloc[np.argmax(sat_results["acc"])])
print('--------------')

vow = results[results["dataset"]=='vowel']
vow_results = vow.groupby(['dataset', 'k', 'r', 'w', 'v']).mean()
print('Vowel')
print('k results')
print(vow_results.groupby(['k']).mean())
print('r results')
print(vow_results.groupby(['r']).mean())
print('w results')
print(vow_results.groupby(['w']).mean())
print('v results')
print(vow_results.groupby(['v']).mean())
print('Best combination for vowel')
print(vow_results.iloc[np.argmax(vow_results["acc"])])
