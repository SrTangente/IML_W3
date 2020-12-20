from classify import *
from drop import drop2, drop3
from fcnn import fcnn
from menn import menn

classify('vowel', k=1, r=1.5, w='mi', v='maj', reduction_alg=drop2, show=False)

results = pd.read_csv('results.csv')

sat = results[results["dataset"]=='satimage']
sat_results = sat.groupby(['dataset', 'k', 'r', 'w', 'v']).mean()
print('Best combination for satimage')
print(sat_results.iloc[np.argmax(sat_results["acc"])])
print('--------------')

vow = results[results["dataset"]=='vowel']
vow_results = vow.groupby(['dataset', 'k', 'r', 'w', 'v']).mean()
print('Best combination for vowel')
print(vow_results.iloc[np.argmax(vow_results["acc"])])

#classify('vowel', k=5, show=False, reduction_alg=drop3)
#classify('vowel', k=5, show=False, reduction_alg=fcnn)
#classify('vowel', k=1, show=False, reduction_alg=menn)
