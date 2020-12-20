from classify import *
from drop import drop2, drop3
from fcnn import fcnn

df = pd.DataDrame()
for k in [1,3,5,7]:
    df=pd.concat([df,classify_fast('satimage', 5, 1, 'eq', 'shep', reduction_alg=fcnn)])
df.to_csv('satimage_drop3_k.csv')

df = pd.DataDrame()
for k in [1,3,5, 7]:
    df=pd.concat([df,classify_fast('vowel',1, 1.5, 'mi', 'maj', reduction_alg=fcnn)])
df.to_csv('vowel_drop3_k.csv')

dfs = classify('satimage', k=5, r=1, w='eq', v='shep', reduction_alg=drop3, show=False)
dfs.to_csv('satimage_drop3.csv')

dfv = classify('vowel', k=1, r=1.5, w='mi', v='maj', reduction_alg=drop3, show=False)
dfv.to_csv('vowel_drop3.csv')

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