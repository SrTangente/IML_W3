from classify import *
from drop import drop2, drop3
from fcnn import fcnn
from menn import menn

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
df = classify('vowel', k=1, r=1.5, w='mi', v='maj', reduction_alg=drop3, show=False)
df.to_csv('vowel_drop3.csv')


classify('vowel', k=5, show=False, reduction_alg=drop3)
classify('vowel', k=5, show=False, reduction_alg=fcnn)

#classify('vowel', k=5, show=False, reduction_alg=drop3)
#classify('vowel', k=5, show=False, reduction_alg=fcnn)
#classify('vowel', k=5, show=False, reduction_alg=fcnn)
#classify('vowel', k=1, show=False, reduction_alg=menn)
