from classify import *
from drop import drop2, drop3
from fcnn import fcnn
from menn import menn

df = pd.DataFrame()
for k in [1,3,5,7]:
    df=pd.concat([df,classify_fast('satimage', 5, 1, 'eq', 'shep', reduction_alg=menn)])
df.to_csv('satimage_menn.csv')

df = pd.DataFrame()
for k in [1,3,5, 7]:
    df=pd.concat([df,classify_fast('vowel',1, 1.5, 'mi', 'maj', reduction_alg=menn)])
df.to_csv('vowel_menn.csv')

df = pd.DataFrame()
df=pd.concat([df,classify_fast('satimage', 5, 1, 'eq', 'shep', reduction_alg=fcnn)])
df.to_csv('satimage_fcnn.csv')

df = pd.DataFrame()
df=pd.concat([df,classify_fast('vowel',1, 1.5, 'mi', 'maj', reduction_alg=fcnn)])
df.to_csv('vowel_fcnn.csv')

df = pd.DataFrame()
for k in [3]:
    df=pd.concat([df,classify_fast('satimage', 5, 1, 'eq', 'shep', reduction_alg=fcnn)])
df.to_csv('satimage_drop3_3.csv')

df = pd.DataFrame()
for k in [3]:
    df=pd.concat([df,classify_fast('vowel',1, 1.5, 'mi', 'maj', reduction_alg=fcnn)])
df.to_csv('vowel_drop3_3.csv')

dfs = classify('satimage', k=5, r=1, w='eq', v='shep', reduction_alg=drop3, show=False)
dfs.to_csv('satimage_drop3.csv')
