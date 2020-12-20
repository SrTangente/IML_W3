from classify import *
from drop import drop2, drop3
from fcnn import fcnn
from menn import menn

df = classify('vowel', k=1, r=1.5, w='mi', v='maj', reduction_alg=drop3, show=False)
df.to_csv('vowel_drop3.csv')

classify('vowel', k=5, show=False, reduction_alg=drop3)
classify('vowel', k=5, show=False, reduction_alg=fcnn)

#classify('vowel', k=5, show=False, reduction_alg=drop3)
#classify('vowel', k=5, show=False, reduction_alg=fcnn)
#classify('vowel', k=1, show=False, reduction_alg=menn)
