from classify import *
from drop import drop2, drop3
from fcnn import fcnn

#classify('vowel', k=5, show=False, reduction_alg=drop3)
classify('vowel', k=5, show=False, reduction_alg=fcnn)
