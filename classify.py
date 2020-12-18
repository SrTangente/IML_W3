
from read_datasets import *
from knn import kNNAlgorithm
from visualize import visualize
import pandas as pd


def classify(dataset, k=1, r=1, w='eq', v='maj', show=False, reduction_alg=None):
    """
    Execute a 10-fold kNN.
    :param dataset: either "adult" or "satimage"
    :param k: the number of nearest neighbours to take into account when voting
    :param show: boolean, whether to show a 2-plot of the results
    :return: None
    """

    df=pd.DataFrame(columns=['dataset','k','r','w','v','reduction','acc','eff','storage'])
    read = read_vowel_fold if dataset == 'vowel' else read_satimage_fold
    for i in range(10):
        x_train, y_train, x_test, y_test = read(i)
        if reduction_alg:
            n0=x_train.shape[0]
            x_train, y_train = reduction_alg(x_train, y_train, k)
            storage=x_train.shape[0]/n0
        else:
            storage=1
        # Call here the KNN
        indexes, y_pred, eff, acc = kNNAlgorithm(x_train,y_train, x_test,y_test, k, r, w, v)

        print(f"{dataset} fold {i}, k={k}, r={r}, w={w}, v={v}: Accuracy {acc}, Time {eff}")
        df=df.append(pd.DataFrame([[dataset,k,r,w,v,reduction_alg,acc,eff,storage]],columns=['dataset','k','r','w','v','reduction','acc','eff','storage']))
        if show:
            visualize(x_test, y_test, y_pred)
    return df