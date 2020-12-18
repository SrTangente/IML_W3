from read_datasets import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from visualize import visualize


def classify(dataset, k=5, show=False, reduction_alg=None):
    """
    Execute a 10-fold kNN.
    :param dataset: either "adult" or "vowel"
    :param k: the number of nearest neighbours to take into account when voting
    :param show: boolean, whether to show a 2-plot of the results
    :return: None
    """

    read = read_vowel_fold if dataset == 'vowel' else read_satimage_fold
    for i in range(10):
        x_train, y_train, x_test, y_test = read(i)
        if reduction_alg:
            x_train, y_train = reduction_alg(x_train, y_train, k)
        # Call here the KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        print(f"Iteration {i}: {accuracy_score(y_test, y_pred)}")

        if show:
            visualize(x_test, y_test, y_pred)