from read_datasets import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

def classify(dataset, k=5):
    read = read_vowel_fold if dataset == 'vowel' else read_adult_fold
    for i in range(10):
        train, test = read(i)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]
        print('Train:')
        print(train[:10, :])
        print('--------------')
        print('Test:')
        print(test[:10, :])
        print('--------------')
        # Call here the KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        print(accuracy_score(y_test, y_pred))
