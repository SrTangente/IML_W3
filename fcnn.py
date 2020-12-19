import numpy as np
from  sklearn.metrics import euclidean_distances


def centroid(class_x_train):
    return np.mean(class_x_train, axis=1)


def fcnn(x_train, y_train):

    # Add an index (to identify the point)
    #ids = np.array(range(len(x_train)))
    #x_train_aux = np.zeros(x_train.shape)
    #x_train_aux[:, 0] = ids
    #x_train_aux[:, 1:] = x_train
    #x_train = x_train_aux

    nearest = [None for p in x_train]
    S = set()
    centroids = [centroid(x_train[y_train == i][:, 1:]) for i in range(y_train.max())]
    delta_S = set(centroids)
    while delta_S:
        S = S.union(delta_S)
        rep = [None for p in S]
        qs = [q for q in x_train if q not in S]  # x_train - S
        for q in qs:
            for p in delta_S:
                if not nearest[q[0]]:
                    nearest[iq] = p
                elif nearest[q[0]] in centroids:
                    euclidean_distances(nearest[q[0]][1:], q[1:]) > euclidean_distances(p[1:], q[1:]):
            if y_train[iq] != y_train[nearest[iq]] and euclidean_distances(nearest[iq], q) < euclidean_distances(nearest[iq], rep[nearest[iq]]):
                rep[nearest[iq]] = q

        delta_S = []
        for p in S:
            if rep[p]:
                delta_S.append(rep[p])

        # x_train - p => distancia entre p y cada sample de x_train

        # guardarse indice del x_train al crear los subsets
