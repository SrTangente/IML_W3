import numpy as np
from sklearn.metrics import euclidean_distances as dist


def centroid(class_x_train):
    return np.mean(class_x_train, axis=1)


def fcnn(x_train, y_train):

    # Add an index (to identify the point)
    #ids = np.array(range(len(x_train)))
    #x_train_aux = np.zeros(x_train.shape)
    #x_train_aux[:, 0] = ids
    #x_train_aux[:, 1:] = x_train
    #x_train = x_train_aux

    # 'nearest' is the dictionary that contains the nearest point within a subset (which can be a centroid, too) for each point
    # 'labels' is the dictionary that contains the y_train label of a point (the centroid labels are added later)
    nearest = {}
    labels = {}
    for idx, p in enumerate(x_train):
        nearest[id(p)] = None
        labels[id(p)] = y_train[idx]

    S = set()

    centroids = []
    for i in range(y_train.max()):
        cent = centroid(x_train[y_train == i])
        centroids.append(cent)
        # add the label of the centroid
        labels[id(cent)] = i


    delta_S = set(centroids)
    while delta_S:
        S = S.union(delta_S)
        rep = {}
        for p in S:
            rep[id(p)] = None
        qs = [q for q in x_train if q not in S]  # x_train - S
        for q in qs:
            for p in delta_S:
                if not nearest[id(q)] or dist(nearest[id(q)], q) > dist(p, q):
                    nearest[id(q)] = p

            if labels[id(q)] != labels[id(nearest[id(q)])] and dist(nearest[id(q)], q) < dist(nearest[id(q)], rep[id(nearest[id(q)])]):
                rep[id(nearest[id(q)])] = q

        delta_S = set()
        for p in S:
            if rep[id(p)]:
                delta_S.add(rep[id(p)])
