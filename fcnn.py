import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as dist
from numpy.linalg import norm


def centroid(class_x_train):
    mean = np.mean(class_x_train, axis=0, keepdims=True)
    distances = dist(class_x_train, mean)
    return np.argmin(distances)


def fcnn(x_train, y_train,k,r,w,v):
    #k, r, w, v are dummies to integrate all reduction algs
    #Vectorized implementation for a more interpretable implementation see below
    n_classes = int(y_train.max() + 1)
    n_samples = x_train.shape[0]
    T = np.concatenate([x_train, np.reshape(y_train, [n_samples, 1])], axis=1)
    ind_delta_S = np.array([centroid(x_train[y_train == i]) for i in range(n_classes)])
    X = T[:, :-1]
    Y = T[:, -1].astype(int)
    ind_T = np.array([i for i in range(n_samples)])
    ind_nearest = np.ones([n_classes + n_samples], dtype=int)
    min_distances = np.inf * np.ones([n_classes + n_samples])
    ind_S = np.array([])

    while ind_delta_S.shape[0] != 0:


        ind_S = np.array(list(set(ind_S.tolist()).union(set(ind_delta_S.tolist()))))
        ind_Q = np.array(list(set(ind_T.tolist()) - set(ind_S.tolist())))
        distances = np.min(dist(X[ind_Q], X[ind_delta_S]), axis=1)
        ind_distances = np.array([ind_delta_S[i] for i in np.argmin(dist(X[ind_Q], X[ind_delta_S]), axis=1)]).astype(
            int)
        ind_change_nearest = min_distances[ind_Q] > distances
        min_distances[ind_Q[ind_change_nearest]] = distances[ind_change_nearest]
        ind_nearest[ind_Q[ind_change_nearest]] = ind_distances[ind_change_nearest]
        # We do the same for representatives

        ind_missmatch = ind_Q[np.logical_not(np.equal(Y[ind_Q], Y[ind_nearest[ind_Q]]))]  # Different labels

        ind_rep = np.array([], dtype=int)

        def_rep = np.unique(ind_nearest[ind_Q])
        delete=[]
        for m,n in enumerate(def_rep):
            n_indexes = ind_Q[np.where(ind_nearest[ind_Q] == n)[0]]
            n_indexes = np.array(list(set(n_indexes.tolist()).intersection(set(ind_missmatch.tolist()))))
            if n_indexes.size == 0:
                delete.append(m)
                continue
            rep_distances = np.linalg.norm(X[ind_nearest[n_indexes]] - X[n_indexes], axis=1)
            ind_rep = np.concatenate([ind_rep, [n_indexes[np.argmin(rep_distances)]]])
        def_rep=np.delete(def_rep,delete)
        ind_delta_S = ind_rep[np.in1d(def_rep, ind_S)]

    # Return final subset
    x_train = np.stack(X[ind_S])
    y_train = np.array(Y[ind_S])
    return x_train, y_train


def fcnn_slow(x_train, y_train):
    n_classes = int(y_train.max() + 1)
    n_samples = x_train.shape[0]
    train = np.concatenate([x_train, np.reshape(y_train, [n_samples, 1])], axis=1)

    nearest = {}
    for p in train:
        nearest[str(p)] = None

    S = {}
    delta_S = {
        str(train[centroid(x_train[y_train == i])]): train[centroid(x_train[y_train == i])]
        for i in range(n_classes)}

    while delta_S:
        for s in delta_S:
            S[s] = delta_S[s]
            delete = [delta_S[s].tolist() == row for row in train.tolist()]
            train = np.delete(train, delete, axis=0)

        rep = {}
        for p in S:
            rep[str(p)] = None
        for q in train:
            for p in delta_S:
                p = delta_S[p]
                if (nearest[str(q)] is None) or norm(nearest[str(q)][:-1] - q[:-1]) > norm(p[:-1] - q[:-1]):
                    nearest[str(q)] = p

            if q[-1] != nearest[str(q)][-1] and (
                    rep[str(nearest[str(q)])] is None or norm(nearest[str(q)][:-1] - q[:-1]) < norm(
                nearest[str(q)][:-1] - rep[str(nearest[str(q)])][:-1])):
                rep[str(nearest[str(q)])] = q

        delta_S = {}
        for p in S:
            if rep[p] is not None:
                delta_S[str(rep[p])] = rep[p]
        pass
    # Return final subset
    train = np.array(list(S.values()))

    return train[:, :-1], train[:, -1]
