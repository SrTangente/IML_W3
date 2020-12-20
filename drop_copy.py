import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def drop2(x_train, y_train, k):
    subset = np.copy(x_train)
    subset = np.c_[subset, y_train]
    original = subset.copy()
    associates = {}
    n = len(x_train)
    for i in range(n):
        associates[i] = set()

    # Computing the distance to the nearest enemy
    def nearest_enemy_distance(point):
        point_y = point[-1]
        point_x = point[:-1]
        distinct_y = y_train[np.where(y_train != point_y)]
        distinct_x = x_train[np.where(y_train != point_y)]
        knn = KNeighborsClassifier(1)
        knn.fit(distinct_x, distinct_y)
        distance, index = knn.kneighbors([point_x], 1)
        return distance[0][0]

    subset = sorted(subset, key=lambda point: nearest_enemy_distance(point), reverse=True)
    subset = np.array(subset)

    knn = KNeighborsClassifier(k + 1)
    knn.fit(x_train, y_train)
    # building the list of nearest neighbors and associates
    for ind in range(n):
        x = subset[ind, :]
        distances, n_index = knn.kneighbors([x[:-1]], k + 1)
        for n_i in n_index[0]:
            associates[ind].add(n_i)
            associates[n_i].add(ind)
    nearests = associates.copy()
    nearests_temp = nearests.copy()

    for p_i in range(n):
        p = original[p_i, :]
        with_ = 0
        without_ = 0
        # compute instances well predicted with P
        for a_i in associates[p_i]:
            a = original[a_i, :]
            near_inst = [original[near, :] for near in nearests]
            # we already calculated k+1 neighbors before, so we can speed up the prediction prunning the examples
            y_near = [n[-1] for n in near_inst]
            pred = np.bincount(y_near).argmax()
            if pred == a[-1]:
                with_ += 1

        # compute instances correctly classified without p
        # first we remove p from the subset
        subset_without_p = np.delete(subset, np.where((subset == p).all(axis=1)), axis=0)
        knn_without_p = KNeighborsClassifier(k + 1)
        knn_without_p.fit(subset_without_p[:, :-1], subset_without_p[:, -1])
        for a_i in associates[p_i]:
            a = original[a_i, :]
            # then for each associate we get the k+1 NN
            distances, n_index = knn_without_p.kneighbors([a[:-1]], k + 1)
            # those are stored in nearest_temp dictionary
            nearests_temp[a_i] = set(n_index[0])
            # now we check if the prediction is correct
            y_near_temp = [original[nn, -1] for nn in n_index[0]]
            pred = np.bincount(y_near_temp).argmax()
            if pred == a[-1]:
                without_ += 1

        associates_temp = []
        if without_ >= with_:
            # p is removed from subset
            subset = subset_without_p.copy()
            # nearests are updated
            nearests = nearests_temp.copy()
            for a_i in associates[p_i]:
                # new points are added to associates without deleting anything
                [associates_temp.append([a_i, nn]) for nn in nearests[a_i]]

        [associates[at[0]].add(at[1]) for at in associates_temp]
    return subset[:, :-1], subset[:, -1]


def drop3(x_train, y_train, k):
    knn = KNeighborsClassifier(k)
    knn.fit(x_train, y_train)
    filtered_x = []
    filtered_y = []
    # remove the points that are not correctly classified
    for i in range(len(x_train)):
        x_i = x_train[i]
        y_i = y_train[i]
        if knn.predict([x_i])[0] == y_i:
            filtered_x.append(x_i)
            filtered_y.append(y_i)
    # apply drop2
    return drop2(np.array(filtered_x), np.array(filtered_y), k)
