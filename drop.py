import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def drop2(x_train, y_train, k):
    subset = np.copy(x_train)
    subset = np.c_[subset, y_train]
    associates = {}
    for p in x_train:
        associates[p] = set([])

    # Computing the distance to the nearest enemy
    def nearest_enemy_distance(point):
        point_y = point[-1]
        point_x = point[:-1]
        distinct_x = x_train[np.where(y_train != point_y)]
        distinct_y = y_train[np.where(y_train != point_y)]
        knn = KNeighborsClassifier(1)
        knn.fit(distinct_x, distinct_y)
        distance, index = knn.kneighbors(point_x, 1)
        return distance

    subset.sort(key=nearest_enemy_distance, reverse=True)

    knn = KNeighborsClassifier(k + 1)
    knn.fit(subset[:, :-1], subset[:, -1])
    # building the list of nearest neighbors and associates
    for x in subset:
        distances, n_index = knn.kneighbors(x, k + 1)
        neighbors = [subset[i] for i in n_index]
        for n in neighbors:
            associates[n].add(x)
            associates[x].add(n)
    nearests = associates.copy()
    nearests_temp = nearests.copy()

    for p in subset:
        with_ = 0
        without_ = 0
        # compute instances well predicted with P
        for a in associates[p]:
            # we already calculated k+1 neighbors before, so we can speed up the prediction prunning the examples
            y_near = [n[-1] for n in nearests[a]]
            pred = np.bincount(y_near).argmax()
            if pred == a[-1]:
                with_ += 1

        # compute instances correctly classified without p
        # first we remove p from the subset
        subset_without_p = np.delete(subset, np.where(subset == p))
        knn_without_p = KNeighborsClassifier(k + 1)
        knn_without_p.fit(subset_without_p[:, :-1], subset_without_p[:, -1])
        for a in associates[p]:
            # the for each associate we get the k+1 NN
            distances, n_index = knn_without_p.kneighbors(a, k + 1)
            # those are stored in nearest_temp dictionary
            nearests_temp[a] = set([subset[i] for i in n_index])
            # now we check if the prediction is correct
            y_near_temp = [n[-1] for n in nearests_temp[a]]
            pred = np.bincount(y_near_temp).argmax()
            if pred == a[-1]:
                without_ += 1

        if without_ >= with_:
            # p is removed from subset
            subset = subset_without_p.copy()
            # nearests are updated
            nearests = nearests_temp.copy()
            for a in associates[p]:
                # new points are added to associates without deleting anything
                [associates[a].add(nn) for nn in nearests[a]]

    return subset


def drop3(x_train, y_train, k):
    knn = KNeighborsClassifier(k)
    knn.fit(x_train, y_train)
    filtered_x = []
    filtered_y = []
    for i in range(len(x_train)):
        x_i = x_train[i]
        y_i = y_train[i]
        if knn.predict(x_i) == y_i:
            filtered_x.append(x_i)
            filtered_y.append(y_i)

    return drop2(filtered_x, filtered_y, k)
