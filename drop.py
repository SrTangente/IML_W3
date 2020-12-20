import numpy as np
from knn import kNNeighbours


def drop2(x_train, y_train, k, r, w, v):
    n = len(x_train)
    print('Initial length (drop2):', n)
    # an extra column is added to the subset to keep track of the original point
    true_indexes = [i for i in range(n)]
    original = np.c_[np.copy(x_train), y_train]
    subset = np.c_[np.copy(original), true_indexes]
    associates = {}
    for i in range(n):
        associates[i] = set()

    # building the list of nearest neighbors and associates
    for ind in range(n):
        x = subset[ind, :]
        distances, n_index, preds = kNNeighbours(x_train, y_train, [x[:-2]], k + 1, r, w, v)
        for n_i in n_index[0]:
            associates[ind].add(n_i)
            associates[n_i].add(ind)
    nearests = associates.copy()
    nearests_temp = nearests.copy()

    # Computing the distance to the nearest enemy
    def nearest_enemy_distance(point):
        point_y = point[-2]
        point_x = point[:-2]
        distinct_x = x_train[np.where(y_train != point_y)]
        distinct_y = y_train[np.where(y_train != point_y)]
        distance, index, prds = kNNeighbours(distinct_x, distinct_y, [point_x], 1, r, w, v)
        return distance[0][0]

    subset = sorted(subset, key=lambda point: nearest_enemy_distance(point), reverse=True)
    subset = np.array(subset)

    for p_i in range(n):
        p = original[p_i, :]
        with_ = 0
        without_ = 0
        # compute instances well predicted with P
        for a_i in associates[p_i]:
            a = original[a_i, :]
            near_inst = np.array([original[near, :] for near in nearests])
            # we already calculated k+1 neighbors before, so we can speed up the prediction prunning the examples
            distances, n_index, preds = kNNeighbours(near_inst[:, :-1], near_inst[:, -1], [a[:-1]], k + 1, r, w, v)
            pred = preds[0]
            if pred == a[-1]:
                with_ += 1

        # compute instances correctly classified without p
        # first we remove p from the subset
        subset_without_p = np.delete(subset, np.where((subset[:, :-1] == p).all(axis=1)), axis=0)
        for a_i in associates[p_i]:
            a = original[a_i, :]
            # then for each associate we get the k+1 NN
            distances, n_index, preds = kNNeighbours(subset_without_p[:, :-2], subset_without_p[:, -2],
                                                     [a[:-1]], k + 1, r, w, v)
            #fix indexes to get true ones
            true_ind = [int(subset_without_p[j, -1]) for j in n_index[0]]
            # those are stored in nearest_temp dictionary
            nearests_temp[a_i] = set(true_ind)
            # now we check if the prediction is correct
            pred = preds[0]
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

        [associates[at[0]].add(int(at[1])) for at in associates_temp]
    print('After reduction length :', len(subset))
    return subset[:, :-2], subset[:, -2]


def drop3(x_train, y_train, k, r, w, v):
    print('Initial length (drop3): ')
    filtered_x = []
    filtered_y = []
    # remove the points that are not correctly classified
    for i in range(len(x_train)):
        x_i = x_train[i]
        y_i = y_train[i]
        distance, index, prds = kNNeighbours(x_train, y_train, [x_i], k, r, w, v)
        if prds[0] == y_i:
            filtered_x.append(x_i)
            filtered_y.append(y_i)
    # apply drop2
    return drop2(np.array(filtered_x), np.array(filtered_y), k, r, w, v)
