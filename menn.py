import numpy as np
from sklearn.metrics import euclidean_distances as dist


def menn(x_train, y_train, k=1):

    delete_from_train = [False for _ in range(len(x_train))]

    all_distances = dist(x_train)
    print("all distances computed")
    for ix, x in enumerate(x_train):

        distances = all_distances[ix]
        # In order to make use of sample indexes (for x_train and y_train),
        # we assign the distance to the point itself as infinity to ignore it,
        # instead of deleting it
        distances[ix] = np.infty

        i = 1
        idx = distances.argmin()
        while i <= k:

            if (y_train[idx] != y_train[ix]):
                # print(f"Deleting {idx} because y_train[{idx}]={y_train[idx]} and y_train[{ix}]={y_train[ix]}")
                delete_from_train[ix] = True
                break

            # Assign it to infinity instead of delete it to keep the indexes synchronized
            min_dist = distances[idx]
            distances[idx] = np.infty

            next_idx = distances.argmin()
            # Increase k only if not reached the kth NN yet or if the next one has a greater distance (not 'l' element)
            if i < k or min_dist < distances[next_idx]:
                i += 1

            idx = next_idx

    reduced_x_train = np.delete(x_train, delete_from_train, axis=0)
    reduced_y_train = np.delete(y_train, delete_from_train, axis=0)
    delete_from_train = np.array(delete_from_train)
    print(f"MENN algorithm deleted {len(delete_from_train[delete_from_train == True])} elements of the initial {len(x_train)} from the reference set")
    return reduced_x_train, reduced_y_train
