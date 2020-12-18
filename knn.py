from sklearn.feature_selection import mutual_info_classif,chi2
import numpy as np
import time


def kNNAlgorithm(x_train,y_train, x_test,y_test, k, r, w, v):
    start = time.time()

    n_classes = len(np.unique(y_train))
    n_feat = x_train.shape[1]
    n_test = x_test.shape[0]

    # Set distance weights
    if w == 'eq':
        d_weights = np.ones([n_feat, 1])
    elif w == 'mi':
        d_weights = mutual_info_classif(x_train, y_train).reshape([n_feat, 1])
    elif w == 'chi':
        d_weights = chi2(x_train, y_train)[0].reshape([n_feat, 1])
    else:
        raise Exception("Invalid distance weighting, choose eq for equal, mi for mutual information or chi for chi2")
    # Calculate distances
    distances = np.zeros([n_test, k])
    knn_labels = np.zeros([n_test, k])
    indexes = np.zeros([n_test, k],dtype=np.int64)

    for i in range(n_test):
        # Get deltas for each feature
        delta = (abs(x_train - x_test[i])) ** r
        # Get weighted sum and take r root
        wdelta = (np.matmul(delta, d_weights)) ** (1 / r)
        # Get indices of k lowest distances
        a=np.squeeze(np.argsort(np.transpose(wdelta)))[:k]
        indexes[i,:] = np.squeeze(np.argsort(np.transpose(wdelta)))[:k]
        # Store distances and labels
        distances[i, :] = np.squeeze(wdelta[indexes[i,:]])
        knn_labels[i, :] = np.squeeze(y_train[indexes[i,:]])

    # Set voting weights
    if v == 'maj':
        v_weights = np.ones([n_test, k])
    elif v == 'inv':
        v_weights = 1 / np.maximum(distances, np.finfo(float).eps)
    elif v == 'shep':
        v_weights = np.exp(-distances)
    else:
        raise Exception(
            "Invalid voting weighting, choose maj for majority, inv for inverse distances or shep for Sheppard's work")

    # Get voting scores
    scores = np.zeros([n_test, n_classes])
    for i in range(n_test):
        for j in range(n_classes):
            v_index = np.where(knn_labels[i,:] == j)
            scores[i, j] = np.sum(v_weights[i,v_index])

    # To break ties choose the lowest index. We do this like sklearn's implementation.
    preds = np.argmax(scores, axis=1)
    end = time.time()

    acc = np.sum(preds == y_test) / n_test

    return indexes, preds, end - start, acc
