from read_datasets import *

def classify(dataset):
    read = read_vowel_fold if dataset == 'vowel' else read_adult_fold
    for i in range(10):
        train, test = read(i)
        print('Train:')
        print(train[:10, :])
        print('--------------')
        print('Test:')
        print(test[:10, :])
        print('--------------')
        # Call here the KNN