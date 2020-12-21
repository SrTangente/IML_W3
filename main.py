from classify import *
from fcnn import *
from menn import *
from drop import *
from analysis import *
from stats import *

def input_dataset():
    print("Select dataset to which execute KNN:")
    print("1 vowel")
    print("2 satimage")
    num_dataset = 0
    while True:
        try:
            num_dataset = int(input("Dataset number: "))
        except:
            print("Incorrect value. It has to be one of (1, 2).")
            continue
        if num_dataset in range(1, 3):
            break
        else:
            print("Incorrect value. It has to be one of (1, 2).")

    return num_dataset


def input_reduction():
    print("------------------------")
    print("Select reduction:")
    print("1 None")
    print("2 FCNN")
    print("3 MENN")
    print("4 DROP3")
    num_reduction = 0
    while True:
        try:
            num_reduction = int(input("Reduction number: "))
        except:
            print("Incorrect value. It has to be one of (1, 2, 3, 4).")
            continue
        if num_reduction in range(1, 5):
            break
        else:
            print("Incorrect value. It has to be one of (1, 2, 3, 4).")

    return num_reduction


def execute(num_dataset, num_reduction):
    """
    Execute the selected visualization on the selected dataset with the best parameters obtained
    :param num_dataset: the number of the dataset selected
    :param num_visualization: the number of the dataset selected
    :return: the tagged_data predicted and the real classes of the dataset
    """
    reduction_algs = [None, None, fcnn, menn, drop3]
    if num_dataset==1:
        reductionKNNAlgorithm('vowel', k=1, r=1.5, w='mi', v='maj', show=False, reduction_alg=reduction_algs[num_reduction])
    else:
        reductionKNNAlgorithm('satimage', k=5, r=1.0, w='eq', v='shep', show=False, reduction_alg=reduction_algs[num_reduction])

if __name__ == "__main__":
    print("Select action perform:")
    print("1 execute KNN")
    print("2 see results summary")
    action = 0
    while True:
        try:
            action = int(input("Action number: "))
        except:
            print("Incorrect value. It has to be one of (1, 2).")
            continue
        if action in range(1, 3):
            break
        else:
            print("Incorrect value. It has to be one of (1, 2).")

    if(action==1):
        num_dataset = input_dataset()
        num_reduction = input_reduction()
        execute(num_dataset, num_reduction)
    else:
        print('*****************************************************************')
        print('Result summary:')
        analize()
        print('******************************************************************')
        print('Statistical tests')
        stats()

