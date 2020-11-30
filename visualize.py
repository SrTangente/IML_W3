from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def visualize(x_test, y_test, y_pred):
    """
    Plot a comparison between the real and the predicted values
    """
    # Visualize data
    tsne = TSNE()
    x_test_trans = tsne.fit_transform(x_test)
    fig, axis = plt.subplots(1, 2)
    axis[0].set_title("Real values")
    axis[0].scatter(x_test_trans[:, 0], x_test_trans[:, 1], c=y_test)
    axis[1].set_title("Predicted values")
    axis[1].scatter(x_test_trans[:, 0], x_test_trans[:, 1], c=y_pred)
    plt.show()
