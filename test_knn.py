import read_datasets
import knn

x_train,y_train, x_test,y_test=read_datasets.read_adult_fold(0)
a=knn.knn(x_train,y_train, x_test,y_test,1,1,'chi','inv')
