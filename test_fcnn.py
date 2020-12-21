import fcnn
import read_datasets

x_train,y_train, x_test,y_test=read_datasets.read_satimage_fold(0)
a=fcnn.fcnn1(x_train,y_train)