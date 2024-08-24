import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
def load_mnist(debug:bool=False):
    #get train data
    train = pd.read_csv('Datasets/MNIST/mnist_train.csv')
    y_train = train.iloc[:,0]
    x_train = train.iloc[: , 1:]
    x_train_scaled = scale_data(x_train)

    #get test data
    test = pd.read_csv('Datasets/MNIST/mnist_test.csv')
    y_test = test.iloc[:,0]
    x_test = test.iloc[: , 1:]
    x_test_scaled = scale_data(x_test)

    return [x_train_scaled,x_test_scaled,y_train,y_test]

def load_fashion(debug:bool=False):
    #get train data
    train = pd.read_csv('Datasets/FashionMNIST/fashion-mnist_train.csv')
    y_train = train.iloc[:,0]
    x_train = train.iloc[: , 1:]
    x_train_scaled = scale_data(x_train)

    #get test data
    test = pd.read_csv('Datasets/FashionMNIST/fashion-mnist_test.csv')
    y_test = test.iloc[:,0]
    x_test = test.iloc[: , 1:]
    x_test_scaled = scale_data(x_test)

    return [x_train_scaled,x_test_scaled,y_train,y_test]

def load_cifar_10():
    #get train data
    train = pd.read_csv('Datasets/CIFAR10/cifar10_train.csv')
    y_train = train.iloc[:, -1]
    x_train = train.iloc[:, :-1]
    x_train_scaled = scale_data(x_train)

    #get test data
    test = pd.read_csv('Datasets/CIFAR10/cifar10_test.csv')
    y_test = test.iloc[:,-1]
    x_test = test.iloc[: , :-1]
    x_test_scaled = scale_data(x_test)

    return [x_train_scaled,x_test_scaled,y_train,y_test]


def load_cifar_100_coarse():
    data_pre_path = 'Datasets/CIFAR100/' # change this path
    # File paths
    data_train_path = data_pre_path + 'train'
    data_test_path = data_pre_path + 'test'
    # Read dictionary
    data_train_dict = unpickle(data_train_path)
    data_test_dict = unpickle(data_test_path)
    # Get data (change the coarse_labels if you want to use the 100 classes)
    x_train = data_train_dict[b'data']
    y_train = np.array(data_train_dict[b'coarse_labels'])
    x_train_scaled = scale_data(pd.DataFrame(x_train))

    x_test = data_test_dict[b'data']
    y_test = np.array(data_test_dict[b'coarse_labels'])
    x_test_scaled = scale_data(pd.DataFrame(x_test))
    return [x_train_scaled,x_test_scaled,y_train,y_test]

def load_cifar_100_fine():
    data_pre_path = 'Datasets/CIFAR100/' # change this path
    # File paths
    data_train_path = data_pre_path + 'train'
    data_test_path = data_pre_path + 'test'
    # Read dictionary
    data_train_dict = unpickle(data_train_path)
    data_test_dict = unpickle(data_test_path)
    # Get data (change the coarse_labels if you want to use the 100 classes)
    x_train = data_train_dict[b'data']
    y_train = np.array(data_train_dict[b'fine_labels'])
    x_train_scaled = scale_data(pd.DataFrame(x_train))

    x_test = data_test_dict[b'data']
    y_test = np.array(data_test_dict[b'fine_labels'])
    x_test_scaled = scale_data(pd.DataFrame(x_test))
    return [x_train_scaled,x_test_scaled,y_train,y_test]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def scale_data(data: pd.DataFrame,):
    """Scales the data using standard scaler."""
    min_max_scaler = preprocessing.StandardScaler()
    scaled_data = pd.DataFrame(min_max_scaler.fit_transform(data.values))
    
    return scaled_data