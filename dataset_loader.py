# Imanuel Dantuma
# 012595685
# CECS 551 - Semester Project
# Benchmark Comparison on CIFAR-100

#### Library imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.utils import to_categorical

#### Importing Dataset
from keras.datasets import cifar100

def download_dataset():
    print("Downloading Dataset......")
    (x1_train, y1_train), (x1_test, y1_test) = cifar100.load_data()
    print("Dataset Downloaded.\n\n")

    x_train = [np.zeros((32,32,3))]
    y_train = [np.zeros((1))]

    x_test = [np.zeros((32, 32, 3))]
    y_test = [np.zeros((1))]

    print("Selecting 50 Classes from Training Dataset")
    for counter,element in enumerate(y1_train):
        if(counter%500 == 0):
            message = "Progress = " + str(counter/500) + "%"
            print(message, end='\r')
        if(element < 50):
            x_train = np.concatenate((x_train, [x1_train[counter]]), axis=0)
            y_train = np.concatenate((y_train,[y1_train[counter]]), axis=0)

    x_train = np.delete(x_train, 0, 0)
    y_train = np.delete(y_train, 0, 0)

    np.save("../CIFAR-100 Project/data/train/x_train", x_train)
    np.save("../CIFAR-100 Project/data/train/y_train", y_train)

    print("Data is stored in ../CIFAR-100 Project/data/train/\n\n")

    print("Selecting 50 Classes from Testing Dataset")
    for counter, element in enumerate(y1_test):
        if (counter % 100 == 0):
            message = "Progress = " + str(counter / 100) + "%"
            print(message, end='\r')
        if (element < 50):
            x_test = np.concatenate((x_test, [x1_test[counter]]), axis=0)
            y_test = np.concatenate((y_test, [y1_test[counter]]), axis=0)

    x_test = np.delete(x_test, 0, 0)
    y_test = np.delete(y_test, 0, 0)

    np.save("../CIFAR-100 Project/data/test/x_test", x_test)
    np.save("../CIFAR-100 Project/data/test/y_test", y_test)

    print("Data is stored in ../CIFAR-100 Project/data/test/\n\n")


def load_dataset():
    num_of_classes = 50

    print("Loading Dataset....")
    x_train = np.load("../CIFAR-100 Project/data/train/x_train.npy")
    y_train = np.load("../CIFAR-100 Project/data/train/y_train.npy")

    x_test = np.load("../CIFAR-100 Project/data/test/x_test.npy")
    y_test = np.load("../CIFAR-100 Project/data/test/y_test.npy")
    print("Dataset has been loaded.\n")

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, num_of_classes)
    y_test = to_categorical(y_test, num_of_classes)

    return (x_train, y_train), (x_test, y_test)
