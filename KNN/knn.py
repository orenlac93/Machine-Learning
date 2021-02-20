
# using Python 3.8

from math import inf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os


# compute euclidean distance (L2) between 2 points #
def euclidean_distance(x1, x2, y1, y2):
    return np.sqrt((np.sum(x1 - x2) ** 2) + np.sum(y1 - y2) ** 2)


# compute (L1) distance between 2 points #
def L_1_distance(x1, x2, y1, y2):
    return (np.abs((x1 - x2))) + (np.abs((y1 - y2)))


# compute (L infinity) distance between 2 points #
def L_inf_distance(x1, x2, y1, y2):
    return max(np.abs(x1 - x2), np.abs(y1 - y2))


# compute euclidean distances between point to each point in train #
def all_euclidean_distance(x_p, y_p, x_list=[], y_list=[]):
    distances_list = []
    for point_index in range(x_list.__len__()):
        ans = euclidean_distance(x_p, x_list[point_index], y_p, y_list[point_index])
        distances_list.append(ans)

    return distances_list


# compute (L1) distances between point to each point in train #
def all_L_1_distance(x_p, y_p, x_list=[], y_list=[]):
    distances_list = []
    for point_index in range(x_list.__len__()):
        ans = L_1_distance(x_p, x_list[point_index], y_p, y_list[point_index])
        distances_list.append(ans)

    return distances_list


# compute (L infinity) distances between point to each point in train #
def all_L_inf_distance(x_p, y_p, x_list=[], y_list=[]):
    distances_list = []
    for point_index in range(x_list.__len__()):
        ans = L_inf_distance(x_p, x_list[point_index], y_p, y_list[point_index])
        distances_list.append(ans)

    return distances_list


# get k min values indexes in list
def get_k_min_indexes(k_indexes, distances_list=[]):
    for d_ in range(distances_list.__len__()):
        distances_list[d_] = distances_list[d_] + random.choice(range(99)) * 0.0001
    copy_list = distances_list.copy()
    indexes_list = []
    for k_ in range(k_indexes):
        ans = min(copy_list)
        ans_index = distances_list.index(ans, 0, distances_list.__len__())
        indexes_list.append(ans_index)
        copy_list.remove(ans)

    return indexes_list


# compute prediction accuracy #
def get_prediction(label_list1=[], label_list2=[]):
    success_count = 0
    for label_indx in range(label_list1.__len__()):
        if label_list1[label_indx] == label_list2[label_indx]:
            success_count = success_count + 1

    return success_count / label_list1.__len__()


# load the HC Body Temperature data set

# data_HC = np.loadtxt("data/HC_Body_Temperature.txt")
data_HC = np.loadtxt("data/HC_Body_Temperature(changed).txt")


# all the data
x = []
y = []
label = []

# train data
x_train = []
y_train = []
label_train = []

# test data
x_test = []
y_test = []
label_test = []

x_nearest = []
y_nearest = []
label_nearest = []

# the algorithm test label classify
label_test_ans = []

# train and test size
train_size = 65
test_size = 65

# initialize data as points and labels
for line in data_HC:
    x.append(line[0])
    y.append(line[2])
    label.append(line[1])


# compute k nearest neighbors #
# k (number of neighbors)
# distance_type : 1 (L1)   2 (euclidean distance)   3 (L_infinity)
def knn(k, distance_type):

    # shuffle the data for split to train and test
    data_index_list = []
    for number in range(label.__len__()):
        data_index_list.append(number)

    random.shuffle(data_index_list)

    for i in range(train_size):
        x_train.append(x[data_index_list[i]])
        y_train.append(y[data_index_list[i]])
        label_train.append(label[data_index_list[i]])

    for j in range(label.__len__()):
        if j > train_size and label_test.__len__() < test_size:
            x_test.append(x[data_index_list[j]])
            y_test.append(y[data_index_list[j]])
            label_test.append(label[data_index_list[j]])

    for test_point in range(label_test.__len__()):
        min_distance = inf

        # get the distances from the current test point to all the train points
        if distance_type == 1:
            distances = all_L_1_distance(x_test[test_point], y_test[test_point], x_train, y_train)
        elif distance_type == 2:
            distances = all_euclidean_distance(x_test[test_point], y_test[test_point], x_train, y_train)
        else:
            distances = all_L_inf_distance(x_test[test_point], y_test[test_point], x_train, y_train)

        # get the nearest k points indexes
        nearest_k_indexes = get_k_min_indexes(k, distances)

        # debug printing #
        # print('distances: ', distances)
        # print(k, ' nearest (indexes in train):', nearest_k_indexes, '   to (index ', test_point, ' in test)')

        # find if there is more blue or more red neighbors
        count1 = 0
        count2 = 0
        for near in range(k):
            if label_train[nearest_k_indexes[near]] == 1.0:
                count1 = count1 + 1
            else:
                count2 = count2 + 1

        # debug printing
        # print('1s: ', count1)
        # print('2s: ', count2)
        #

        if count1 > count2:
            label_test_ans.append(1.0)
        else:
            label_test_ans.append(2.0)

        # plotting the points
        for index in range(label_train.__len__()):
            if label_train[index] == 1:
                plt.scatter(x_train[index], y_train[index], s=10, c='blue')  # blue (positive  1)
            else:
                plt.scatter(x_train[index], y_train[index], s=10, c='red')  # red (negative  2)

        plt.scatter(x_test[test_point], y_test[test_point], s=15, c='black')

        for point in range(k):
            plt.plot([x_train[nearest_k_indexes[point]], x_test[test_point]],
                     [y_train[nearest_k_indexes[point]], y_test[test_point]], c='black')

        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')

        # giving a title to the graph
        plt.title('nearest neighbors from train points:')

        if test_point % test_size == 0:
            # function to show the plot
            plt.show(block=False)
            plt.pause(10)
        plt.close()

    # debug printing
    # print('the real test labels:')
    # print(label_test)
    # print('the answer test label:')
    # print(label_test_ans)

    print('accuracy rate: ', get_prediction(label_test, label_test_ans) * 100, ' %')
    label_test_ans.clear()

    x_test.clear()
    y_test.clear()
    label_test.clear()
    x_train.clear()
    y_train.clear()
    label_train.clear()


print('k = 1, L = 1 :')
knn(1, 1)
print('k = 1, L = 2 :')
knn(1, 2)
print('k = 1, L = inf :')
knn(1, 3)

print('k = 3, L = 1 :')
knn(3, 1)
print('k = 3, L = 2 :')
knn(3, 2)
print('k = 3, L = inf :')
knn(3, 3)

print('k = 5, L = 1 :')
knn(5, 1)
print('k = 5, L = 2 :')
knn(5, 2)
print('k = 5, L = inf :')
knn(5, 3)

print('k = 7, L = 1 :')
knn(7, 1)
print('k = 7, L = 2 :')
knn(7, 2)
print('k = 7, L = inf :')
knn(7, 3)

print('k = 9, L = 1 :')
knn(9, 1)
print('k = 9, L = 2 :')
knn(9, 2)
print('k = 9, L = inf :')
knn(9, 3)


