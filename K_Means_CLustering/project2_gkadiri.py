from scipy import io
import matplotlib.pyplot as plt
import numpy as np

temp = io.loadmat('AllSamples.mat')                                                         # Importing Data
resource = temp['AllSamples']
dataset_length = len(resource)
np.random.seed(0)
colors = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'red', 'brown', 'gray', 'violet', 'pink']

plt.scatter(resource[:, 0], resource[:, 1], c = colors[6])
plt.title('Given data scatter plot')
plt.grid()
plt.show()

###
### STRATEGY 1
###

def points_dis(k_means, key, bunch):
    interval = [0]*k_means
    for i in range(k_means):
        interval[i] = np.sum(
            (resource[np.where(bunch == i)[0]]-key[i])**2)
    return sum(interval)

def k_means_start_1(k_means, reps=50):                                                      # Objective Function for Strat 1
    key_m = np.random.choice(dataset_length, k_means)
    key = resource[key_m]
    bunch = None
    separate_total = []
    for _ in range(reps):
        interval = np.array([0]*k_means*dataset_length, dtype='float').reshape(k_means, dataset_length)
        for i, j in enumerate(key):
            interval[i] = np.sum((resource-j)**2, axis=1)
        interval = np.array(interval).T
        bunch = np.argmin(interval, axis= 1)
        for i in range(k_means):
            key[i] = np.mean(resource[np.where(bunch == i)[0]], axis=0)
        separate_total.append(points_dis(k_means, key, bunch))
    rating = points_dis(k_means, key, bunch)
    return key, bunch, separate_total, rating

bunch_1 = []                                                                                # First Initialization for STRAT 1
rating_1 = []

for k_means in range(2, 11):
    key, bunch, separate_total, rating = k_means_start_1(
        k_means)
    bunch_1.append(key)
    rating_1.append(rating)
    plt.subplot(3,3, k_means-1)
    plt.grid()
    for i in range(k_means):
        x = resource[np.where(bunch == i)[0]]
        plt.scatter(x[:, 0], x[:, 1], c = colors[i])
    # for i in range(k_means):
    #     plt.plot(separate_total, c = colors[i])
plt.show()

plt.plot(range(2, 11), rating_1)
plt.xlabel('Number of Clusters')
plt.ylabel('Onjective Score')
plt.grid()
plt.title('STRATEGY 1 - First Initialization Score plot')
plt.show()

bunch_2 = []                                                                                # 2nd Initialization for STRAT 1
rating_2 = []
for k_means in range(2, 11):
    key, bunch, separate_total, rating = k_means_start_1(
        k_means)
    bunch_2.append(key)
    rating_2.append(rating)

    plt.subplot(3, 3, k_means-1)
    plt.grid()
    for i in range(k_means):
        x = resource[np.where(bunch == i)[0]]
        plt.scatter(x[:, 0], x[:, 1], c = colors[i])
    # for i in range(k_means):
    #     plt.plot(separate_total, c = colors[i])
plt.show()

plt.plot(range(2, 11), rating_2)
plt.xlabel('Number of Clusters')
plt.ylabel('Objective Score')
plt.title('STRATEGY1: Second initialization Score plot')
plt.grid()
plt.show()

for i in range(9):                                                                              # Cluster locations
    plt.subplot(3, 3, i+1)
    plt.grid()
    plt.scatter(bunch_1[i][:, 0], bunch_1[i][:, 1])
    plt.grid()
    plt.scatter(bunch_2[i][:, 0], bunch_2[i][:, 1])
    plt.grid()
plt.show()

###
### STRATEGY 2
###

def create_randoms_centro(k_means):
    reso = resource
    key_m = []
    key_m.extend(np.random.choice(dataset_length, 1))
    key = reso[key_m]
    reso = np.delete(reso, key_m, axis=0)
    for i in range(1, k_means):
        previous = np.mean(key, axis=0)
        interval = np.sum((reso-previous)**2, axis=1)
        next = np.argmax(interval)
        key = np.append(key, [reso[next]], axis=0)
        reso = np.delete(reso, [next], axis=0)
    return key


def k_means_start_2(k_means, reps=50):                                                      # Objective func for START 2
    key = create_randoms_centro(k_means)
    bunch = None
    separate_total = []
    for _ in range(reps):
        interval = np.array([0]*k_means*dataset_length, dtype='float').reshape(k_means, dataset_length)
        for i, j in enumerate(key):
            interval[i] = np.sum((resource-j)**2, axis=1)
        interval = np.array(interval).T
        bunch = np.argmin(interval, axis=1)
        for i in range(k_means):
            key[i] = np.mean(resource[np.where(bunch == i)[0]], axis=0)
        separate_total.append(points_dis(k_means, key, bunch))
    rating = points_dis(k_means, key, bunch)
    return key, bunch, separate_total, rating


bunch_1 = []                                                                                # First Initialization for STRAT 2
rating_1 = []
for k_means in range(2, 11):
    key, bunch, separate_total, rating = k_means_start_2(
        k_means)
    bunch_1.append(key)
    rating_1.append(rating)
    plt.subplot(3, 3, k_means-1)
    plt.grid()
    for i in range(k_means):
        x = resource[np.where(bunch == i)[0]]
        plt.scatter(x[:, 0], x[:, 1], c = colors[i])
    # for i in range(k_means):
    #     plt.plot(separate_total, c = colors[i])
plt.show()

plt.plot(range(2, 11), rating_1, c = colors[6])
plt.xlabel('Number of Clusters')
plt.ylabel('Objective Score')
plt.title('Strategy 2: 1st initialization objective function vs number of clusters')
plt.show()

bunch_2 = []                                                                                # Second Initialization for STRAT 2
rating_2 = [] 
for k_means in range(2, 11):
    key, bunch, separate_total, rating = k_means_start_2(
        k_means)
    bunch_2.append(key)
    rating_2.append(rating)
    plt.subplot(3, 3, k_means-1)
    plt.grid()
    for i in range(k_means):
        x = resource[np.where(bunch == i)[0]]
        plt.scatter(x[:, 0], x[:, 1], c = colors[i])
    # for i in range(k_means):
    #     plt.plot(separate_total, c = colors[i])
plt.show()

plt.plot(range(2, 11), rating_2, c = colors[6])
plt.xlabel('Number of Clusters')
plt.ylabel('Objective Score')
plt.title('Strategy 2: 2nd initialization objective function vs number of clusters')
plt.show()

for i in range(9):                                                                             # Cluster Locations for STRAT 2
    plt.subplot(3, 3, i+1)
    plt.grid()
    plt.scatter(bunch_1[i][:, 0], bunch_1[i][:, 1])
    plt.grid()
    plt.scatter(bunch_2[i][:, 0], bunch_2[i][:, 1])
    plt.grid()
plt.show()