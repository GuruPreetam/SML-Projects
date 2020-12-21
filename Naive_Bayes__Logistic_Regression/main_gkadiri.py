import numpy as np
from scipy import io
import matplotlib.pyplot as plt

mnist_data = io.loadmat('mnist_data.mat')                                       # Importing Data

training_X = mnist_data['trX']                                                  # Training Set
training_Y = mnist_data['trY']                                                  # Training Labels

testing_X = mnist_data['tsX']                                                   # Testing Set
testing_Y = mnist_data['tsY']                                                   # Testing Label

print(mnist_data)

##  TASK 1  ###
# Extracting features - Mean, Standard Deviation
no_samples = len(training_X)                                                    # number of trainign samples
classes = 2                                                                     # number of classes
features = 2                                                                    # number of features
tot7 = 1028 
tot8 = 974

    # Training
mean_training_X = np.array(np.mean(training_X, axis=1)).reshape(-1, 1)
sd_training_x = np.array(np.std(training_X, axis=1)).reshape(-1, 1)
training_X = np.concatenate((mean_training_X, sd_training_x), axis=1)

    # Testing
mean_testing_X = np.array(np.mean(testing_X, axis=1)).reshape(-1, 1)
sd_testing_X = np.array(np.std(testing_X, axis=1)).reshape(-1, 1)
testing_X = np.concatenate((mean_testing_X, sd_testing_X), axis=1)

t_size = testing_X.shape[0]


# Training Features of class 7
index_7 = np.where(training_Y[0] == 0)
N7_training = len(index_7[0])                                                              # number of training samples for class 7
training_X7 = training_X[index_7]

# Training Features for class 8
index_8 = np.where(training_Y[0] == 1)
N8_training = len(index_8[0])                                                              # number of training samples for class 8
training_X8 = training_X[index_8]

# Testing features for class 7
index_7 = np.where(testing_Y[0] == 0)
N7_testing = len(index_7[0])                                                               # number of testing samples for class 7
testing_X7 = testing_X[index_7]

# test features for class 8
index_8 = np.where(testing_Y[0] == 1)
N8_testing = len(index_8[0])                                                               # number of testing samples for class 8
testing_X8 = testing_X[index_8]

print('\nNAIVE BAYES')
print('\n')

## MLE Density estimation

def MLE():
    # For 7 class
    probability_Y7 = N7_training/no_samples
    mean = np.mean(training_X7, axis=0)
    mu7 = np.array(mean).reshape(features, 1)
    difference_mu7_X = (training_X7) - (mu7.T)
    sig7 = ((difference_mu7_X.T).dot(difference_mu7_X)) / (no_samples-1)

    # For 8 class
    probability_Y8 = N8_training/no_samples
    mean = np.mean(training_X8, axis=0)
    mu8 = np.array(mean).reshape(features, 1)
    difference_mu8_X = (training_X8) - (mu8.T) 
    sig8 = ((difference_mu8_X.T).dot(difference_mu8_X)) / (no_samples-1)

    return probability_Y7, mu7, sig7, probability_Y8, mu8, sig8


probability_Y7, mu7, sig7, probability_Y8, mu8, sig8 = MLE()

print('FOR 7')
print('probability(y=7):- \n', probability_Y7)
print('Mean for class 7:- \n', mu7)
print('covaraiance matrix for class 7:- \n', sig7)
print('\n')
print('FOR 8')
print('probability(y=8):- \n', probability_Y8)
print('Mean for class 8:- \n', mu8)
print('Covariance matrix for class 8:- \n', sig8)

# Diagonal covariance matrix
sig7_diagonal = np.diag(np.diag(sig7))
sig8_diagonal = np.diag(np.diag(sig8))
print("Diagonal covariance matrix of 7:\n", sig7_diagonal)
print("Diagonal covariance matrix of 8:\n", sig8_diagonal)


###  TASK 2 ###
## Naive Bayes classification

def MultiVGuassian_log(A, mu, sig):
    return -(np.log(np.linalg.det(sig)**0.5)+((0.5)*((A-mu).T).dot(np.linalg.inv(sig)).dot(A-mu)))


def predicting_NB(A):
    probability_yx7 = np.log(probability_Y7) + MultiVGuassian_log(A, mu7, sig7_diagonal)
    probability_yx8 = np.log(probability_Y8) + MultiVGuassian_log(A, mu8, sig8_diagonal)
    return np.argmax([probability_yx7, probability_yx8])


naivebayes_predictedY = np.zeros((t_size))

for i in range(t_size):
    naivebayes_predictedY[i] = predicting_NB(testing_X[i].reshape(features, 1))


naive_bayes_acc = 0
naive_bayes_acc7 = 0
naive_bayes_acc8 = 0
for i in range(t_size):
    if naivebayes_predictedY[i] == testing_Y[0][i]:
        naive_bayes_acc += 1
    if naivebayes_predictedY[i] == 0 and naivebayes_predictedY[i] == testing_Y[0][i]:
        naive_bayes_acc7 += 1
    if naivebayes_predictedY[i] == 1 and naivebayes_predictedY[i] == testing_Y[0][i]:
        naive_bayes_acc8 += 1

###   TASK 3    ###
## Logistic Regression

def sigm(score):
    return 1/(1+np.exp(-score))


def log_reg(rate_of_learning, itr):                                                            # Initial weights
    weights = np.zeros((features+1, 1), dtype=np.float64)
    x = np.array([1]*no_samples).reshape(no_samples, 1)
    x = np.concatenate((x, training_X.reshape(no_samples, features)), axis=1).reshape(features+1, no_samples)
    y = training_Y.reshape(1, no_samples)
    for i in range(itr):
        score = np.dot(weights.T, x)
        predY = sigm(score)
        err = y-predY
        grad = np.dot(x, err.T)
        weights = weights + (rate_of_learning*grad)
    return weights


# Trained Weights
trained_weights = log_reg(0.00001, 10000)


# TESTING
x = np.array([1]*t_size).reshape(t_size, 1)
testing_X = np.concatenate((x, testing_X.reshape(t_size, features)), axis=1).reshape(features+1, t_size)
predicted_logReg = sigm(np.dot(trained_weights.T, testing_X))
predictions = []
for val in predicted_logReg[0]:
    if val <= 0.5:
        predictions.append(0)
    else:
        predictions.append(1)


# accuracy for Class 7
acc_logReg = 0
acc_logReg7 = 0
acc_logReg8 = 0
for i in range(t_size):
    if predictions[i] == testing_Y[0][i]:
        acc_logReg += 1
    if predictions[i] == 0 and predictions[i] == testing_Y[0][i]:
        acc_logReg7 += 1
    if predictions[i] == 1 and predictions[i] == testing_Y[0][i]:
        acc_logReg8 += 1

print('The trained weights are:',trained_weights)

##    TASK 4    ###
# END RESULTS
print('\nFINAL RESULT')

print('Naive Bayes - Test Sample Accuracy:', str(naive_bayes_acc*100/t_size))
print('Naive Bayes - Samples for class 7 accuracy:', str(naive_bayes_acc7*100/tot7))
print('Naive Bayes - Samples for class 8 accuracy:', str(naive_bayes_acc8*100/tot8))

print('Logistic Regression - Test Samples accuracy:', str((acc_logReg)*100/t_size))
print('Logistic Regression - Samples for class 7 accuracy:', str(acc_logReg7*100/tot7))
print('Logistic Regression - Samples for class 8 accuracy:', str(acc_logReg8*100/tot8))