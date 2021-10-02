Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@anandkuwadia 
Harika-Pothina
/
Anamoly-Detection
Public
0
00
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Anamoly-Detection/Torque and Force Analysis.py /
@Harika-Pothina
Harika-Pothina Add files via upload
Latest commit 405892f on Nov 29, 2018
 History
 1 contributor
126 lines (100 sloc)  4.28 KB
   
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from numpy import random
import csv
import io
# from pandas import df


def read_dataset(file_path, delimiter=','):
    # return genfromtxt(file_path, delimiter=delimiter, dtype=int)
    with io.open(file_path, 'r', encoding='utf-8-sig') as csvFile:  # encoding removes unnecessary characters
        read = csv.reader(csvFile, delimiter=delimiter, dialect='excel')
        # read is just an iterator and should be converted to a list to have length
        return np.array(list(read)).astype(np.float)  # an array of list is returned; np.float to remove flexible error


def read_gt(file_path, delimiter=','):
    # return genfromtxt(file_path, delimiter=delimiter, dtype=int)
    with io.open(file_path, 'r', encoding='utf-8-sig') as csvFile:  # encoding removes unnecessary characters
        read = csv.reader(csvFile, delimiter=delimiter, dialect='excel')
        # read is just an iterator and should be converted to a list to have length
        return np.array(list(read))  # an array of list is returned; np.float to remove flexible error


def estimate_gaussian(dataset):
    mu = np.mean(dataset, axis=0)
    cov = np.cov(dataset.T)
    return np.ravel(mu), cov  # ravel flattens the 1-D array
    # x = np.array([[1, 2, 3], [4, 5, 6]])
    # >>> print(np.ravel(x))
    # [1 2 3 4 5 6]


tr_data = read_dataset('tr-sample.csv')
gt_data = read_dataset('gt-sample.csv')

print('The training dataset:\n', tr_data)
print('The ground-truth dataset:\n', gt_data)

# random.shuffle(tr_data)  # shuffles the dataset randomly
print('\nTraining Dataset after random shuffling:\n', tr_data)

M = tr_data.shape[0]  # no. of rows
tr = tr_data[:int((M+1)*.60)]  # 0 to 60% i.e 60%
cv = tr_data[int(M*.60):int(M*.80)]  # 60 to 80 i.e 20%
test = tr_data[int(M*.80):]  # 80 to 100% i.e 20%

gt_d = gt_data[int(M*.60):int(M*.80)]

print('\nNo. of instances:\n', M)
print('\nNo. of instances in split training dataset:\n', tr.shape[0])
print('\nNo. of instances in split test dataset:\n', test.shape[0])
print('\nNo. of instances in split gt dataset:\n', gt_d.shape[0])


def select_threshold_by_cv(probs, gt):
    best_epsilon = 0
    best_f1 = 0
    stepsize = (max(probs) - min(probs)) / 10
    epsilons = np.arange(min(probs), max(probs), stepsize)
    print('\nepsilons = ', epsilons)
    # encoded_labels = df['label'].map(lambda x: 1 if x == 'anomaly' else 0).values
    for epsilon in np.nditer(epsilons):  # nditer iterates over an array
        print('\nepsilon val = ', epsilon)
        predictions = (probs < epsilon)
        print('\npredictions = ', predictions)
        f = f1_score(gt, predictions, average='binary')
        print('\n f = ', f)
        if f[0]/2 > best_f1:
            best_f1 = f[0]/2
            best_epsilon = epsilon

    return np.float(best_f1), np.float(best_epsilon)


# function calling starts here
mean, covariance = estimate_gaussian(tr_data)
print('\nmean=\n', mean)
print('\ncovariance=\n', covariance)

# to convert to PSD
min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
if min_eig < 0:
    covariance -= 10*min_eig * np.eye(*covariance.shape)

print('\ncovariance again = ', covariance)

p = multivariate_normal.pdf(tr_data, mean, covariance)
print('\np=\n', p)

# mean_, covariance_ = estimate_gaussian(cv)
# print('\nmean_=\n', mean_)
# print('\ncovariance_=\n', covariance_)

p_cv = multivariate_normal.pdf(cv, mean, covariance)
print('\np_cv=\n', p_cv)
fscore, ep = select_threshold_by_cv(p_cv, gt_d)
print(fscore, ep)
k = []
# ep = 0.0000903
x = epsilon
y = p_cv
plt.scatter(x, y, label= "stars", color= "green",
            marker= "*", s=30)
plt.plot(x, y)
plt.xlabel('epsilon')
plt.ylabel('p_cv')
plt.show()

for r in range(test.shape[0]):
    var = list(test[r, :])
    print('\n var = ', var)
    p_test = multivariate_normal.pdf(var, mean, covariance)
    print('\nptest', p_test)
    outliers = (p_test < ep)
    print('\noutliers before', outliers)
    if outliers is not True:
        print('\nThere is no anomaly')
        k.append(r)
    print('\noutliers = ', outliers)

print('The following are the instances with no anomaly', k)
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
Loading complete
