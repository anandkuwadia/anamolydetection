# anamolydetection
Detect anomaly using Gaussian discriminant analysis for a multivariate distribution data set

This repo implements a gaussian process with Gaussian noise, making the maximum likelihood integral analytically solveable. The function is in infer.py.

It implements a typical gaussian process with a covariance kernel function :

The null hypothesis is that the data set is similar to the normal distribution, therefore a sufficiently small p-value indicates non-normal data. 

An example problem using the force and torque data is used.

Code has been tested with Python 2.7 only. Requires Smatplotlib.

