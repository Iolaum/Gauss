from __future__ import division
import numpy
import pickle


# steps is used to define the convengence step
# alpha is the rate of approaching minimum error
# beta is used to avoid overfitting. give approximation of R


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        print(" Step: " + str(step))
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][
                    j] != numpy.nan:  # find the error between R[i][j] and pridiction nR[i][j], only exist value give changing to P and Q
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in xrange(K):  # generate new P and Q
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P, Q)
        e = 0

        # find the error between R[i][j] and nR[i][j]
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] != numpy.nan:  # Only calculate the error of exist value
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in xrange(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        # if the error is smaller than a value
        if e < 2:
            print("Converged after {} steps".format(step))
            break
    return P, Q.T


R = [
    [5, 0, numpy.nan, 1, 0, 0, 6],
    [2, 0, 0, 1, 0, numpy.nan, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [1, 0, 5, 4, 0, 7, 0],
    [0, 1, 0, numpy.nan, 3, 0, 0],
]

# with open("../../dataset/03_transformed_tr_dataframe.p", 'rb') as handle:
#     R = pickle.load(handle)
#
#
# R = R[:1000, :]

R = numpy.array(R)
print(R)
print(R.shape)

# define the size of matrix
N = len(R)
M = len(R[0])

# give the size of matrix P and Q, K could be 2, 3 ,..., 20, it is determined by test
K = 2

P = numpy.random.rand(N, K)
Q = numpy.random.rand(M, K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)
