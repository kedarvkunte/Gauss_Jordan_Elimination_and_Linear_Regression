import numpy as np
import numpy.linalg as la
import timeit
import unittest
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import copy

############################################################
# Problem 1: Gauss-Jordan Elimination
############################################################

def gauss_jordan(A):
    ## Add code here ##
    A_original = copy.deepcopy(A)
    A = A.astype(float)
    #i_star_array = []
    B = []
    for k in range(A.shape[0]):
        for i in range(k, A.shape[0]):
            #i_star_array.append(np.argmax(np.abs(A[i][k]), axis=0))
            i_star = np.argmax(np.abs(A[i][k]), axis=0)

            if A[i_star][k] == 0:
                B.append(0)

            tmp = k
            A[k, :] = A[i_star, :]
            A[i_star, :] = A[tmp, :]

        for j in range(k + 1, A.shape[0]):

            f = A[j][k] / A[k][k]
            A[j, :] = A[j, :] - f * A[k, :]



    for k in range(A.shape[0] - 1, -1, -1):
        A[k, :] = A[k, :] / A[k, k]
        for j in range(k - 1, -1, -1):
            f = A[j][k] / A[k][k]
            A[j, :] = A[j, :] - f * A[k, :]

    if all(v == 0 for v in B):
        print("Matrix is not invertible")
        return A
    else:
        return A

Ar = np.array([[1, 2], [3, 4]],dtype=float)
print(gauss_jordan(Ar))





    #return -1

    
############################################################
# Problem 2: Ordinary Least Squares Linear Regression
############################################################

def linear_regression_inverse(X,y):
    ## Add code here ##
    #return -1
    Beta = (la.inv(X.transpose().dot(X))).dot(X.transpose().dot(y))
    return Beta



def linear_regression_moore_penrose(X,y):
    ## Add code here ##
    #return -1
    Beta = la.pinv(X).dot(y)
    return Beta

def generate_data(n,m):
    """
        Generates a synthetic data matrix X of size n by m 
        and a length n response vector.
    
        Input:
            n - Integer number of data cases.
            m - Integer number of independent variables.
    
        Output:
            X - n by m numpy array containing independent variable
                observasions.
            y - length n numpy array containg dependent variable
                observations.
    """
    X = np.random.randn(n,m)
    beta = np.random.randn(m)
    epsilon = np.random.randn(n)*0.5
    y = np.dot(X,beta) + epsilon
    
    return X,y
    
def time_linear_regression(method,n,m,n_runs):
    """
        Times a linear regression method on synthetic data of size n by m.
        Tests the function n_runs times and takes the minimum runtime.
    
        Usage:
        #>>> time_linear_regression('linear_regression_inverse',100,10,100)
    
        Input:
            method  - String specifying the method to be used. Should be 
                      either 'linear_regression_inverse' or
                      'linear_regression_moore_penrose'.
            n       - Integer number of data cases.
            m       - Integer number of independent variables.
            n_runs  - Integer specifying the number of times to test the method.
        
        Ouput:
            run_time - Float specifying the number of seconds taken by the 
                       shortest of the n_runs trials.
    """
    setup_code = "import numpy as np; from __main__ import generate_data, %s; X,y = generate_data(%d,%d)"%(method,n,m)
    test_code = "%s(X,y)"%method
    return timeit.timeit(test_code,number=n_runs,setup=setup_code)     

def problem2_plots():

    ## Add code here ##
    print("Part a")
    sizes = np.arange(25,250,10)
    runtimes_inverse = np.zeros(sizes.shape[0])
    runtimes_pseudo_inverse = np.zeros(sizes.shape[0])

    for i,m in enumerate(sizes):
        runtimes_inverse[i] = time_linear_regression('linear_regression_inverse',1000,m,100)
        runtimes_pseudo_inverse[i] = time_linear_regression('linear_regression_moore_penrose',1000,m,100)

    log_runtimes_inverse = np.log(runtimes_inverse)
    #coef

    plt.clf()
    plt.semilogy(sizes, runtimes_inverse, label="Runtimes_Inverse")
    plt.semilogy(sizes,runtimes_pseudo_inverse, label="Runtimes_Pseudo_Inverse")
    plt.ylabel("Algorithms' runtimes in seconds")
    plt.xlabel("m covariates")
    plt.legend(loc=0)
    plt.show()

    print("Part b")
    sizes = np.arange(1000, 10000, 50)
    runtimes_inverse = np.zeros(sizes.shape[0])
    runtimes_pseudo_inverse = np.zeros(sizes.shape[0])

    for i,n in enumerate(sizes):
        runtimes_inverse[i] = time_linear_regression('linear_regression_inverse',n,25,100)
        runtimes_pseudo_inverse[i] = time_linear_regression('linear_regression_moore_penrose',n,25,100)

    plt.clf()
    plt.semilogy(sizes, runtimes_inverse, label="Runtimes_Inverse")
    plt.semilogy(sizes, runtimes_pseudo_inverse, label="Runtimes_Pseudo_Inverse")
    plt.ylabel("Algorithms' runtimes in seconds")
    plt.xlabel("n data cases")
    plt.legend(loc=0)
    plt.show()

if __name__=="__main__":
    problem2_plots()