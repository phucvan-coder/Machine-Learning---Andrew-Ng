import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def estimateGaussian(X):
    # Useful variables
    m, n = X.shape
    
    #===========CODE================
    mu = (1 / m) * np.sum(X, axis=0)
    sigma2 = (1 / m) * np.sum(np.square(X - mu), axis=0)
    #===========CODE================
    
    return mu.reshape(n, 1), sigma2.reshape(n, 1)

def multivariateGaussian(X, mu, Sigma2):
    """
    Computes the probability density function of the examples X
    under the multivariate gaussian distribution with parameters
    mu and sigma2. If Sigma2 is a matrix, it is treated as the
    covariance matrix. If Sigma2 is a vector, it is treated as the
    sigma^2 values of the variances in each dimension (a diagonal
    covariance matrix).
    Args:
        X     : array(# of training examples m, # of features n)
        mu    : array(# of features n, 1)
        Sigma2: array(# of features n, # of features n)
    Returns:
        p     : array(# of training examples m,)
    """
    k = len(mu)
    
    if(Sigma2.shape[0] == 1) or (Sigma2.shape[1] == 1):
        Sigma2 = linalg.diagsvd(Sigma2.flatten(), len(Sigma2.flatten()), len(Sigma2.flatten()))
        
        X = X - mu.transpose()
        p = np.dot(np.power(2 * np.pi, -k / 2.0), np.power(np.linalg.det(Sigma2), -0.5)) * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(Sigma2)) * X, axis=1))
        
    return p.reshape(len(p), 1)

# Create a function to visualize the dataset and its estimated distribution.
def visualizeFit(X, mu, Sigma2):
    """
    Visualizes the dataset and its estimated distribution.
    This visualization shows the probability density function
    of the Gaussian distribution. Each example has a location
    (x1, x2) that depends on its feature values.
    Args:
        X     : array(# of training examples m, # of features n)
        mu    : array(# of features n, 1)
        sigma2: array(# of features n, 1)
    """
    X1, X2 = np.meshgrid(np.arange(0, 30, 0.5), np.arange(0, 30, 0.5))
    Z = multivariateGaussian(np.column_stack((X1.reshape(X1.size), X2.reshape(X2.size))), mu, Sigma2)
    
    Z = Z.reshape(X1.shape)
    
    plt.plot(X[:, 0], X[:, 1], 'bx', markersize=3)
    
    # Do not plot if there are infinities
    if (np.sum(np.isinf(Z)) == 0):
        plt.contour(X1, X2, Z, np.power(10, (np.arange(-20, 0.1, 3)).transpose()))
         
#Create a function to find the best threshold epsilon.
def selectThreshold(yval, pval):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (pval) and
    the ground truth (yval).
    Args:
        yval       : array(# of cv examples,)
        pval       : array(# of cv examples,)
    Returns:
        bestEpsilon: float
        bestF1     : float
    """
    # Init values
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    
    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        # Use predictions to get a binary vector of
        # 0's and 1's of the outlier predictions.
        predictions = (pval < epsilon).reshape(len(pval), 1)
        
        tp = np.sum((yval == 1) & (predictions == 1))
        fp = np.sum((yval == 0) & (predictions == 1))
        fn = np.sum((yval == 1) & (predictions == 0))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = (2 * precision * recall) / (precision + recall)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
            
    return bestEpsilon, bestF1
        
#Create a function to compute the cost J and grad
def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_coef):
    """
    Returns the cost and gradient for
    the collaborative filtering problem.
    Args:
        params      : array(num_movies x num_features + num_users x num_features,)
        Y           : array(num_movies, num_users)
        R           : array(num_movies, num_users)
        num_users   : int
        num_movies  : int
        num_features: int
        lambda_coef : float
    Returns:
        J           : float
        grad        : array(num_movies x num_features + num_users x num_features,)
    """
    # Unfold params back into the parameters X and Theta
    X = np.reshape(params[:num_movies * num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies * num_features:], (num_users, num_features))
    
    #Init values
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    
    # Compute squared error
    error = np.square(np.dot(X, Theta.T) - Y)
    
    # Compute regularization term
    reg_term = (lambda_coef / 2) * (np.sum(np.square(Theta)) + np.sum(np.square(X)))
    
    # Compute cost function but sum only if R(i, j)=1; vectorized solution
    J = (1 / 2) * np.sum(error * R) + reg_term
    
    # Compute the gradients
    X_grad = np.dot((np.dot(X, Theta.T) - Y) * R, Theta) + lambda_coef * X
    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X) + lambda_coef * Theta
    
    grad = np.concatenate((X_grad.reshape(X_grad.size), Theta_grad.reshape(Theta_grad.size)))
    
    return J, grad

# Create a function to compute numerical gradient
def computeNumericalGradient(J, Theta):
    """
    Computes the numerical gradient of the function J
    around theta using "finite differences" and gives
    a numerical estimate of the gradient.
    Notes: The following code implements numerical
           gradient checking, and returns the numerical
           gradient. It sets numgrad(i) to (a numerical 
           approximation of) the partial derivative of J
           with respect to the i-th input argument,
           evaluated at theta. (i.e., numgrad(i) should
           be the (approximately) the partial derivative
           of J with respect to Theta(i).)
    Args:
        J      : function
        Theta  : array(num_movies x num_features + num_users x num_features,)
    Returns:
        numgrad: array(num_movies x num_features + num_users x num_features,)
    """
    # Initialize parameters
    numgrad = np.zeros(Theta.shape)
    perturb = np.zeros(Theta.shape)
    e = 1e-4
    
    for p in range(Theta.size):
        # Set the perturbation vector
        perturb.reshape(perturb.size)[p] = e
        loss1, _ = J(Theta - perturb)
        loss2, _ = J(Theta + perturb)
        
        # Compute the Numerical Gradient
        numgrad.reshape(numgrad.size)[p] = (loss2 - loss1) / (2 * e)
        perturb.reshape(perturb.size)[p] = 0
        
    return numgrad

# Create a function to check the cost function and gradients
def checkCostFunction(lambda_coef):
    """
    Creates a collaborative filering problem 
    to check the cost function and gradients.
    It will output the analytical gradients
    and the numerical gradients computed using
    computeNumericalGradient. These two gradient 
    computations should result in very similar values.
    Args:
        lambda_coef : float
    """
    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)
    
    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y!=0] = 1
    
    # Run gradient checking
    X = np.random.rand(X_t.shape[0], X_t.shape[1])
    Theta = np.random.rand(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]
    
    #Create short hand for cost function
    costFunc = lambda p: cofiCostFunc(p, Y, R, num_users, num_movies, num_features, lambda_coef)
    
    params = np.concatenate((X.reshape(X.size), Theta.reshape(Theta.size)))
    numgrad = computeNumericalGradient(costFunc, params)
    J, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_coef)
    
    # Visually examine the two gradient computations
    for numerical, analytical in zip(numgrad, grad):
        print('Numerical Gradient: {0:10f}, Analytical Gradient {1:10f}'.format(numerical, analytical))
        
    print('\nThe above two columns should be very similar.\n')
    
    # Evaluate the norm of the difference between two solutions.
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    
    print('If the backpropagation implementation is correct, then \n' \
             'the relative difference will be small (less than 1e-9). \n' \
             '\nRelative Difference: {:.10E}'.format(diff))
    
#Create a function to load movies
def loadMovieList():
    """
    Reads the fixed movie list in movie_idx.txt
    and returns a cell array of the words in movieList.
    Returns:
        movieList: list
    """
    # Read the fixed movie list
    with open('movie_ids.txt', encoding="ISO-8859-1") as f:
        movieList = []
        for line in f:
            movieName = line.split()[1:]
            movieList.append("".join(movieName))
            
    return movieList

# Create a function to normalize ratings
def normalizeRatings(Y, R):
    """
    Preprocesses data by subtracting mean rating for every
    movie (every row). Normalizes Y so that each movie has
    a rating of 0 on average, and returns the mean rating in Ymean.
    Args:
        Y    : array(num_movies, num_users)
        R    : array(num_movies, num_users)
    Returns:
        Ynorm: array(num_movies, num_users)
        Ymean: array(num_movies, 1)
    """
    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = R[i, :] == 1
        # Compute the mean only of the rated movies
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
        
    return Ynorm, Ymean
    
    
    