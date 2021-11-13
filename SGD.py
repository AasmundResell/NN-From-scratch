from ols import FrankesFunction, CreateXmat, MSE
import numpy as np
from random import random, seed
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

np.random.seed(2021)


def GradientDescent(X, z):

    # Creating Hessian matrix
    H = (2.0 / X.shape[0]) * X.T @ X

    EigVal, EigVec = np.linalg.eig(H)

    beta = np.random.randn(X.shape[1], z.shape[1])

    eta = 1.0 / np.max(EigVal)

    iterations = 1000
    for n in range(iterations):
        grad = (2.0 / X.shape[0]) * X.T @ (X @ beta - z)
        beta -= eta * grad

    return beta


def StochasticGradientDescent(X, z, M=5, epochs=4000, method = "OLS", lambda_ = 0.0,eta_init= 0.1):
    """StochasticGradientDescent: Solves a cost function using stochastic
            gradient decent, and returns the trained set of variables

    Args:
        param1 numpy array: The features
        param2 numpy array: The corresponding labels
        param3 (optional): The size of each minibatch, set to five by default
        param4 (optional): The number of epocs, set to ten by default
        param4 (optional): The regression method used for fitting, OLS by default

    Returns:
        numpy array: The trained beta array
    """

    n = z.shape[0]

    m = int(n / M)  
    print("Number of minibatches = ", m)

    theta = np.random.randn(X.shape[1], 1)

    t0 = 1.0
    t1 = t0/eta_init
    eta = t0/t1
    gamma = 0.5
    data_indices = np.arange(X.shape[0])
    dtheta_prev = 0.0 #Momentum based GD
    for e in range(1, epochs + 1):
        for i in range(m):
            datapoints = np.random.choice(data_indices, size=M, replace=False)
            X_k = X[datapoints]
            z_k = z[datapoints]
            if method == "OLS":
                grad = (2.0 / X_k.shape[0]) * X_k.T @ (X_k @ theta - z_k)
            elif method == "Ridge":
                grad = (2.0 / X_k.shape[0]) * X_k.T @ (X_k @ theta - z_k) + 2*lambda_*theta
            
            # Calculating the decaying step-length
            t = (e * m + i) / (m * 10e3)
            eta = t0 / (t + t1)
            dtheta =-eta * grad + gamma*dtheta_prev
            theta += dtheta
            dtheta_prev = dtheta
    return theta


if __name__ == "__main__":
    n = 10  # Highest order polynomial in the variables
    N = 300 # Number of points sampled along the x and y axis

    # Generate random input coordinates for the Franke's Function
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    X = CreateXmat(x, y, n)
    z = FrankesFunction(x, y) + 0.2*np.random.randn(x.shape[0])

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    #X = X - np.mean(X)
    z = z[:, np.newaxis]
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    beta_linalg = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
    beta_sgd_ols = StochasticGradientDescent(X_train, z_train,method="OLS")
    beta_gd_ols = GradientDescent(X_train, z_train)
    z_pred_linalg = X_test @ beta_linalg
    z_pred_gd_ols = X_test @ beta_gd_ols
    z_pred_sgd_ols = X_test @ beta_sgd_ols

    print("Test MSE linalg OLS:", MSE(z_test, z_pred_linalg))
    print("Test MSE Gradient descent OLS:", MSE(z_test, z_pred_gd_ols))
    print("Test MSE Stochastic gradient descent OLS:", MSE(z_test, z_pred_sgd_ols))


    eta_init_vals = np.logspace(-7,-2,4)
    lambdas = np.logspace(-8, -2, 6)
    
    # store the models for later use
    train_accuracy = np.zeros((len(eta_init_vals), len(lambdas)))
    test_accuracy = np.zeros((len(eta_init_vals), len(lambdas)))
    
    # Grid search Ridge regression and SGD
    for i, eta in enumerate(eta_init_vals):
        for j, lambda_ in enumerate(lambdas):
            beta_sgd_ridge = StochasticGradientDescent(X_train, z_train,method="Ridge",eta_init=eta,lambda_=lambda_)
            train_pred = X_train @ beta_sgd_ridge
            test_pred = X_test @ beta_sgd_ridge
            train_accuracy[i][j] = MSE(z_train, train_pred)
            test_accuracy[i][j] = MSE(z_test, test_pred)
            print("Accuracy score on test set: ", test_accuracy[i][j])
            print("Accuracy score on training set: ", train_accuracy[i][j])

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()
    
    z_plot_linalg = X @ beta_linalg
    z_plot_sgd = X @ beta_sgd
    
    
    # Plot the surface.
    fig = plt.figure(1)
    plt.rcParams['figure.figsize'] = (12,12)

    ax = fig.gca(projection='3d')

    
    trisurf1 = ax.plot_trisurf(x, y, z_plot_sgd[:,0], cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(trisurf1, shrink=0.5, aspect=5)

    plt.show()

    fig = plt.figure(2)

    ax = fig.gca(projection='3d')

    
    trisurf2 = ax.plot_trisurf(x, y, z_plot_linalg[:,0], cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(trisurf2, shrink=0.5, aspect=5)

    plt.show()

    fig = plt.figure(3)
    x = np.sort(x)
    y = np.sort(y)
    l1 = len(x)
    l2 = len(y)
    x,y = np.meshgrid(x,y)
    z = FrankesFunction(x,y) + 0.1*np.random.randn(l1,l2)

    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
