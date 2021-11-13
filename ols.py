import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

np.random.seed(2021)


def FrankesFunction(x, y):
    return (
        3 / 4 * np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
        + 3 / 4 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
        + 1 / 2 * np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
        - 1 / 5 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    )


def CreateXmat(
    x, y, n
):  # returns a matrix for the variables with n being the highest order term of the variables.
    L = len(y)
    X = np.ones((L, int((n + 1) * (n + 2) / 2)))
    for i in range(n + 1):
        t = int((i) * (i + 1) / 2)
        for j in range(i + 1):
            X[:,t + j] = x ** (i - j) * y ** j

    return X


def MSE(z, z_pred):
    return np.mean((z - z_pred) ** 2)


def R2(z, z_pred):
    return 1 - np.sum((z - z_pred) ** 2) / np.sum((z - np.mean(z)) ** 2)

if __name__ == "__main__":
    n =3  # Highest order polynomial in the variables
    N = 1000

    # Generate random input coordinates for the Franke's Function
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    X = CreateXmat(x, y, n)
    
    z = FrankesFunction(x, y) + np.random.randn(x.shape[0])
    print(z.shape)
    z = z[:,np.newaxis]
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    
    print("X_train: ",X_train.shape)
    print("z_train: ",z_train.shape)

    beta = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
    print("beta: ",beta.shape)

    z_t = X_train @ beta

    print("Training R2 (before scaling):", R2(z_train, z_t))
    print("Training MSE (before scaling):", MSE(z_train, z_t))
    
    z_pred = X_test @ beta
    print(z_pred.shape)
    print("Test R2 (before scaling):", R2(z_test, z_pred))
    print("Test MSE (before scaling):", MSE(z_test, z_pred))

    """
    # Scaling the data using the standard scaling (without standard deviation)
    
    X_train_scaled = X_train - np.mean(X_train)
    X_test_scaled = X_test - np.mean(X_test)
    
    beta = np.linalg.pinv(X_train_scaled.T.dot(X_train_scaled)).dot(X_train_scaled.T).dot(z_train)
    
    z_t = X_train_scaled @ beta

    print("Training R2 (after scaling):", R2(z_train, z_t))
    print("Training MSE (after scaling):", MSE(z_train, z_t))
    
    z_pred = X_test_scaled @ beta

    print("Test R2 (after scaling):", R2(z_test, z_pred))
    print("Test MSE (after scaling):", MSE(z_test, z_pred))
    """
    Z = X @ beta
    print(Z.shape)
    X = np.sort(x)
    Y = np.sort(y)
    print(X.shape)
    X, Y = np.meshgrid(X,Y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z[:,0], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
