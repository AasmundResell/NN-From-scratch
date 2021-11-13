import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ols import FrankesFunction, CreateXmat, MSE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)


class NeuralNetworkRegression:
    def __init__(
        self,
        X_data,
        y_data,
        layer_list,
        epochs,
        batch_size=100,
        eta=0.1,
        lambda_=0.0,
        hidden_type = "sigmoid",
        output_type = "linear",
    ):
        self.n_inputs, self.n_features = X_data.shape

        self.n_outputs = y_data.shape[1]
        self.eta = eta
        self.lambda_ = lambda_
        self.n_layers = len(layer_list)
        self.layers = layer_list
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_all_data = X_data
        self.y_all_data = y_data

        self.initilize_weight_and_bias()

        if hidden_type == "sigmoid":
            self.f_h = lambda x: 1 / (1 + np.exp(-x)) 
            self.df_h = lambda x: self.f_h(x)*(1-self.f_h(x))
        elif hidden_type == "relu":
            self.f_h = lambda x: np.maximum(0,x)
            self.df_h = lambda x: np.heaviside(x,1)
        if output_type == "softmax":
            self.f_o = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
            self.df_o = lambda x: self.f_o(x)*(1-self.f_o(x))
        elif output_type == "linear":
            self.f_o = lambda x: x
            self.df_o = lambda x: 1
        
    def initilize_weight_and_bias(self):
        self.weights = []
        self.bias = []
        for i in range(self.n_layers):
            if i == 0:
                self.weights.append(np.random.randn(self.n_features, self.layers[0]))
                self.bias.append(np.zeros(self.layers[0]) + 0.01)
            else:
                self.weights.append(np.random.randn(self.layers[i - 1], self.layers[i]))
                self.bias.append(np.zeros(self.layers[i]) + 0.01)

        # Adding output weights and output bias
        self.weights.append(np.random.randn(self.layers[-1], self.n_outputs))
        self.bias.append(np.zeros(self.n_outputs) + 0.01)


    def FeedForward(self):

        a_l = self.X_ex
        self.a = []
        self.z = []
        self.a.append(a_l)
        for i in range(self.n_layers):
            z_l = np.matmul(a_l, self.weights[i]) + self.bias[i]
            a_l = self.f_h(z_l)
            self.a.append(a_l)  # Stored for the backpropagation
            self.z.append(z_l)
        z_l = np.matmul(a_l, self.weights[-1]) + self.bias[-1]
        self.output = self.f_o(z_l)
        self.z.append(z_l)
        
    def FeedForwardPred(self, X):

        a_l = X

        for i in range(self.n_layers):

            z_l = np.matmul(a_l, self.weights[i]) + self.bias[i]

            a_l = self.f_h(z_l)

        z_o = np.matmul(a_l, self.weights[-1]) + self.bias[-1]
        return self.f_o(z_o)

    def Predict(self, X):
        p = self.FeedForwardPred(X)
        return p

    def BackPropagation(self):

        errors = []
        self.weight_gradients = []
        self.bias_gradients = []

        errors.append((self.output - self.y_ex)*self.df_o(self.z[-1]))

        self.weight_gradients.append(np.matmul(self.a[-1].T, errors[0]))
        self.bias_gradients.append(np.sum(errors[0], axis=0))

        # backward for-loop calculating errors
        for i in range(self.n_layers - 1, -1, -1):
            error_hidden = (
                np.matmul(errors[0], self.weights[i + 1].T)
                *self.df_h(self.z[i])
                )
            # *self.a[i+1]*(1-self.a[i+1])
            errors.insert(0, error_hidden)
            self.weight_gradients.insert(0, np.matmul(self.a[i].T, errors[0]))
            self.bias_gradients.insert(0, np.sum(errors[0], axis=0))

        # Updating weights
        for l in range(len(self.weights)):
            if self.lambda_ > 0.0:
                self.weight_gradients[l] += self.lambda_ * self.weights[l]

            self.weights[l] -= self.eta * self.weight_gradients[l]
            self.bias[l] -= self.eta * self.bias_gradients[l]

    def Train(self):

        indices = np.arange(self.n_inputs)
        batches = self.n_inputs // self.batch_size

        for e in range(self.epochs):
            for m in range(batches):
                datapoints = np.random.choice(
                    indices, size=self.batch_size, replace=False
                )

                self.X_ex = self.X_all_data[datapoints]
                self.y_ex = self.y_all_data[datapoints]

                self.FeedForward()
                self.BackPropagation()


def FrankesFunctionRegression():
    n = 10  # Highest order polynomial in the variables
    N = 200 # Number of points sampled along the x and y axis

    # Generate random input coordinates for the Franke's Function
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    X = CreateXmat(x, y, n)
    z = FrankesFunction(x, y) + 0.2*np.random.rand(x.shape[0])
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    #X = X - np.mean(X)
    z = z[:, np.newaxis]
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    

    layer_list = [100, 100]

    eta = 0.005
    lmbd = 0.1
    epochs = 100
    batch_size = 10

    nn = NeuralNetworkRegression(
        X_train,
        z_train,
        layer_list=layer_list,
        epochs=epochs,
        batch_size=batch_size,
        eta=eta,
        lambda_=lmbd,
    )
    nn.Train()

    beta_linalg = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
    

    z_predict = nn.Predict(X_test)
    z_pred_linalg = X_test @ beta_linalg

    print("Test MSE linalg:", MSE(z_test, z_pred_linalg))
    print("Test MSE Neural network:", MSE(z_test, z_predict))
    

    z_plot_nn = nn.Predict(X) 
    
    
    # Plot the surface.
    fig = plt.figure(1)
    plt.rcParams['figure.figsize'] = (12,12)

    ax = fig.gca(projection='3d')

    
    trisurf1 = ax.plot_trisurf(x, y, z_plot_nn[:,0], cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(trisurf1, shrink=0.5, aspect=5)

    plt.show()
    sns.set()

    eta_vals = np.linspace(0.001, 0.01, 6)
    lmbd_vals = np.logspace(-5, 1, 7)
    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    
    # Grid search
    for i, eta in enumerate(eta_vals):
        for j, lambda_ in enumerate(lmbd_vals):
            dnn = NeuralNetworkRegression(
                X_train,
                z_train,
                layer_list=layer_list,
                epochs=epochs,
                batch_size=batch_size,
                eta=eta,
                lambda_=lambda_,
            )
            dnn.Train()
            DNN_numpy[i][j] = dnn

            train_pred = dnn.Predict(X_train)
            test_pred = dnn.Predict(X_test)
            
            train_accuracy[i][j] = MSE(z_train, train_pred)
            test_accuracy[i][j] = MSE(z_test, test_pred)
            print("Accuracy score on test set: ", test_accuracy[i][j])

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

def OneDimensionalRegression():
    # Regression analysis on a function with a single dimension
    N = 100
    x = np.random.rand(100)
    y = 2.0+5*x*x+0.1*np.random.randn(100)

     
    y = y[:, np.newaxis]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    X_train = np.c_[x_train, x_train**2]
    X_test = np.c_[x_test, x_test**2]
    
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    layer_list = [50]

    eta = 0.01
    lmbd = 0.1
    epochs = 100
    batch_size = 5

    nn = NeuralNetworkRegression(
        X_train_scaled,
        y_train,
        layer_list=layer_list,
        epochs=epochs,
        batch_size=batch_size,
        eta=eta,
        lambda_=lmbd,
    )
    nn.Train()
    Y_predict = nn.Predict(X_test_scaled)
    
    print("Accuracy score on test set: ", MSE(y_test, Y_predict))

    
    plt.scatter(x_test, Y_predict)
    
    plt.scatter(x_test, y_test)
    plt.show()

    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    sns.set()
    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    # Grid search
    for i, eta in enumerate(eta_vals):
        for j, lambda_ in enumerate(lmbd_vals):
            nn = NeuralNetworkRegression(
                X_train_scaled,
                y_train,
                layer_list=layer_list,
                epochs=epochs,
                batch_size=batch_size,
                eta=eta,
                lambda_=lambda_,
            )
            nn.Train()
            DNN_numpy[i][j] = nn

            train_pred = nn.Predict(X_train)
            test_pred = nn.Predict(X_test)
            
            train_accuracy[i][j] = MSE(y_train, train_pred)
            test_accuracy[i][j] = MSE(y_test, test_pred)
            print("Accuracy score on test set: ", test_accuracy[i][j])

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
    
if __name__ == "__main__":
    FrankesFunctionRegression()
