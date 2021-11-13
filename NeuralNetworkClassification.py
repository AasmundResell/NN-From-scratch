import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

class NeuralNetworkClassification:
    def __init__(
        self,
        X_data,
        y_data,
        outputs,
        layer_list,
        epochs,
        batch_size=100,
        eta=0.1,
        lambda_=0.0,
        hidden_type = "sigmoid",
        output_type = "softmax",
    ):
        self.n_inputs, self.n_features = X_data.shape

        self.n_outputs = outputs
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


        self.f = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid
        self.softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

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
        return np.argmax(p, axis=1)

    def BackPropagation(self):

        errors = []
        self.weight_gradients = []
        self.bias_gradients = []

        errors.append(self.output - self.y_ex)

        self.weight_gradients.append(np.matmul(self.a[-1].T, errors[0]))
        self.bias_gradients.append(np.sum(errors[0], axis=0))

        # backward for-loop calculating errors
        for i in range(self.n_layers - 1, -1, -1):
            error_hidden = (
                np.matmul(errors[0], self.weights[i + 1].T)
                * self.df_h(self.z[i])
            )
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
        print(self.weights)

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector


def accuracy_score(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


def ClassifyBreastCancer():
    
    cancer = datasets.load_breast_cancer()

    x = cancer.data
    y = cancer.target
    labels = cancer.feature_names[0:30]

    print(
        "The content of the breast cancer dataset is:"
    )  # Print information about the datasets
    print(labels)
    print("-------------------------")
    print("inputs =  " + str(x.shape))
    print("outputs =  " + str(y.shape))
    print("labels =  " + str(labels.shape))

    # Generate training and testing datasets

    # Select features relevant to classification (texture,perimeter,compactness and symmetery)
    # and add to input matrix

    temp1 = np.reshape(x[:, 1], (len(x[:, 1]), 1))
    temp2 = np.reshape(x[:, 2], (len(x[:, 2]), 1))
    X = np.hstack((temp1, temp2))
    temp = np.reshape(x[:, 5], (len(x[:, 5]), 1))
    X = np.hstack((X, temp))
    temp = np.reshape(x[:, 8], (len(x[:, 8]), 1))
    X = np.hstack((X, temp))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_oh = to_categorical(y_train)
    y_test_oh = to_categorical(y_test)

    del temp1, temp2, temp

    eta_vals = np.logspace(-3, -1, 3)
    lamda_vals = np.logspace(-5, 1, 7)

    layer_list = [10,10]
    epochs = 100
    batch_size = 100
    sns.set()
    test_accuracy = np.zeros((len(eta_vals), len(lamda_vals)))
 
    for i, eta in enumerate(eta_vals):
        for j, lamda in enumerate(lamda_vals):
            nn = NeuralNetworkClassification(
                X_train_scaled,
                y_train_oh,
                outputs=2,
                layer_list=layer_list,
                epochs=epochs,
                batch_size=batch_size,
                eta=eta,
                lambda_=lamda,
                hidden_type= "sigmoid",
            )
            nn.Train()

            test_pred = nn.Predict(X_test_scaled)
            test_accuracy[i][j] = accuracy_score(y_test, test_pred)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

def ClassifyWrittenDigits():
    layer_list = [50]

    
    epochs = 100
    batch_size = 100
    n_categories = 10
    # download MNIST dataset
    digits = datasets.load_digits()

    # define inputs and labels
    inputs = digits.images
    labels = digits.target
    inputs = inputs.reshape(len(inputs), -1)
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(
        inputs, labels, train_size=train_size, test_size=test_size
    )
    Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(
        Y_test
    )

    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    sns.set()
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    # Grid search
    for i, eta in enumerate(eta_vals):
        for j, lambda_ in enumerate(lmbd_vals):
            nn = NeuralNetworkClassification(
                X_train,
                Y_train_onehot,
                outputs=n_categories,
                layer_list=layer_list,
                epochs=epochs,
                batch_size=batch_size,
                eta=eta,
                lambda_=lambda_,
            )
            nn.Train()
            DNN_numpy[i][j] = nn

            train_pred = nn.Predict(X_train)
            train_accuracy[i][j] = accuracy_score(Y_train, train_pred)

            test_pred = nn.Predict(X_test)
            test_accuracy[i][j] = accuracy_score(Y_test, test_pred)

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
    ClassifyBreastCancer()
