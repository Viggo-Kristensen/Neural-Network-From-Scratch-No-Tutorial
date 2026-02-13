import numpy as np
from tensorflow.keras.datasets import mnist

# Layers
class NeuralNetwork():
    def __init__(self, y):
        # variables
        self.layers = 5
        self.input_s = 784
        self.shape = (784 , 5, 5, 5, 2)
        self.lr = 0.001
        self.epochs = 100
        self.n= len(y)

        # parameters
        self.w1 = np.random.uniform(low=-1, high=1, size=(5, self.input_s))
        self.b1 = np.random.uniform(low=-1, high=1, size=(5,1))
        self.w2 = np.random.uniform(low=-1, high=1, size=(5,5))
        self.b2 = np.random.uniform(low=-1, high=1, size=(5,1))
        self.w3 = np.random.uniform(low=-1, high=1, size=(5,5))
        self.b3 = np.random.uniform(low=-1, high=1, size=(5,1))
        self.w4 = np.random.uniform(low=-1, high=1, size=(2,5))
        self.b4 = np.random.uniform(low=-1, high=1, size=(2,1))

        self.w_list = []
        self.w_list.append(self.w1)
        self.w_list.append(self.w2)
        self.w_list.append(self.w3)
        self.w_list.append(self.w4)

        self.b_list = []
        self.b_list.append(self.b1)
        self.b_list.append(self.b2)
        self.b_list.append(self.b3)
        self.b_list.append(self.b4)

        # gradients init
        self.w1_g = np.zeros((5, self.input_s))
        self.b1_g = np.zeros((5, 1))
        self.w2_g = np.zeros((5, 5))
        self.b2_g = np.zeros((5, 1))
        self.w3_g = np.zeros((5, 5))
        self.b3_g = np.zeros((5, 1))
        self.w4_g = np.zeros((2, 5))
        self.b4_g = np.zeros((2, 1))

        self.weight_grad = [self.w1_g, self.w2_g, self.w3_g, self.w4_g]
        self.bias_grad = [self.b1_g, self.b2_g, self.b3_g, self.b4_g]

        # gradient intermediate values init
        self.z1 = np.array([])
        self.z2 = np.array([])
        self.z3 = np.array([])
        self.z4 = np.array([])
        self.a0 = np.array([])
        self.a1 = np.array([])
        self.a2 = np.array([])
        self.a3 = np.array([])
        self.a4 = np.array([])

        self.z_list = []
        self.a_list = []

    def error_function(self):
        pass

    def _sigmoid(self, x):
        x = 1 / (1 + (np.e ** - x))
        return x

    def forward_pass(self, x):
        x = self.w1 @ x + self.b1
        x = self._sigmoid(x)
        x = self.w2 @ x + self.b2
        x = self._sigmoid(x)
        x = self.w3 @ x + self.b3
        x = self._sigmoid(x)
        x = self.w4 @ x + self.b4
        x = self._sigmoid(x)
        return x

    def training_forward_pass(self, x):
        self.z_list = []
        self.a_list = []
        self.a0 = x
        x = self.w1 @ x + self.b1
        self.z1 = x
        x = self._sigmoid(x)
        self.a1 = x
        x = self.w2 @ x + self.b2
        self.z2 = x
        x = self._sigmoid(x)
        self.a2 = x
        x = self.w3 @ x + self.b3
        self.z3 = x
        x = self._sigmoid(x)
        self.a3 = x
        x = self.w4 @ x + self.b4
        self.z4 = x
        x = self._sigmoid(x)
        self.a4 = x

        self.a_list.append(self.a0)
        self.a_list.append(self.a1)
        self.a_list.append(self.a2)
        self.a_list.append(self.a3)
        self.a_list.append(self.a4)

        self.z_list.append(self.z1)
        self.z_list.append(self.z2)
        self.z_list.append(self.z3)
        self.z_list.append(self.z4)
        return x

    def backpropagation(self, x_sample, y_sample):
        self.w1_g = np.zeros((5, self.input_s))
        self.b1_g = np.zeros((5, 1))
        self.w2_g = np.zeros((5, 5))
        self.b2_g = np.zeros((5, 1))
        self.w3_g = np.zeros((5, 5))
        self.b3_g = np.zeros((5, 1))
        self.w4_g = np.zeros((2, 5))
        self.b4_g = np.zeros((2, 1))

        self.weight_grad = [self.w1_g, self.w2_g, self.w3_g, self.w4_g]
        self.bias_grad = [self.b1_g, self.b2_g, self.b3_g, self.b4_g]

        y_pred = self.training_forward_pass(x_sample)

        # multiply start_grad by 2 / n if switched away from SGD
        start_grad = y_pred - y_sample
        self.recursive_gradient(self.layers, start_grad)

    def recursive_gradient(self, cur_layer, cur_grad):
        cur_grad = cur_grad * (self.a_list[cur_layer - 1] * (1 - self.a_list[cur_layer - 1]))

        if cur_layer == 5:
            self.bias_grad[cur_layer - 2] += cur_grad

        else:
            self.bias_grad[cur_layer - 2] += cur_grad


        self.weight_grad[cur_layer - 2] += (cur_grad @ self.a_list[cur_layer - 2].T)
        cur_grad = self.w_list[cur_layer - 2].T @ cur_grad

        cur_layer -= 1
        if cur_layer == 1:
            return
        self.recursive_gradient(cur_layer, cur_grad)

    def update_parameters(self):
        for i in range(4):
            self.w_list[i] -= self.weight_grad[i] * self.lr
            self.b_list[i] -= self.bias_grad[i] * self.lr


(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_filter = np.where((y_train == 0) | (y_train == 1))
test_filter  = np.where((y_test == 0) | (y_test == 1))

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test   = x_test[test_filter], y_test[test_filter]

x_train = x_train.reshape(-1, 28*28, 1) / 255.0
x_test  = x_test.reshape(-1, 28*28, 1) / 255.0

y_train_oh = np.zeros((y_train.shape[0], 2, 1))
y_train_oh[y_train == 0, 0, 0] = 1
y_train_oh[y_train == 1, 1, 0] = 1

y_test_oh = np.zeros((y_test.shape[0], 2, 1))
y_test_oh[y_test == 0, 0, 0] = 1
y_test_oh[y_test == 1, 1, 0] = 1

print(x_train.shape, y_train_oh.shape)  # e.g., (12000, 784,1), (12000,2,1)


# Training loop
neural_network = NeuralNetwork(y_train)

for i in range(neural_network.epochs):
    print(f"epoch: {i+1}")
    for j in range(neural_network.n):
        x_sample = x_train[j].flatten().reshape(-1, 1)
        y_sample = y_train_oh[j].flatten().reshape(-1, 1)
        neural_network.backpropagation(x_sample, y_sample)
        neural_network.update_parameters()

# Testing loop
correct = 0
total = x_test.shape[0]

for i in range(total):
    x_sample = x_test[i].reshape(-1,1)
    y_true   = y_test_oh[i]

    y_pred = neural_network.forward_pass(x_sample)

    pred_class = np.argmax(y_pred)
    true_class = np.argmax(y_true)

    if pred_class == true_class:
        correct += 1

accuracy = correct / total
print(f"Test Accuracy: {accuracy*100:.2f}%")
