import numpy as np
import random
import utils

NUM_CLASSES = 2
np.random.seed(42)

def activation_sigmoid(x, backward=False):
    if backward == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

class Layer:
    def __init__(self, n_from, n_to) -> None:
        self.weights = np.random.randn(n_from, n_to)
        self.biases = np.random.randn(n_to)

    def layer_forward(self, x):
        return np.dot(x, self.weights) + self.biases
    
class Network:
    def __init__(self, network_shape) -> None:
        self.network_shape = network_shape
        self.layers = [Layer(ch_i, ch_o) for ch_i, ch_o in zip(network_shape[:-1], network_shape[1:])]

    def forward(self, x):
        outputs = [x]
        for i in range(len(self.layers)):
            output = self.layers[i].layer_forward(outputs[i])
            output = activation_sigmoid(output, backward=False)
            outputs.append(output)
        return outputs
    
    def backprop(self, x, y):
        x = np.array(x)
        x = x[np.newaxis, :]
        y = utils.label_to_one_hot(y, num_classes=NUM_CLASSES)

        nabla_w = [np.zeros((ch_i, ch_o)) for ch_i, ch_o in zip(self.network_shape[:-1], self.network_shape[1:])]
        nabla_b = [np.zeros((ch_o)) for ch_o in self.network_shape[1:]]

        outputs = self.forward(x)
        loss = utils.mse_loss(outputs[-1], y)
        # 1. compute gradient of output layer
        delta = (outputs[-1] - y) * activation_sigmoid(outputs[-1], backward=True)
        nabla_w[-1] = np.dot(outputs[-2].T, delta)
        nabla_b[-1] = delta

        # 2. compute gradient of other layers
        for i in range(2, len(self.layers) + 1):
            l = -i
            output_l = outputs[l]
            delta = np.dot(delta, self.layers[l+1].weights.T) * activation_sigmoid(output_l, backward=True)
            nabla_w[l] = np.dot(outputs[l-1].T, delta)
            nabla_b[l] = delta

        return nabla_w, nabla_b, loss
    
    def training(self, train_data, epochs, batch_size, lr, test_data):

        if test_data:
            n_test = len(test_data)
        n_train = len(train_data)

        for epoch in range(epochs):
            random.shuffle(train_data)
            mini_batchs = [train_data[k: k+batch_size] for k in range(0, n_train, batch_size)]

            for mini_batch in mini_batchs:
                loss = self.update_mini_batch(mini_batch, lr)

            if test_data:
                print(f"epoch: {epoch + 1}, accuracy: {self.evaluate(test_data)} / {n_test}, loss: {loss}.")
                if epoch == epochs - 1:
                    return self.get_outputs(test_data)
            else:
                print(f"epoch: {epoch + 1} finished.")

    def update_mini_batch(self, batch_data, lr):

        nabla_w = [np.zeros((ch_i, ch_o)) for ch_i, ch_o in zip(self.network_shape[:-1], self.network_shape[1:])]
        nabla_b = [np.zeros((ch_o)) for ch_o in self.network_shape[1:]]
        loss = 0
        for x, y in batch_data:
            nabla_w_, nabla_b_, loss_ = self.backprop(x, y)
            nabla_w = [accu + cur for accu, cur in zip(nabla_w, nabla_w_)]
            nabla_b = [accu + cur for accu, cur in zip(nabla_b, nabla_b_)]
            loss += loss_

        nabla_w = [w / len(batch_data) for w in nabla_w]
        nabla_b = [b / len(batch_data) for b in nabla_b]
        loss = loss / len(batch_data)

        for layer, w, b in zip(self.layers, nabla_w, nabla_b):
            layer.weights = layer.weights - lr * w
            layer.biases = layer.biases - lr * b

        return loss
    
    def evaluate(self, test_data):

        results = [(np.argmax(self.forward(x)[-1]), y) for x, y in test_data]
        correct = sum(int(pred == y) for pred, y in results)
        return correct
    
    def get_outputs(self, test_data):
        results = [(x, np.argmax(self.forward(x)[-1])) for x, _ in test_data]
        return results



