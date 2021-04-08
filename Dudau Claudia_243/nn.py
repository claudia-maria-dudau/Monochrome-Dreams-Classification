import numpy as np
from sklearn.model_selection import KFold
import read_data as rd
import normalization as norm

train_images, train_labels, validation_images, validation_labels, test_images, test_str = rd.read_data()
scaled_train, scaled_validation, scaled_test = norm.normalization(train_images, validation_images, test_images, "l2")

# Neural Network
class LayerdNeuralNetwork:
    def __init__(self, no_layers, no_labels):
        self.no_layers = no_layers
        self.no_labels = no_labels
        self.weights = {}
        self.bias = {}

    def initialize_parameters(self, features):
        # initializare parametrii
        # pentru fiecare layer matricea de ponderi o sa fie de forma:
        # (nr linii matricea de ponderi de pe layerul anterior / 2, nr linii matricea de ponderi de pe layerul anterior)
        # biasul o sa fie de forma:
        # (nr linii matricea de ponderi de pe lyerul curent, 1)

        np.random.seed(1)
        k = features
        for layer in range(self.no_layers - 1):
            self.weights[layer] = np.random.randn(int(k / 2) if k / 2 > self.no_labels else self.no_labels, k)
            self.bias[layer] = np.random.randn(int(k / 2) if k / 2 > self.no_labels else self.no_labels, 1)
            k = int(k / 2) if k / 2 > self.no_labels else self.no_labels
        self.weights[self.no_layers - 1] = np.random.randn(self.no_labels, k)
        self.bias[self.no_layers - 1] = np.random.randn(self.no_labels, 1)

    def sigmoid(self, weighted_data):
        # calculare functia sigmoid
        return 1 / (1 + np.exp((-1) * weighted_data))

    def sigmoid_derivative(self, weighted_data):
        # calculare derivata functiei sigmoid
        s = 1 / (1 + np.exp((-1) * weighted_data))
        return s * (1 - s)

    def sofmax(self, weighted_data):
        # calcularea probabilitatilor cu functia softmax
        probabilities = np.zeros((weighted_data.shape[1], weighted_data.shape[0]))
        for i in range(weighted_data.shape[1]):
            exp_data = np.exp(weighted_data[:, i] - np.max(weighted_data[:, i]))
            probabilities[i] = exp_data / np.sum(exp_data)
        return probabilities.T

    def cost_function(self, probabilities, labels):
        # calculare funcie de pierdere - cross-entropy
        cost = 0
        prob = probabilities.T
        for i in range(prob.shape[0]):
            cost += np.log(prob[i][labels[i]] + 1e-8)
        return (-1) * cost / prob.shape[0]

    def forward_propagation(self, input):
        # propagarea informatiei prin retea
        layer_output_history = {}
        weighted_data_history = {}

        layer_input = input.T
        layer_output_history[-1] = layer_input

        # propragare prin primele no_layers - 1 layere (sigmoid layers)
        for layer in range(self.no_layers - 1):
            weighted_input = np.dot(self.weights[layer], layer_input) + self.bias[layer]
            layer_input = self.sigmoid(weighted_input)
            layer_output_history[layer] = layer_input
            weighted_data_history[layer] = weighted_input

        # propagare prin ultimul layer (softmax layer)
        weighted_input = np.dot(self.weights[self.no_layers - 1], layer_input) + self.bias[self.no_layers - 1]
        probabilities = self.sofmax(weighted_input)
        layer_output_history[self.no_layers - 1] = probabilities
        weighted_data_history[self.no_layers - 1] = weighted_input

        return probabilities, layer_output_history, weighted_data_history

    def backward_propagation(self, data, labels, layer_output_history, weighted_data_history):
        # calcularea gradientilor pentru propagarea inapoi
        derivatives_weights = {}
        derivatives_bias = {}

        # backward propagation pentru ultimul layer (softmax layer)
        layer_input = layer_output_history[self.no_layers - 1]
        dweighted_data = layer_input - labels.T

        derivatives_weights[self.no_layers - 1] = np.dot(dweighted_data, layer_output_history[self.no_layers - 2].T) / \
                                                  data.shape[0]
        derivatives_bias[self.no_layers - 1] = np.sum(dweighted_data, axis=1, keepdims=True) / data.shape[0]
        dlayer_input_prev = np.dot(self.weights[self.no_layers - 1].T, dweighted_data)

        # backward propagation pentru urmatoarele no_layers - 1 layere (sigmoid layers)
        for layer in range(self.no_layers - 2, -1, -1):
            dweighted_data = dlayer_input_prev * self.sigmoid_derivative(weighted_data_history[layer])
            derivatives_weights[layer] = np.dot(dweighted_data, layer_output_history[layer - 1].T) / data.shape[0]
            derivatives_bias[layer] = np.sum(dweighted_data, axis=1, keepdims=True) / data.shape[0]
            dlayer_input_prev = np.dot(self.weights[layer].T, dweighted_data)

        return derivatives_weights, derivatives_bias

    def train(self, train_data, train_labels, no_iterations, learning_rate):
        # antrenare model
        # self.initialize_parameters(train_data.shape[1])
        labels = np.zeros((train_labels.shape[0], self.no_labels))
        for i in range(train_labels.shape[0]):
            labels[i][train_labels[i]] = 1

        for i in range(no_iterations):
            # print(i)

            # forward propagation
            probabilities, layer_output_history, weighted_data_history = self.forward_propagation(train_data)

            # backward propagation
            derivatives_weights, derivatives_bias = self.backward_propagation(train_data, labels, layer_output_history,
                                                                              weighted_data_history)
            cost = self.cost_function(probabilities, train_labels)

            # update parameters
            for layer in range(self.no_layers):
                self.weights[layer] -= learning_rate * derivatives_weights[layer]
                self.bias[layer] -= learning_rate * derivatives_bias[layer]

            if i % 10 == 0 or i == no_iterations - 1:
                g = open('./scores.txt', "a")
                g.write("---- {} ----\n".format(i))

                pred = self.predict(train_data)
                acc = (pred == train_labels).mean() * 100
                g.write("train: {}\n".format(acc))

                pred = self.predict(scaled_validation)
                acc = (pred == validation_labels).mean() * 100
                g.write("validation: {}\n".format(acc))

                g.write("loss: {}\n\n".format(cost))
                g.close()

    def predict(self, test_data):
        # determinare predictii
        probabilities, layer_output_history, weighted_data_history = self.forward_propagation(test_data)
        predictions = np.zeros(test_data.shape[0], 'int')
        for i in range(test_data.shape[0]):
            predictions[i] = np.argmax(probabilities[:, i])
        return predictions


print('init')
LNN = LayerdNeuralNetwork(2, 9)
LNN.initialize_parameters(scaled_train.shape[1])

print('train')
g = open('./scores.txt', "w")

# impartire date in mai multe batch-uri
kf = KFold(n_splits=10, shuffle=True)
i = 0
for ceva, tr in kf.split(scaled_train):
    print(i)
    LNN.train(scaled_train[tr], train_labels[tr], 200, 0.1)
    i += 1
g.close()

predictions = LNN.predict(scaled_test)